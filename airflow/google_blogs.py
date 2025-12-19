import os
from datetime import datetime

from airflow.sdk import dag, task, task_group

# Define the absolute path to your project and virtual environment
PROJECT_FOLDER = "/home/main-server/Github/data-playground-base/google_blogs"
VENV_PYTHON = f"{PROJECT_FOLDER}/venv/bin/python"

@dag(
    schedule=None, 
    start_date=datetime(2025, 1, 1), 
    catchup=False,
    # default_args applies these settings to EVERY task in the DAG
    default_args={
        "python": VENV_PYTHON, 
        "env": {"PYTHONPATH": PROJECT_FOLDER} # Tells the venv where your code lives
    }
)
def google_blogs():

    @task.external_python(task_id="proc_start")
    def create_data():
        import data_gathering  # Now works automatically via PYTHONPATH
        return data_gathering.init()
    
    @task.external_python(task_id="get_google_keys")
    def get_website_keys(init_data):
        # Filter for Google blogs
        return [name for name in init_data['WEBSITES'].keys() 
                if name.startswith("GOOGLE") and name != 'GOOGLE_DEVS_SITEMAP']

    @task.external_python(
        task_id="gather_individual_blog", 
        map_index_template="{{ name }}"
    )
    def use_data(init, name):
        import data_gathering

        url = init['WEBSITES'][name]

        if name == "GOOGLE_WORKSPACE_BLOG":
            return data_gathering.workspace_data_get(init)
        
        # Determine if we need to enrich the data
        enrich = name in ["GOOGLE_APPS_UPDATES", "GOOGLE_TECHNOLOGY_BLOG"]
        return data_gathering.fetch_and_parse_feed(init, url, name, enrich=enrich)

    @task_group(group_id="google_data")
    def gather_google_data(names_list, init_obj):
        # use_data will inherit the python path and env from default_args
        use_data.partial(init=init_obj).expand(name=names_list)

    # --- Execution Flow ---
    init_obj = create_data()
    site_names = get_website_keys(init_obj)
    
    # Trigger the mapped task group
    gather_google_data(site_names, init_obj)

# Initialize the DAG
google_blogs()
import os
import sys
from datetime import datetime

from airflow.sdk import dag, task, task_group

# Define the path you want to add
# Path for local development
# folder_to_add = r"C:\Users\Llubr\Desktop\Github\data-playground-base\google_blogs"
# Path for airflow
folder_to_add = "/home/main-server/Github/data-playground-base/google_blogs"

# Path to Pyhton virtual environment
PATH_TO_PYTHON_BINARY = f"{folder_to_add}/venv/bin/python"

# Add folder to system path
# sys.path.append(os.path.abspath(folder_to_add))

@dag(
    schedule=None, 
    start_date=datetime(2025, 1, 1), 
    catchup=False, 
    default_args={
        "python": PATH_TO_PYTHON_BINARY, 
        "env": {"PYTHONPATH": folder_to_add}
    }
)

def google_blogs():

    @task.external_python(task_id="proc_start", python=PATH_TO_PYTHON_BINARY, 
    default_args={
        "python": PATH_TO_PYTHON_BINARY, 
        "env": {"PYTHONPATH": folder_to_add}
    })
    def create_data():
        # Import process script
        import data_gathering

        # Return a simple dictionary (always serializable)
        return data_gathering.init()
    
    @task(task_id="get_google_keys")
    def get_website_keys(init_data):
        return [name for name in init_data['WEBSITES'].keys() 
                if name.startswith("GOOGLE") and name != 'GOOGLE_DEVS_SITEMAP']

    @task.external_python(task_id = "gather_individual_blog", map_index_template="{{ name }}", python=PATH_TO_PYTHON_BINARY, 
    default_args={
        "python": PATH_TO_PYTHON_BINARY, 
        "env": {"PYTHONPATH": folder_to_add}
    })
    def use_data(init, name):
        # Import process script
        import data_gathering

        url = init['WEBSITES'][name]

        # Pass the data into your logic function
        if name == "GOOGLE_WORKSPACE_BLOG":
            return data_gathering.workspace_data_get(init)
        elif name in ["GOOGLE_APPS_UPDATES", "GOOGLE_TECHNOLOGY_BLOG"]:
            return data_gathering.fetch_and_parse_feed(init, url, name, enrich=True)

        return data_gathering.fetch_and_parse_feed(init, url, name, enrich=False)

    @task_group(group_id="google_data")
    def gather_google_data(names_list, init_obj):
        # DYNAMIC MAPPING: Creates one 'use_data' task for every name in site_names
        use_data.partial(init=init_obj).expand(name=names_list)


    # Execution Flow
    init_obj = create_data()
    site_names = get_website_keys(init_obj)
    
    gather_google_data(site_names, init_obj)


google_blogs()
import os
from datetime import datetime

from airflow.sdk import dag, task, task_group

# --- CONFIGURATION ---
# Ensure these paths are absolute and accessible by the Airflow worker
PROJECT_ROOT = "/home/main-server/Github/data-playground-base/google_blogs"
PYTHON_BIN = f"{PROJECT_ROOT}/venv/bin/python"

@dag(
    dag_id="google_blogs",
    schedule=None, 
    start_date=datetime(2025, 1, 1), 
    catchup=False,
    # env in default_args ensures every task gets the PYTHONPATH
    default_args={
        "env": {"PYTHONPATH": PROJECT_ROOT}
    }
)
def google_blogs():

    # Task 1: Initialize data
    # We explicitly define the 'python' path to satisfy Airflow 3 requirements
    @task.external_python(python=PYTHON_BIN, task_id="proc_start")
    def create_data():
        import data_gathering
        # init() must return a JSON-serializable object (dict/list/str/int)
        return data_gathering.init()
    
    # Task 2: Filter website keys
    @task.external_python(python=PYTHON_BIN, task_id="get_google_keys")
    def get_website_keys(init_data: dict):
        return [
            name for name in init_data['WEBSITES'].keys() 
            if name.startswith("GOOGLE") and name != 'GOOGLE_DEVS_SITEMAP'
        ]

    # Task 3: The actual processing task (to be mapped)
    @task.external_python(
        python=PYTHON_BIN, 
        task_id="gather_individual_blog",
        map_index_template="{{ name }}"
    )
    def use_data(init: dict, name: str):
        import data_gathering

        url = init['WEBSITES'][name]

        if name == "GOOGLE_WORKSPACE_BLOG":
            return data_gathering.workspace_data_get(init)
        
        # Enriched feed logic
        enrich = name in ["GOOGLE_APPS_UPDATES", "GOOGLE_TECHNOLOGY_BLOG"]
        return data_gathering.fetch_and_parse_feed(init, url, name, enrich=enrich)

    # Task Group for Dynamic Mapping
    @task_group(group_id="google_data_group")
    def gather_google_data_group(names_list, init_obj):
        # .partial defines the constant arguments
        # .expand defines the iterable to map over
        use_data.partial(init=init_obj).expand(name=names_list)

    # --- DAG FLOW EXECUTION ---
    # 1. Start processing and get init object
    init_obj = create_data()
    
    # 2. Extract keys from that object
    site_names = get_website_keys(init_obj)
    
    # 3. Pass keys and init object to the task group for mapping
    gather_google_data_group(site_names, init_obj)

# Register the DAG
google_blogs()
import os
from datetime import datetime

from airflow.sdk import dag, task, task_group

# --- CONFIGURATION ---
# Ensure these paths are absolute and accessible by the Airflow worker
PROJECT_ROOT = "/home/main-server/Github/data-playground-base/google_blogs"
PYTHON_BIN = f"{PROJECT_ROOT}/venv/bin/python"

# Update this path to point to your actual Google Service Account JSON key
# This is required to solve the DefaultCredentialsError
GOOGLE_KEY_PATH = "/home/main-server/keys/impactful-post-292301-ea5136b0da63.json"

@dag(
    dag_id="google_blogs",
    schedule=None, 
    start_date=datetime(2025, 1, 1), 
    catchup=False,
    # Adding the credentials path to the environment of the external processes
    default_args={
        "env": {
            "PYTHONPATH": PROJECT_ROOT,
            "GOOGLE_APPLICATION_CREDENTIALS": GOOGLE_KEY_PATH
        }
    }
)
def google_blogs():

    # Task 1: Initialize data
    @task.external_python(python=PYTHON_BIN, task_id="proc_start")
    def create_data(root_path: str):
        import sys
        # Manual path injection to ensure the module is found
        if root_path not in sys.path:
            sys.path.append(root_path)
            
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
    def use_data(init: dict, name: str, root_path: str):
        from airflow.sdk import get_current_context

        context = get_current_context()
        context["name"] = name
        
        import sys
        if root_path not in sys.path:
            sys.path.append(root_path)
            
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
        # We pass PROJECT_ROOT as a constant via .partial()
        use_data.partial(init=init_obj, root_path=PROJECT_ROOT).expand(name=names_list)

    # --- DAG FLOW EXECUTION ---
    # 1. Start processing and get init object - passing PROJECT_ROOT explicitly
    init_obj = create_data(PROJECT_ROOT)
    
    # 2. Extract keys from that object
    site_names = get_website_keys(init_obj)
    
    # 3. Pass keys and init object to the task group for mapping
    gather_google_data_group(site_names, init_obj)

# Register the DAG
google_blogs()
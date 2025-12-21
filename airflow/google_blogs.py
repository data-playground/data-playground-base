import os
from datetime import datetime

from airflow.sdk import dag, task, task_group

# Setting root folder where the project lives
PROJECT_ROOT = "/home/main-server/Github/data-playground-base/google_blogs"

# Setting location for the Python executable inside the Python virtual environment built for this process
PYTHON_BIN = f"{PROJECT_ROOT}/venv/bin/python"

# Defining the paht for the Google Service Account
GOOGLE_KEY_PATH = "/home/main-server/keys/impactful-post-292301-ea5136b0da63.json"


@dag(
    dag_id="google_blogs",
	schedule='@daily',
    start_date=datetime(2025, 1, 1), 
    catchup=False,
    # The env key in the default_args allow the DAG to use the variables defined above
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
        """Build the dictionary that will be used in downstream tasks"""

        # Manual path injection to ensure the module is found
        import sys
        if root_path not in sys.path:
            sys.path.append(root_path)
        
        # Importing the module
        import data_gathering

        # Returning the result from the init function, which provides all the necessary start data for the process
        return data_gathering.init()
    
    # Task 2: Filter website keys to only use Google
    @task.external_python(python=PYTHON_BIN, task_id="get_google_keys")
    def get_website_keys(init_data: dict):
        """Filter to Google sites that will be scraped by downstream tasks"""
        return [
            name for name in init_data['WEBSITES'].keys() 
            if name.startswith("GOOGLE") and name != 'GOOGLE_DEVS_SITEMAP'
        ]

    # Task 3: The actual processing task (to be mapped in the task group)
    @task.external_python(
        python=PYTHON_BIN, 
        task_id="gather_individual_blog"
        # map_index_template="{{ map_value }}" 
    )
    def use_data(init: dict, name: str, root_path: str):
        """Perfom the scraping process (and enriching, if selected) for one website"""

        # Manual path injection to ensure the module is found
        import sys
        if root_path not in sys.path:
            sys.path.append(root_path)
            
        # Importing the module
        import data_gathering

        # Get URL for the selected website based on the name
        url = init['WEBSITES'][name]

        # Perform process for GOOGLE_WORKSPACE_BLOG, since it is a separate function
        if name == "GOOGLE_WORKSPACE_BLOG":
            return data_gathering.workspace_data_get(init)
        
        # Define if the enrich process should be run
        enrich = name in ["GOOGLE_APPS_UPDATES", "GOOGLE_TECHNOLOGY_BLOG"]

        # Perform process for all other selected websites
        return data_gathering.fetch_and_parse_feed(init, url, name, enrich=enrich)

    # Task Group for Dynamic Mapping
    @task_group(group_id="google_data_group")
    def gather_google_data_group(names_list, init_obj):
        """This task group will map data from the init_obj (from Task 1) to the use_data function for each website"""
        # Map the filtered websites for the function for each website
        use_data.partial(init=init_obj, root_path=PROJECT_ROOT).expand(name=names_list)

    # Task 4: Update the submodule in the main GitHub repo
    @task.external_python(python=PYTHON_BIN, task_id="update_submodule")
    def update_submodule(init: dict, root_path: str):
        # Manual path injection to ensure the module is found
        import sys
        if root_path not in sys.path:
            sys.path.append(root_path)

        # Importing the module
        import data_gathering

        # Run process to update the submodule
        return data_gathering.update_submodule_direct(init)

    # --- DAG FLOW EXECUTION ---
    # 1. Start processing and get init object - passing PROJECT_ROOT explicitly
    init_obj = create_data(PROJECT_ROOT)
    
    # 2. Extract keys from that object
    site_names = get_website_keys(init_obj)
    
    # 3. Pass keys and init object to the task group for mapping
    group_output = gather_google_data_group(site_names, init_obj)

    # 4. Update GitHub submodule in main repository
    group_output >> update_submodule(init=init_obj, root_path=PROJECT_ROOT)

# Register the DAG
google_blogs()
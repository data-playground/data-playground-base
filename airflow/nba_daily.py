import os
from datetime import datetime

from airflow.sdk import dag, task, task_group

# Setting root folder where the project lives
PROJECT_ROOT = "/home/main-server/Github/data-playground-base/nba"

# Setting location for the Python executable inside the Python virtual environment built for this process
PYTHON_BIN = f"{PROJECT_ROOT}/venv/bin/python"

# Defining the paht for the Google Service Account
GOOGLE_KEY_PATH = "/home/main-server/keys/impactful-post-292301-ea5136b0da63.json"

# Common environment for all external tasks
TASK_ENV = {
    "PYTHONPATH": PROJECT_ROOT,
    "GOOGLE_APPLICATION_CREDENTIALS": GOOGLE_KEY_PATH
}

@dag(
    dag_id="nba_daily",
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
def nba_daily_proc():

    # Task 1: Initialize data
    @task.external_python(python=PYTHON_BIN, task_id="proc_start")
    def create_data(root_path: str):
        """Build the dictionary that will be used in downstream tasks"""

        # Manual path injection to ensure the module is found
        import sys
        if root_path not in sys.path:
            sys.path.append(root_path)

        print(sys.path)
        
        # Importing the module
        import nba_game_summary

        # Returning the result from the init function, which provides all the necessary start data for the process
        return nba_game_summary.NBA()

    @task.external_python(python=PYTHON_BIN, task_id="test_task")
    def simple_print(nba):
        """Simple print test from the class defined above"""

        print(f"NBA game date is defined as: {nba.GAME_DATE}")
    
    # --- DAG FLOW EXECUTION ---
    # 1. Start processing and get init object - passing PROJECT_ROOT explicitly
    nba_instance = create_data(PROJECT_ROOT)
    simple_print(nba_instance)

# Register the DAG
nba_daily_proc()
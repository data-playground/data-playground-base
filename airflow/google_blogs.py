import os
import sys
from datetime import datetime

from airflow.decorators import dag, task, task_group

# Define the path you want to add
# Path for local development
# folder_to_add = r"C:\Users\Llubr\Desktop\Github\data-playground-base\google_blogs"
# Path for airflow
folder_to_add = "/home/main-server/Github/data-playground-base/google_blogs"

# Path to Pyhton virtual environment
PATH_TO_PYTHON_BINARY = f"{folder_to_add}/venv/bin/activate"

# Add folder to system path
sys.path.append(os.path.abspath(folder_to_add))

@dag(schedule=None, start_date=datetime(2025, 1, 1), catchup=False, default_args={"python": PATH_TO_PYTHON_BINARY})
def google_blogs():

    @task(task_id="proc_start")
    def create_data():
        # Import process script
        import data_gathering

        # Return a simple dictionary (always serializable)
        return data_gathering.init()

    @task_group(group_id="google_data")
    def gather_google_data(init):
        for name, url in init['WEBSITES'].items():
            if name.startswith("GOOGLE") and name not in ['GOOGLE_DEVS_SITEMAP']:
                @task(task_id=f"gather_{name}")
                def use_data(init, url, name):
                    # Import process script
                    import data_gathering

                    # Pass the data into your logic function
                    if name == "GOOGLE_WORKSPACE_BLOG":
                        data_added = data_gathering.workspace_data_get(init)
                    elif name in ["GOOGLE_APPS_UPDATES", "GOOGLE_TECHNOLOGY_BLOG"]:
                        data_added = data_gathering.fetch_and_parse_feed(init, url, name, enrich=True)
                    else:
                        data_added = data_gathering.fetch_and_parse_feed(init, url, name, enrich=False)

                    return data_added

    # Flow: Data is passed via XCom automatically
    blog_sites_init = create_data()
    gather_google_data(blog_sites_init)

google_blogs()
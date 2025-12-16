from airflow import DAG
from airflow.sdk import task
from datetime import datetime

# Import your function if it's in a separate file
# from my_script import my_python_task_function

@task
def my_decorated_python_task_function(some_argument):
	print(f"Executing decorated Python task with argument: {some_argument}")
	return "Decorated task completed! Testing new file"

with DAG(
	dag_id='python_script_example',
	start_date=datetime(2023, 1, 1),
	schedule='@daily',
	catchup=False,
) as dag:
	decorated_task = my_decorated_python_task_function(some_argument='hello_from_decorator')
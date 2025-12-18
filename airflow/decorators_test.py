from datetime import datetime

from airflow.decorators import dag, task


class MyClass:
    def __init__(self, value):
        self.value = value

    def get_value(self):
        return self.value * 2

@dag(schedule_interval=None, start_date=datetime(2023, 1, 1), catchup=False)
def class_instance_dag():

    @task(task_id='class_instance')
    def create_instance():
        # Instantiate the class and return it (auto-pushed to XCom)
        instance = MyClass(value=10)
        return instance

    @task(task_id="function_run")
    def use_instance(instance_from_upstream):
        # The instance is automatically pulled and passed as an argument
        # Note: The object is deserialized here
        result = instance_from_upstream.get_value()
        print(f"The result is: {result}")

    # Define the workflow
    instance_obj = create_instance()
    use_instance(instance_obj)

decorator_test = class_instance_dag()
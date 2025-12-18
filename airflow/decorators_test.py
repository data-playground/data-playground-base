from dataclasses import dataclass
from datetime import datetime

from airflow.decorators import dag, task


@dataclass
class MyClass:
    value: int
    
    def get_value(self):
        return self.value * 2
    
@dag(schedule=None, start_date=datetime(2023, 1, 1), catchup=False)
def class_instance_dag():
    @task
    def create_instance():
        # Airflow automatically serializes @dataclass objects
        return MyClass(value=10)

    @task
    def use_instance(instance: MyClass):
        # Airflow automatically deserializes it back into a MyClass object
        print(f"The result is: {instance.get_value()}")

    use_instance(create_instance())

class_instance_dag()
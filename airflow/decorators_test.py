from datetime import datetime

from airflow.decorators import dag, task


# 1. Define logic as standalone functions
def calculate_logic(value):
    return value * 2

@dag(schedule=None, start_date=datetime(2025, 1, 1), catchup=False)
def functional_dag():

    @task
    def create_data():
        # Return a simple dictionary (always serializable)
        return {"value": 10}

    @task
    def use_data(data_dict):
        # Pass the data into your logic function
        val = data_dict["value"]
        result = calculate_logic(val)
        print(f"The result is: {result}")

    # Flow: Data is passed via XCom automatically
    use_data(create_data())

functional_dag()
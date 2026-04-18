# secrets.py
"""
Lightweight GCP Secret Manager helper.
Intentionally has zero FastAPI/SQLAlchemy dependencies so it can be
imported safely by both the FastAPI app and Airflow DAGs.
"""
from google.cloud import secretmanager


def get_key(secret_name: str) -> str:
    client = secretmanager.SecretManagerServiceClient()
    project_id = "impactful-post-292301"
    name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")
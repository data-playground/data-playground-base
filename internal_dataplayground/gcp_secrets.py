# secrets.py
"""
Lightweight GCP Secret Manager helper.
Intentionally has zero FastAPI/SQLAlchemy dependencies so it can be
imported safely by both the FastAPI app and Airflow DAGs.
"""
from google.cloud import secretmanager
import os

def get_key(secret_name: str) -> str:
    # Check environment variable first (faster, no GCP call)
    # Env var name: replace hyphens with underscores, uppercase
    # e.g. "TMDB-API-Key" → "TMDB_API_KEY"
    env_key = secret_name.upper().replace("-", "_")
    env_val = os.environ.get(env_key)
    if env_val:
        return env_val

    # Fall back to GCP Secret Manager
    client = secretmanager.SecretManagerServiceClient()
    project_id = "impactful-post-292301"
    name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")
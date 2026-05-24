# ml_service/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install only what sentence-transformers needs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Model will be downloaded here on first startup and cached across restarts
ENV MODEL_CACHE_DIR=/app/model_cache
VOLUME ["/app/model_cache"]

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]

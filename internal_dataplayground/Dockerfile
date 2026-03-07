# 1. Start with a "Base Image" (A tiny version of Linux with Python already installed)
FROM python:3.12.3-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies (needed for MariaDB/MySQL drivers)
RUN apt-get update && apt-get install -y \
    build-essential \
    libmariadb-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your project files
COPY . .

# Expose FastAPI's default port
EXPOSE 8000

# Start Uvicorn
CMD ["uvicorn", "main.py:app", "--host", "0.0.0.0", "--port", "8000"]
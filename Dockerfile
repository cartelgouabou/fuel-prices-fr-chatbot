# Dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --fix-missing \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Add this line:
ENV PYTHONPATH=/app

CMD ["python", "etl/run_etl.py"]

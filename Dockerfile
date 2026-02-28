FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# NOTE : Retirer "python load_data.py &&" si la DB est pré-remplie par les organisateurs
CMD ["sh", "-c", "python load_data.py && uvicorn main:app --host 0.0.0.0 --port 8000"]

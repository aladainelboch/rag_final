FROM python:3.11-slim

WORKDIR /app

# Dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev && rm -rf /var/lib/apt/lists/*

# Dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code applicatif
COPY . .

# Démarrage : charger les données puis lancer l'API
# NOTE : Si la DB est pré-remplie par les organisateurs, retirer "python load_data.py &&"
CMD ["sh", "-c", "python load_data.py && uvicorn main:app --host 0.0.0.0 --port 8000"]

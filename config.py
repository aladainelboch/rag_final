import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     int(os.getenv("DB_PORT", 5432)),
    "dbname":   os.getenv("DB_NAME", "rag_boulangerie"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
}

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Modèle imposé – dim 384
TOP_K = 3                               # Nombre de résultats à retourner

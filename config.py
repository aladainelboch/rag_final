import os, urllib.parse
from dotenv import load_dotenv
load_dotenv()

database_url = os.getenv("DATABASE_URL")
if database_url:
    r = urllib.parse.urlparse(database_url)
    DB_CONFIG = {
        "host":     r.hostname,
        "port":     r.port or 5432,
        "dbname":   r.path.lstrip("/"),
        "user":     r.username,
        "password": r.password,
    }
else:
    DB_CONFIG = {
        "host":     os.getenv("DB_HOST", "localhost"),
        "port":     int(os.getenv("DB_PORT", 5432)),
        "dbname":   os.getenv("DB_NAME", "rag_boulangerie"),
        "user":     os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "postgres"),
    }

EMBEDDING_MODEL  = "all-MiniLM-L6-v2"   # Modèle imposé — dim 384
TOP_K            = 3                      # Nombre de résultats
TRANSLATE_QUERY  = os.getenv("TRANSLATE_QUERY", "false").lower() == "true"

# 🍞 RAG Boulangerie — Recherche Sémantique
**Équipe :** Jeuudi | **Organisation :** STE AGRO MELANGE TECHNOLOGIE — ROSE BLANCHE Group | ai night RAG challenge 2026

---

## 🚀 Lancement en une commande (Docker)

```bash
docker-compose up --build
```
Puis ouvrir → **http://localhost:8000**

---

## 🗂️ Structure du Projet

```
rag_final/
├── main.py                  # FastAPI backend (API + UI)
├── load_data.py             # Ingestion fragments → PostgreSQL
├── config.py                # Paramètres DB + modèle
├── docker-compose.yml       # Orchestration DB + API
├── Dockerfile               # Image Docker Python
├── init.sql                 # Init PostgreSQL (pgvector + table)
├── requirements.txt         # Dépendances Python
├── .env.example             # Template credentials
├── templates/index.html     # Interface web
├── static/                  # Fichiers statiques
└── README.md
```

---

## ⚙️ Option A — Docker (RECOMMANDÉ, zéro configuration)

**Prérequis :** Docker Desktop installé → https://www.docker.com/products/docker-desktop

```bash
# 1. Dézipper
unzip rag_final.zip && cd rag_final

# 2. Lancer (DB + API + chargement données automatique)
docker-compose up --build

# 3. Ouvrir dans le navigateur
#    Interface web  → http://localhost:8000
#    Swagger API    → http://localhost:8000/docs
#    Health check   → http://localhost:8000/health

# Arrêter
docker-compose down

# Reset complet (vide la DB)
docker-compose down -v && docker-compose up --build
```

---

## ⚙️ Option B — Python local (sans Docker)

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Configurer
cp .env.example .env
# Éditer .env avec vos credentials PostgreSQL

# 3. Préparer la DB (si elle n'est pas déjà remplie)
python load_data.py

# 4. Démarrer l'API
uvicorn main:app --reload --port 8000
```

### Prérequis PostgreSQL local (Ubuntu/Debian)
```bash
sudo apt install postgresql-server-dev-14
git clone https://github.com/pgvector/pgvector.git
cd pgvector && make && sudo make install
```
```sql
CREATE DATABASE rag_boulangerie;
\c rag_boulangerie
CREATE EXTENSION vector;
CREATE TABLE embeddings (id SERIAL PRIMARY KEY, id_document INT, texte_fragment TEXT, vecteur VECTOR(384));
```

---

## ☁️ Option C — Base cloud Supabase (gratuit)

1. Créer un projet sur https://supabase.com
2. Dans SQL Editor :
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   CREATE TABLE embeddings (id SERIAL PRIMARY KEY, id_document INT, texte_fragment TEXT, vecteur VECTOR(384));
   ```
3. Settings > Database → copier les credentials dans `.env`
4. Lancer : `uvicorn main:app --reload`

---

## 🏆 Connexion à la DB des organisateurs

Si les organisateurs fournissent une DB pré-remplie, mettre à jour `.env` :
```env
DB_HOST=<host_organisateurs>
DB_PORT=5432
DB_NAME=<db_name>
DB_USER=<user>
DB_PASSWORD=<password>
```
Le script `load_data.py` ne touchera pas aux données existantes (vérification automatique).

---

## 📡 API Endpoints

| Méthode | URL | Description |
|---|---|---|
| GET | `/` | Interface web |
| POST | `/search` | Recherche sémantique JSON |
| GET | `/health` | Santé API + nombre de fragments |
| GET | `/docs` | Swagger UI |
| GET | `/redoc` | ReDoc |

### Exemple curl
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"question": "Dosage alpha-amylase fermentation", "top_k": 3}'
```

---

## 🔧 Paramètres Techniques

| Paramètre | Valeur |
|---|---|
| Modèle | all-MiniLM-L6-v2 |
| Dimension vecteur | 384 |
| Table PostgreSQL | embeddings (id, id_document, texte_fragment, vecteur VECTOR(384)) |
| Similarité | Cosinus |
| Top K | 3 (configurable 1-10 via UI) |
| Backend | FastAPI + Uvicorn |
| Base de données | PostgreSQL + pgvector |

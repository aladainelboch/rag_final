# RAG Boulangerie & Pâtisserie
**Équipe Jeuudiddy** 
AI Night RAG Challenge 2026

---

## Description

Module de recherche sémantique intelligent pour l'assistance à la formulation en boulangerie et pâtisserie.  
Interroge une base vectorielle PostgreSQL via similarité cosinus pour retourner les 3 fragments les plus pertinents.

| Paramètre | Valeur |
|---|---|
| Modèle d'embedding | `all-MiniLM-L6-v2` |
| Dimension | 384 |
| Similarité | Cosinus (pgvector natif + fallback Python) |
| Top K | 3 |
| Backend | FastAPI + PostgreSQL + pgvector |

---

## Structure du projet

```
rag_final/
├── pdf_files/              ← Déposer tous les PDFs ici
├── templates/
│   └── index.html          ← Interface web
├── static/                 ← Fichiers statiques
├── main.py                 ← FastAPI API + recherche sémantique
├── load_data.py            ← Ingestion PDF → PostgreSQL
├── config.py               ← Configuration DB + modèle
├── init.sql                ← Schéma PostgreSQL (auto-exécuté)
├── requirements.txt        ← Dépendances Python
├── Dockerfile              ← Image Docker
├── docker-compose.yml      ← Orchestration DB + API
└── .env                    ← Variables d'environnement (local)
```

---

## Démarrage rapide

### Prérequis
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installé et lancé

### 1 — Cloner / décompresser le projet
```bash
cd rag_final
```

### 2 — Ajouter les PDFs
Copier tous les fichiers PDF dans le dossier `pdf_files/` :
```
rag_final/pdf_files/
    ├── BVZyme-TDS-AF110.pdf
    ├── BVZyme-TDS-AF330.pdf
    └── ... (tous les PDFs)
```

### 3 — Créer le volume cache (une seule fois)
```bash
docker volume create rag_final_model_cache
```

### 4 — Lancer
```bash
docker-compose up --build
```

### 5 — Accéder à l'application
| URL | Description |
|---|---|
| http://localhost:8000 | Interface web |
| http://localhost:8000/docs | Swagger API |
| http://localhost:8000/health | Statut + stats |
| http://localhost:8000/redoc | ReDoc |

---

## Commandes utiles

```bash
# Voir les logs
docker logs rag_api

# Relancer l'ingestion des PDFs sans rebuild
docker exec -it rag_api python load_data.py

# Vider la base et ré-ingérer
docker exec -it rag_db psql -U postgres -d rag_boulangerie \
  -c "TRUNCATE TABLE embeddings RESTART IDENTITY; TRUNCATE TABLE documents RESTART IDENTITY CASCADE;"
docker exec -it rag_api python load_data.py

# Voir les fragments en base
docker exec -it rag_db psql -U postgres -d rag_boulangerie \
  -c "SELECT id, LEFT(texte_fragment, 80) FROM embeddings LIMIT 10;"

# Redémarrer sans rebuild (rapide)
docker-compose down && docker-compose up

# Rebuild avec nouvelles dépendances
docker-compose down && docker-compose up --build
```

---

## Variables d'environnement

| Variable | Défaut | Description |
|---|---|---|
| `DB_HOST` | `db` | Hôte PostgreSQL |
| `DB_PORT` | `5432` | Port PostgreSQL |
| `DB_NAME` | `rag_boulangerie` | Nom de la base |
| `DB_USER` | `postgres` | Utilisateur |
| `DB_PASSWORD` | `postgres` | Mot de passe |
| `DATABASE_URL` | — | URL complète (Railway) |
| `TRANSLATE_QUERY` | `false` | Traduire FR→EN avant embedding |

---

## Architecture

```
[Navigateur]
     │  HTTP
     ▼
[FastAPI :8000]
     │  1. Encode question (all-MiniLM-L6-v2, dim=384)
     │  2. Cosine similarity via pgvector (<=>)
     │  3. Retourne Top 3 fragments
     ▼
[PostgreSQL + pgvector :5432]
     │  Table: embeddings (id, document_id, texte_fragment, vecteur VECTOR(384))
     │  Table: documents  (id, filename, checksum)
     │  Index: IVFFlat cosine
```

---

## Exemple de résultat

**Question :** *Améliorant de panification : quelles sont les quantités recommandées d'alpha-amylase, xylanase et d'Acide ascorbique ?*

```
Résultat 1
Texte : "Dosage recommandé : 0.005% à 0.02% du poids de farine."
Score : 0.91

Résultat 2
Texte : "Alpha-amylase : utilisation entre 5 et 20 ppm selon la farine."
Score : 0.87

Résultat 3
Texte : "Xylanase : améliore l'extensibilité de la pâte…"
Score : 0.82
```

---

## Équipe

**Jeuudiddy**
AI Night RAG Challenge · Polytech Sousse, 2026

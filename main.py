#!/usr/bin/env python3
"""
main.py — FastAPI Backend RAG Boulangerie & Pâtisserie
Équipe : Jeuudi | STE AGRO MELANGE TECHNOLOGIE — ROSE BLANCHE Group
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer
from config import DB_CONFIG, EMBEDDING_MODEL, TOP_K

app = FastAPI(
    title="RAG Boulangerie & Pâtisserie",
    description="Modèle : all-MiniLM-L6-v2 | Similarité : Cosinus | Top K = 3",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

print(f"⚙️  Chargement du modèle : {EMBEDDING_MODEL} ...")
model = SentenceTransformer(EMBEDDING_MODEL)
print(f"✅ Modèle prêt — dimension : {model.get_sentence_embedding_dimension()}")


class SearchRequest(BaseModel):
    question: str
    top_k: int = TOP_K


class FragmentResult(BaseModel):
    rank: int
    texte_fragment: str
    score: float
    id: int
    id_document: int
    source_file: str | None = None


class SearchResponse(BaseModel):
    question: str
    results: list[FragmentResult]
    total_fragments_searched: int


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM embeddings;")
        frag_count = cur.fetchone()[0]
        # Count documents if table exists
        try:
            cur.execute("SELECT COUNT(*) FROM documents;")
            doc_count = cur.fetchone()[0]
        except Exception:
            doc_count = None
        cur.close()
        conn.close()
        return {
            "status": "ok",
            "model": EMBEDDING_MODEL,
            "fragments_in_db": frag_count,
            "documents_in_db": doc_count,
            "top_k": TOP_K,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Erreur DB : {str(e)}")


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide.")

    top_k = max(1, min(req.top_k, 20))

    # Étape 1 — Embedding de la question
    q_vec = model.encode(req.question, normalize_embeddings=True).astype(np.float32)

    # Étape 2 — Récupération des fragments + filename via JOIN
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT e.id, e.document_id, e.texte_fragment,
                       e.vecteur::text, d.filename AS source_file
                FROM embeddings e
                LEFT JOIN documents d ON d.id = e.document_id;
            """)
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        raise HTTPException(status_code=404, detail="La base de données est vide.")

    # Étape 3 — Similarité cosinus
    scored = []
    for row in rows:
        vec = np.array(
            [float(v) for v in row["vecteur"].strip("[]").split(",")],
            dtype=np.float32
        )
        scored.append({
            "id":             row["id"],
            "id_document":    row["document_id"],
            "texte_fragment": row["texte_fragment"],
            "source_file":    row["source_file"],
            "score":          cosine_similarity(q_vec, vec),
        })

    # Étape 4 — Tri décroissant
    scored.sort(key=lambda x: x["score"], reverse=True)

    # Étape 5 — Top K
    results = [
        FragmentResult(
            rank=i + 1,
            texte_fragment=item["texte_fragment"],
            score=round(item["score"], 4),
            id=item["id"],
            id_document=item["id_document"],
            source_file=item["source_file"],
        )
        for i, item in enumerate(scored[:top_k])
    ]

    return SearchResponse(
        question=req.question,
        results=results,
        total_fragments_searched=len(scored),
    )

#!/usr/bin/env python3
"""
main.py — FastAPI Backend RAG Boulangerie & Pâtisserie
Équipe : Jeuudiddy | STE AGRO MELANGE TECHNOLOGIE — ROSE BLANCHE Group

Features:
  - Auto-détection du schéma PostgreSQL (document_id / id_document)
  - Similarité cosinus via pgvector natif (fast) avec fallback Python
  - Traduction optionnelle FR→EN (TRANSLATE_QUERY=true)
  - Compatible DB organisateurs + DB locale
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
from config import DB_CONFIG, EMBEDDING_MODEL, TOP_K, TRANSLATE_QUERY

app = FastAPI(
    title="RAG Boulangerie & Pâtisserie — Jeuudiddy",
    description="Modèle : all-MiniLM-L6-v2 | Similarité : Cosinus | Top K = 3",
    version="2.0.0",
)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

print(f"⚙️  Chargement du modèle : {EMBEDDING_MODEL} ...")
model = SentenceTransformer(EMBEDDING_MODEL)
print(f"✅ Modèle prêt — dimension : {model.get_sentence_embedding_dimension()}")

if TRANSLATE_QUERY:
    try:
        from deep_translator import GoogleTranslator
        print("🌍 Traduction FR→EN activée")
    except ImportError:
        TRANSLATE_QUERY = False
        print("⚠️  deep-translator non installé — traduction désactivée")


# ── Schema auto-detection ─────────────────────────────────────────────────────
def detect_schema() -> dict:
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            # Detect column name: document_id (our DB) or id_document (organizers)
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'embeddings'
                  AND column_name IN ('document_id', 'id_document');
            """)
            row = cur.fetchone()
            doc_col = row[0] if row else 'id_document'

            # Detect documents table (for filename JOIN)
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = 'documents'
                );
            """)
            has_documents = cur.fetchone()[0]

            # Test pgvector native operator (<=> cosine distance)
            try:
                cur.execute("SELECT '[1,2,3]'::vector <=> '[1,2,3]'::vector")
                has_pgvector = True
            except Exception:
                conn.rollback()
                has_pgvector = False

    finally:
        conn.close()

    schema = {
        "doc_col":       doc_col,
        "has_documents": has_documents,
        "has_pgvector":  has_pgvector,
    }
    print(f"🔍 Schéma → colonne: '{doc_col}' | documents: {has_documents} | pgvector natif: {has_pgvector}")
    return schema


SCHEMA = detect_schema()


# ── Pydantic models ───────────────────────────────────────────────────────────
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


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0

def maybe_translate(text: str) -> str:
    if not TRANSLATE_QUERY:
        return text
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception:
        return text

def vec_to_pg(vec: np.ndarray) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in vec.tolist()) + "]"


# ── Routes ────────────────────────────────────────────────────────────────────
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
        doc_count = None
        if SCHEMA["has_documents"]:
            cur.execute("SELECT COUNT(*) FROM documents;")
            doc_count = cur.fetchone()[0]
        cur.close()
        conn.close()
        return {
            "status":           "ok",
            "model":            EMBEDDING_MODEL,
            "fragments_in_db":  frag_count,
            "documents_in_db":  doc_count,
            "top_k":            TOP_K,
            "translate_query":  TRANSLATE_QUERY,
            "schema": {
                "doc_column":    SCHEMA["doc_col"],
                "has_documents": SCHEMA["has_documents"],
                "has_pgvector":  SCHEMA["has_pgvector"],
            }
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Erreur DB : {str(e)}")


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide.")

    top_k    = max(1, min(req.top_k, 20))
    doc_col  = SCHEMA["doc_col"]

    # Étape 1 — (optionnel) traduction + embedding
    query_text = maybe_translate(req.question)
    q_vec = model.encode(query_text, normalize_embeddings=True).astype(np.float32)

    conn = get_connection()
    results = []
    total   = 0

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:

            # ── Méthode A : pgvector natif (rapide, utilise l'index) ──────────
            if SCHEMA["has_pgvector"]:
                join_clause = (
                    f"LEFT JOIN documents d ON d.id = e.{doc_col}"
                    if SCHEMA["has_documents"] else ""
                )
                fname_col = "d.filename AS source_file" if SCHEMA["has_documents"] else "NULL AS source_file"
                q_str = vec_to_pg(q_vec)

                cur.execute(f"""
                    SELECT e.id,
                           e.{doc_col}          AS document_id,
                           e.texte_fragment,
                           {fname_col},
                           ROUND((1 - (e.vecteur <=> %s::vector))::numeric, 4) AS score
                    FROM embeddings e
                    {join_clause}
                    ORDER BY e.vecteur <=> %s::vector
                    LIMIT %s;
                """, (q_str, q_str, top_k))
                rows = cur.fetchall()

                # total fragment count
                cur.execute("SELECT COUNT(*) FROM embeddings;")
                total = cur.fetchone()["count"]

                results = [
                    FragmentResult(
                        rank        = i + 1,
                        texte_fragment = r["texte_fragment"],
                        score       = float(r["score"]),
                        id          = r["id"],
                        id_document = r["document_id"],
                        source_file = r["source_file"],
                    )
                    for i, r in enumerate(rows)
                ]

            # ── Méthode B : fallback Python cosine similarity ─────────────────
            else:
                join_clause = (
                    f"LEFT JOIN documents d ON d.id = e.{doc_col}"
                    if SCHEMA["has_documents"] else ""
                )
                fname_col = "d.filename AS source_file" if SCHEMA["has_documents"] else "NULL AS source_file"

                cur.execute(f"""
                    SELECT e.id, e.{doc_col} AS document_id,
                           e.texte_fragment, e.vecteur::text, {fname_col}
                    FROM embeddings e {join_clause};
                """)
                rows = cur.fetchall()
                total = len(rows)

                if not rows:
                    raise HTTPException(status_code=404, detail="La base de données est vide.")

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

                scored.sort(key=lambda x: x["score"], reverse=True)
                results = [
                    FragmentResult(
                        rank           = i + 1,
                        texte_fragment = item["texte_fragment"],
                        score          = round(item["score"], 4),
                        id             = item["id"],
                        id_document    = item["id_document"],
                        source_file    = item["source_file"],
                    )
                    for i, item in enumerate(scored[:top_k])
                ]

    finally:
        conn.close()

    if not results:
        raise HTTPException(status_code=404, detail="La base de données est vide. Lancez load_data.py d'abord.")

    return SearchResponse(
        question                = req.question,
        results                 = results,
        total_fragments_searched= int(total),
    )

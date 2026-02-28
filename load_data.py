#!/usr/bin/env python3
"""
load_data.py — Ingestion PDF → PostgreSQL
Équipe : Jeuudiddy

- Lecture PDFs depuis ./pdf_files
- Nettoyage du texte (suppression boilerplate)
- Chunking par fenêtre glissante (60 mots, overlap 15)
- Embedding all-MiniLM-L6-v2 (dim 384)
- Déduplication par SHA-256 (ne re-charge pas si inchangé)
- Idempotent : safe à relancer
"""

import os, hashlib, re
import psycopg2
from psycopg2.extras import execute_batch
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from config import DB_CONFIG, EMBEDDING_MODEL

PDF_DIRECTORY = "./pdf_files"
CHUNK_SIZE    = 60
CHUNK_OVERLAP = 15
EMBEDDING_DIM = 384

NOISE_PATTERNS = [
    r"VTR&beyond", r"Pingbei Rd", r"Zhuhai", r"Guangdong",
    r"Stresemann", r"Berlin", r"Tel:.*?\d", r"Mail:.*?@",
    r"Website:.*?www", r"Last updating", r"info@vtrbeyond",
    r"www\.vtrbeyond\.com", r"86-756", r"\+49",
]

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            h.update(block)
    return h.hexdigest()

def clean_text(text):
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or len(line) < 8:
            continue
        if any(re.search(p, line, re.IGNORECASE) for p in NOISE_PATTERNS):
            continue
        lines.append(line)
    return " ".join(lines)

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            pt = page.extract_text()
            if pt:
                text += pt + "\n"
    except Exception as e:
        print(f"  ⚠️  Error reading {pdf_path}: {e}")
    return text

def sliding_window_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    if not words:
        return []
    step   = max(1, chunk_size - overlap)
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        if len(chunk) >= 10:
            chunks.append(" ".join(chunk))
        i += step
    return chunks

def load_data():
    conn  = psycopg2.connect(**DB_CONFIG)
    model = SentenceTransformer(EMBEDDING_MODEL)

    if not os.path.exists(PDF_DIRECTORY):
        os.makedirs(PDF_DIRECTORY)
        print(f"📁 Dossier '{PDF_DIRECTORY}' créé. Ajoutez les PDFs et relancez.")
        return

    pdf_files = sorted(f for f in os.listdir(PDF_DIRECTORY) if f.lower().endswith(".pdf"))
    print(f"📄 {len(pdf_files)} fichier(s) PDF détecté(s).\n")

    total_fragments = 0
    for filename in pdf_files:
        filepath = os.path.join(PDF_DIRECTORY, filename)
        checksum = compute_sha256(filepath)

        with conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, checksum FROM documents WHERE filename = %s", (filename,))
                result = cur.fetchone()

                if result:
                    doc_id, existing = result
                    if existing == checksum:
                        print(f"  ⏭  {filename} (inchangé, ignoré)")
                        continue
                    print(f"  🔄 {filename} modifié — re-ingestion")
                    cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))

                cur.execute(
                    "INSERT INTO documents (filename, checksum) VALUES (%s, %s) RETURNING id",
                    (filename, checksum)
                )
                doc_id = cur.fetchone()[0]

                raw  = extract_text_from_pdf(filepath)
                if not raw.strip():
                    print(f"  ⚠️  Aucun texte extrait de {filename}")
                    continue

                clean  = clean_text(raw)
                chunks = sliding_window_chunks(clean)
                if not chunks:
                    print(f"  ⚠️  Aucun fragment généré pour {filename}")
                    continue

                embeddings = model.encode(
                    chunks, batch_size=32,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

                data = [(doc_id, chunks[i], embeddings[i].tolist()) for i in range(len(chunks))]
                execute_batch(
                    cur,
                    "INSERT INTO embeddings (document_id, texte_fragment, vecteur) VALUES (%s, %s, %s)",
                    data, page_size=100,
                )
                total_fragments += len(chunks)
                print(f"  ✅ {filename} → {len(chunks)} fragments")

    conn.close()
    print(f"\n🏁 Ingestion terminée — {total_fragments} fragment(s) ajouté(s).")

if __name__ == "__main__":
    load_data()

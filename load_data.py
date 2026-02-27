#!/usr/bin/env python3

import os
import hashlib
import psycopg2
from psycopg2.extras import execute_batch
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from config import DB_CONFIG, EMBEDDING_MODEL

PDF_DIRECTORY = "./pdf_files"
CHUNK_WORD_TARGET = 180
EMBEDDING_DIM = 384


def compute_sha256(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            sha256.update(block)
    return sha256.hexdigest()


def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text


def smart_chunk(text, target_words=CHUNK_WORD_TARGET):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current_chunk = []
    current_len = 0

    for para in paragraphs:
        words = para.split()
        if current_len + len(words) <= target_words:
            current_chunk.append(para)
            current_len += len(words)
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [para]
            current_len = len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def load_data():
    conn = psycopg2.connect(**DB_CONFIG)
    model = SentenceTransformer(EMBEDDING_MODEL)

    if not os.path.exists(PDF_DIRECTORY):
        os.makedirs(PDF_DIRECTORY)
        print(f"Directory '{PDF_DIRECTORY}' created. Add PDFs and rerun.")
        return

    pdf_files = [f for f in os.listdir(PDF_DIRECTORY) if f.lower().endswith(".pdf")]
    print(f"{len(pdf_files)} PDF files detected.")

    for filename in pdf_files:
        filepath = os.path.join(PDF_DIRECTORY, filename)
        checksum = compute_sha256(filepath)

        with conn:
            with conn.cursor() as cur:

                cur.execute(
                    "SELECT id, checksum FROM documents WHERE filename = %s",
                    (filename,)
                )
                result = cur.fetchone()

                if result:
                    doc_id, existing_checksum = result
                    if existing_checksum == checksum:
                        print(f"Skipping {filename} (unchanged).")
                        continue
                    else:
                        print(f"{filename} changed. Re-ingesting.")
                        cur.execute(
                            "DELETE FROM documents WHERE id = %s",
                            (doc_id,)
                        )

                cur.execute(
                    "INSERT INTO documents (filename, checksum) VALUES (%s, %s) RETURNING id",
                    (filename, checksum)
                )
                doc_id = cur.fetchone()[0]

                full_text = extract_text_from_pdf(filepath)
                if not full_text.strip():
                    print(f"No text extracted from {filename}.")
                    continue

                fragments = smart_chunk(full_text)
                print(f"{filename}: {len(fragments)} fragments.")

                embeddings = model.encode(
                    fragments,
                    batch_size=32,
                    normalize_embeddings=True
                )

                if len(embeddings[0]) != EMBEDDING_DIM:
                    raise ValueError("Embedding dimension mismatch.")

                data_to_insert = [
                    (doc_id, fragments[i], embeddings[i].tolist())
                    for i in range(len(fragments))
                ]

                execute_batch(
                    cur,
                    "INSERT INTO embeddings (document_id, texte_fragment, vecteur) VALUES (%s, %s, %s)",
                    data_to_insert,
                    page_size=100
                )

                print(f"{filename} ingested successfully.")

    conn.close()
    print("Ingestion completed.")


if __name__ == "__main__":
    load_data()
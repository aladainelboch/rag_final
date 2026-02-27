-- init.sql

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    filename TEXT UNIQUE NOT NULL,
    checksum TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL
        REFERENCES documents(id)
        ON DELETE CASCADE,
    texte_fragment TEXT NOT NULL,
    vecteur VECTOR(384) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_embeddings_document_id
    ON embeddings(document_id);
#!/usr/bin/env python3
"""
load_data.py
============
Ingestion des fiches techniques BVZyme & Acide Ascorbique dans PostgreSQL.
- Ne charge les données QUE si la table est vide (idempotent)
- Génère les embeddings via all-MiniLM-L6-v2 (dim=384)

NOTE : Si la DB est déjà remplie par les organisateurs du challenge,
       ce script détectera automatiquement les données existantes et ne fera rien.

Usage : python load_data.py
"""

import psycopg2
from sentence_transformers import SentenceTransformer
from config import DB_CONFIG, EMBEDDING_MODEL

# 20 fragments extraits des 15 fiches techniques fournies
FRAGMENTS = [
    # ── Acide Ascorbique (E300) ──────────────────────────────────────────────
    (15, "Acide ascorbique (E300) : dosages recommandés → panification directe standard "
         "20-60 ppm, panification avec pousse lente 60-80 ppm, blocage froid positif 80-100 ppm, "
         "surgélation 150-200 ppm. Dosage maximum autorisé : 300 ppm (UE/France/Tunisie INNORPI)."),
    (15, "L'acide ascorbique renforce le gluten par création de ponts disulfurés, améliore "
         "le volume, raccourcit la fermentation de 15-30%. Action rapide en 5-15 minutes "
         "après incorporation. Environ 90% détruit pendant la cuisson."),
    (15, "Acide ascorbique spécifications : Formule C6H8O6, pureté 99.0-99.5%, poudre cristalline "
         "blanche, pH solution 1% : 2.0-2.5. Stockage 15-25°C, humidité < 60%, durée 18-24 mois. "
         "Autorisé UE (2014/34/UE), Tunisie (INNORPI), Codex Alimentarius."),
    (15, "Mode d'emploi acide ascorbique : incorporer aux ingrédients secs AVANT hydratation, "
         "ou diluer en solution 1-2%. Température optimale 25-30°C. Action en 5-15 min. "
         "Documenter chaque utilisation pour traçabilité."),

    # ── BVZyme AF110 – α-amylase fongique 150 000 SKB/g ─────────────────────
    (4,  "BVZyme AF110® : enzyme α-amylase fongique (150 000 SKB/g) produite par Aspergillus oryzae. "
         "Dosage recommandé : 2-12 ppm. Fonction : augmenter le volume, améliorer le pouvoir "
         "fermentaire, améliorer la tendreté, assister la fermentation."),
    (4,  "BVZyme AF110® agit sur l'amidon endommagé produit pendant la mouture par hydrolyse, "
         "produisant des sucres qui facilitent la fermentation. Stockage < 20°C, 24 mois, carton 25 kg."),

    # ── BVZyme AF330 – α-amylase fongique 11 900 FAU/g ──────────────────────
    (3,  "BVZyme AF330® : α-amylase fongique (11 900 FAU/g) produite par Aspergillus. "
         "Dosage : 2-10 ppm. Améliore la tendreté de la mie, la texture, le volume et la fermentation."),

    # ── BVZyme A SOFT205 – Maltogenic Amylase 11 600 NMAU/g ─────────────────
    (7,  "BVZyme A SOFT205® : amylase maltogénique (11 600 NMAU/g) issue de Bacillus subtilis. "
         "Dosage : 15-100 ppm. Améliore la fraîcheur, la douceur, l'élasticité du pain "
         "et prolonge la durée de conservation."),

    # ── BVZyme A SOFT305 – Maltogenic Amylase 10 500 NMAU/g ─────────────────
    (6,  "BVZyme A SOFT305® : amylase maltogénique (10 500 NMAU/g) issue de Bacillus sp. "
         "Plage de dosage 15-50 ppm, dosage suggéré 30 ppm. Fraîcheur et tendreté du pain."),

    # ── BVZyme A SOFT405 – Maltogenic Amylase 11 720 NMAU/g ─────────────────
    (5,  "BVZyme A SOFT405® : amylase maltogénique (11 720 NMAU/g) issue de Bacillus subtilis. "
         "Dosage : 15-90 ppm. Augmente le volume et la tendreté de la mie, améliore la fraîcheur "
         "et l'élasticité, prolonge la durée de vie du pain."),

    # ── BVZyme A FRESH202 – Maltogenic Amylase 10 950 NMAU/g ────────────────
    (8,  "BVZyme A FRESH202® : amylase maltogénique (10 950 NMAU/g) issue de Bacillus subtilis. "
         "Dosage : 10-90 ppm. Améliore la fraîcheur du pain, la tendreté et prolonge la durée de conservation."),

    # ── BVZyme AMG1400 – Amyloglucosidase 80 000 AGI/g ──────────────────────
    (1,  "BVZyme AMG1400® : amyloglucosidase fongique (80 000 AGI/g) produite par Aspergillus niger. "
         "Dosage : 10-100 ppm. Améliore la coloration dorée de la croûte, la luminosité de la mie, "
         "la pousse au four et la fermentation."),

    # ── BVZyme AMG880 – Amyloglucosidase 70 000 AGI/g ───────────────────────
    (2,  "BVZyme AMG880® : amyloglucosidase fongique (70 000 AGI/g) produite par Aspergillus niger. "
         "Dosage : 10-100 ppm. Améliore la coloration de la croûte, la luminosité, la fermentation "
         "et la pousse au four."),

    # ── BVZyme TG883 – Transglutaminase 400 U/g ─────────────────────────────
    (10, "BVZyme TG883® : transglutaminase (400 U/g) pour boulangerie. "
         "Dosage : 5-30 ppm. Améliore le volume, la texture, l'élasticité et la résistance de la pâte."),

    # ── BVZyme TG881 – Transglutaminase ─────────────────────────────────────
    (11, "BVZyme TG881® : transglutaminase pour boulangerie. Dosage : 10-40 ppm. "
         "Améliore le volume, la texture et l'élasticité de la pâte. Produit VTR&Beyond."),

    # ── BVZyme TG MAX63 – Transglutaminase ──────────────────────────────────
    (12, "BVZyme TG MAX63® : transglutaminase. Dosage : 5-25 ppm. "
         "Améliore volume, texture, élasticité, résistance, stabilité et tolérance de la pâte."),

    # ── BVZyme GOX 110 – Glucose Oxydase ────────────────────────────────────
    (14, "BVZyme GOX 110® : glucose oxydase pour boulangerie. Dosage : 5-40 ppm. "
         "Renforce le gluten, améliore la stabilité de la pâte et le volume du pain."),

    # ── BVZyme GO MAX 65 – Glucose Oxydase 11 000 U/g ───────────────────────
    (9,  "BVZyme GO MAX 65® : glucose oxydase (11 000 U/g). "
         "Propriétés physicochimiques, microbiologie et métaux lourds conformes. Produit VTR&Beyond."),

    # ── Généralités stockage et sécurité ────────────────────────────────────
    (99, "Tous les enzymes BVZyme : conditionnés en cartons de 25 kg, durée de vie minimale 24 mois, "
         "stockage en endroit frais et sec (< 20°C). Contient gluten (Annexe II, Règlement UE 1169/2011). "
         "Sans irradiation. Statut GMO conforme aux Règlements européens 1829/2003 et 1830/2003."),
    (99, "Microbiologie BVZyme (valeurs indicatives) : Flore totale < 50 000 UFC/g, "
         "Salmonella absent dans 25g, Coliformes < 30 UFC/g, Staphylococcus aureus absent dans 1g. "
         "Métaux lourds : Cd < 0.5, Hg < 0.5, As < 3, Pb < 5 mg/kg."),
]


def load_if_empty():
    """Charge les fragments uniquement si la table embeddings est vide."""
    print(f"⚙️  Connexion à PostgreSQL ({DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']})...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM embeddings;")
    count = cur.fetchone()[0]

    if count > 0:
        print(f"✅ Base déjà remplie ({count} fragments présents). Aucune action nécessaire.")
        cur.close()
        conn.close()
        return

    print(f"📥 Chargement du modèle {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"✅ Modèle prêt. Insertion de {len(FRAGMENTS)} fragments...")

    for i, (id_doc, texte) in enumerate(FRAGMENTS, 1):
        vec = model.encode(texte, normalize_embeddings=True).tolist()
        cur.execute(
            "INSERT INTO embeddings (id_document, texte_fragment, vecteur) VALUES (%s, %s, %s)",
            (id_doc, texte, vec)
        )
        print(f"  [{i:02d}/{len(FRAGMENTS)}] doc_id={id_doc} ✅")

    conn.commit()
    cur.close()
    conn.close()
    print(f"\n🏁 {len(FRAGMENTS)} fragments chargés avec succès dans PostgreSQL.")


if __name__ == "__main__":
    load_if_empty()

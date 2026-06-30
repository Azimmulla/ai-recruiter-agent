"""Precompute dense embeddings for the catalog -> data/embeddings.npy.

Run after scrape_catalog.py. Committing the embeddings means the deployed
service never embeds the whole catalog at boot (fast cold start); at request
time it only embeds the short user query.

Usage:  python -m scraper.build_index
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import CATALOG_PATH, EMBED_IDS_PATH, EMBED_MODEL, EMBEDDINGS_PATH  # noqa: E402


def main() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    if not catalog:
        raise SystemExit("Empty catalog.json — run scraper.scrape_catalog first.")

    from fastembed import TextEmbedding

    print(f"Embedding {len(catalog)} assessments with {EMBED_MODEL} ...")
    model = TextEmbedding(EMBED_MODEL)
    texts = [r["text"] for r in catalog]
    vecs = np.array(list(model.embed(texts)), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12  # L2-normalize for cosine

    np.save(EMBEDDINGS_PATH, vecs)
    EMBED_IDS_PATH.write_text(json.dumps([r["id"] for r in catalog]), encoding="utf-8")
    print(f"Wrote {vecs.shape} -> {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()

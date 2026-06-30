"""Hybrid retrieval: dense (bge-small) + sparse (BM25), fused with RRF.

Dense vectors are precomputed (data/embeddings.npy); only the short query is
embedded per request. BM25 catches exact tokens ("java", "c++", "verify") that
dense similarity blurs. If the embedding model can't load (e.g. constrained
host), retrieval degrades gracefully to BM25-only rather than failing.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from app.catalog import Catalog, get_catalog
from app.config import EMBED_IDS_PATH, EMBED_MODEL, EMBEDDINGS_PATH, RETRIEVE_K

_TOKEN = re.compile(r"[a-z0-9#+]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Retriever:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self.records = catalog.records

        # Sparse index (cheap to build for a few hundred docs).
        self._bm25 = BM25Okapi([tokenize(r["text"]) for r in self.records])

        # Dense index (precomputed). Row i corresponds to id self._ids[i].
        self._vecs: np.ndarray | None = None
        self._ids: list[int] = []
        if EMBEDDINGS_PATH.exists() and EMBED_IDS_PATH.exists():
            self._vecs = np.load(EMBEDDINGS_PATH)
            self._ids = json.loads(EMBED_IDS_PATH.read_text())
        self._embed_model = None  # lazy

    # -- dense query embedding (lazy model load) --
    def _embed_query(self, query: str) -> np.ndarray | None:
        if self._vecs is None:
            return None
        try:
            if self._embed_model is None:
                from fastembed import TextEmbedding

                self._embed_model = TextEmbedding(EMBED_MODEL)
            qv = next(self._embed_model.query_embed([query]))
            qv = np.asarray(qv, dtype=np.float32)
            return qv / (np.linalg.norm(qv) + 1e-12)
        except Exception:
            return None  # fall back to BM25-only

    @staticmethod
    def _ranking(scores: np.ndarray, ids: list[int]) -> list[int]:
        """ids ordered best-first by score."""
        order = np.argsort(-scores)
        return [ids[i] for i in order]

    def search(
        self,
        query: str,
        k: int = RETRIEVE_K,
        test_types: list[str] | None = None,
        restrict_types: list[str] | None = None,
    ) -> list[dict]:
        """Return up to k catalog records, best-first, by fused hybrid score.

        test_types softly boosts matching items; restrict_types hard-filters the
        result to items carrying one of those types (used for multi-aspect needs).
        """
        if not query.strip():
            return []
        all_ids = [r["id"] for r in self.records]
        if restrict_types:
            keep = {t.upper() for t in restrict_types}
            all_ids = [i for i in all_ids if keep & set(self.catalog.by_id[i]["test_type"])]
            if not all_ids:
                return []

        # Sparse ranking.
        bm = np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float32)
        sparse_rank = {rid: i for i, rid in enumerate(self._ranking(bm, all_ids))}

        # Dense ranking (optional).
        dense_rank: dict[int, int] = {}
        qv = self._embed_query(query)
        if qv is not None:
            sims = self._vecs @ qv
            dense_rank = {rid: i for i, rid in enumerate(self._ranking(sims, self._ids))}

        # Reciprocal Rank Fusion.
        C = 60.0
        fused: dict[int, float] = {}
        for rid in all_ids:
            s = 1.0 / (C + sparse_rank.get(rid, len(all_ids)))
            if dense_rank:
                s += 1.0 / (C + dense_rank.get(rid, len(all_ids)))
            fused[rid] = s

        # Mild boost for explicitly-requested test types (e.g. refine: "add personality").
        wanted = {t.upper() for t in (test_types or [])}
        if wanted:
            for rid in all_ids:
                if wanted & set(self.catalog.by_id[rid]["test_type"]):
                    fused[rid] *= 1.25

        ranked = sorted(all_ids, key=lambda r: fused[r], reverse=True)[:k]
        return [self.catalog.by_id[r] for r in ranked]


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever(get_catalog())

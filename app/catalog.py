"""Catalog access: load once, look up by id, fuzzy-match by name.

The catalog is the single source of truth for every name and URL the agent is
allowed to emit. Recommendations are always assembled from here by id, so a
hallucinated name or URL can never reach the response.
"""
from __future__ import annotations

import difflib
import json
import re
from functools import lru_cache

from app.config import CATALOG_PATH


def _norm(s: str) -> str:
    """Lowercase, drop punctuation/parenthetical noise for fuzzy comparison."""
    s = s.lower()
    s = re.sub(r"\(.*?\)", " ", s)            # drop "(New)", "(Short Form)"
    s = re.sub(r"[^a-z0-9+ ]", " ", s)        # keep + (e.g. "verify g+")
    return re.sub(r"\s+", " ", s).strip()


class Catalog:
    def __init__(self, records: list[dict]):
        self.records = records
        self.by_id = {r["id"]: r for r in records}
        self._norm_names = {r["id"]: _norm(r["name"]) for r in records}

    def __len__(self) -> int:
        return len(self.records)

    def get(self, _id: int) -> dict | None:
        return self.by_id.get(_id)

    def valid_ids(self, ids: list[int]) -> list[int]:
        """Keep only ids that exist — the grounding gate for recommendations."""
        seen: set[int] = set()
        out: list[int] = []
        for i in ids:
            if i in self.by_id and i not in seen:
                seen.add(i)
                out.append(i)
        return out

    def find_by_name(self, query: str, cutoff: float = 0.55) -> dict | None:
        """Best fuzzy match for a free-text assessment name (compare flow)."""
        q = _norm(query)
        if not q:
            return None
        best, best_score = None, cutoff
        for _id, name in self._norm_names.items():
            if q == name or q in name or name in q:
                return self.by_id[_id]
            score = difflib.SequenceMatcher(None, q, name).ratio()
            # reward shared significant tokens (e.g. "opq" within "opq32r")
            if set(q.split()) & set(name.split()):
                score += 0.15
            if score > best_score:
                best, best_score = self.by_id[_id], score
        return best


@lru_cache(maxsize=1)
def get_catalog() -> Catalog:
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"{CATALOG_PATH} missing — run `python -m scraper.scrape_catalog`."
        )
    records = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    return Catalog(records)

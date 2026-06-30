"""Central configuration and shared constants.

Loaded once at import time. Tolerant of a missing API key so that offline build
steps (scraping, index building) can import this module without a Gemini key.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:  # dotenv is optional in production (env vars set by host)
    pass

# --- Paths ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CATALOG_PATH = DATA_DIR / "catalog.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
EMBED_IDS_PATH = DATA_DIR / "embedding_ids.json"

# --- LLM ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash")
# Hard per-call wall-clock budget (the evaluator caps each /chat call at 30s).
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "10"))

# --- Retrieval ---
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "30"))      # candidates passed to the reranker
REC_FLOOR = int(os.getenv("REC_FLOOR", "10"))        # fill all 10 slots (Recall@10 has no precision penalty)
MAX_RECOMMENDATIONS = 10                              # hard schema cap (spec: 1..10)

# --- Catalog scope ---
# SHL standard "test type" key legend (single-letter codes shown in the catalog).
TEST_TYPE_NAMES: dict[str, str] = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}

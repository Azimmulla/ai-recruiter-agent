"""Build data/catalog.json from the official SHL catalog JSON.

SHL provided the authoritative catalog for this task as a single JSON document.
We use it as the source of truth (its `link` URLs and item set are exactly what
the evaluator grades against), transforming it into the app's catalog schema.

The raw file is cached at data/official_catalog.json so the build is reproducible
offline. Run scraper.load_official_catalog after updating that file (or pass --url).

Usage:  python -m scraper.load_official_catalog [--url <URL>]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import CATALOG_PATH, DATA_DIR  # noqa: E402

OFFICIAL_URL = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
RAW_PATH = DATA_DIR / "official_catalog.json"

# Official "keys" are full names; map to the single-letter SHL test-type codes.
KEY_TO_LETTER = {
    "Ability & Aptitude": "A",
    "Assessment Exercises": "E",
    "Biodata & Situational Judgment": "B",
    "Biodata & Situational Judgement": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Personality & Behaviour": "P",
    "Simulations": "S",
}


def _clean(s: str) -> str:
    """Strip control characters (the raw file contains a few) and collapse space."""
    if not s:
        return ""
    s = re.sub(r"[\x00-\x1f\x7f]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_raw(url: str | None) -> list[dict]:
    if url:
        import requests

        text = requests.get(url, timeout=60).text
        RAW_PATH.write_text(text, encoding="utf-8")
    elif RAW_PATH.exists():
        text = RAW_PATH.read_text(encoding="utf-8")
    else:
        raise SystemExit(
            f"{RAW_PATH} not found — pass --url {OFFICIAL_URL} to download it first."
        )
    return json.loads(text, strict=False)  # tolerate raw control characters


def transform(records: list[dict]) -> list[dict]:
    out: list[dict] = []
    for r in records:
        name = _clean(r.get("name", ""))
        link = (r.get("link") or "").strip()
        if not name or not link:
            continue
        key_names = [k for k in (r.get("keys") or []) if k in KEY_TO_LETTER]
        letters: list[str] = []
        for k in key_names:
            code = KEY_TO_LETTER[k]
            if code not in letters:
                letters.append(code)

        description = _clean(r.get("description", ""))
        job_levels = [j for j in (r.get("job_levels") or []) if j]
        languages = [lng for lng in (r.get("languages") or []) if lng]
        duration_raw = _clean(r.get("duration", ""))
        m = re.search(r"(\d+)", duration_raw)
        length_min = int(m.group(1)) if m else None

        embed_text = " — ".join(filter(None, [
            name,
            description,
            ("Test types: " + ", ".join(key_names)) if key_names else "",
            ("Job levels: " + ", ".join(job_levels)) if job_levels else "",
        ]))

        out.append({
            "id": len(out),
            "name": name,
            "slug": link.rstrip("/").rsplit("/", 1)[-1],
            "url": link,
            "test_type": letters,
            "test_type_names": key_names,
            "remote_testing": str(r.get("remote", "")).lower() == "yes",
            "adaptive_irt": str(r.get("adaptive", "")).lower() == "yes",
            "description": description,
            "job_levels": job_levels,
            "languages": languages,
            "assessment_length_min": length_min,
            "text": embed_text,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", nargs="?", const=OFFICIAL_URL, default=None,
                    help="download the official catalog from this URL (default cached file)")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    catalog = transform(load_raw(args.url))
    if not catalog:
        raise SystemExit("No records parsed from the official catalog.")
    CATALOG_PATH.write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8")

    from collections import Counter
    types = Counter(t for r in catalog for t in r["test_type"])
    print(f"Wrote {len(catalog)} assessments -> {CATALOG_PATH}")
    print(f"Test-type distribution: {dict(types)}")
    print(f"Sample URL: {catalog[0]['url']}")


if __name__ == "__main__":
    main()

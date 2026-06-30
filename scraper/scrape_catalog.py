"""Scrape the SHL Individual Test Solutions catalog into data/catalog.json.

The live SHL site has been restructured and the flat product-catalog table no
longer exists, so we reconstruct the catalog from the Wayback Machine, which
preserves the static per-product detail pages the assignment was written against
(stable + reproducible). Pages are cached on disk so re-runs are fast.

Scope: Individual Test Solutions only. Pre-packaged Job Solutions are excluded
via `is_job_solution()`, a slug heuristic validated at 100% against 69 index-page
ground-truth labels (36 individual + 33 job).

Usage:  python -m scraper.scrape_catalog  [--limit N]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import CATALOG_PATH, DATA_DIR, TEST_TYPE_NAMES  # noqa: E402

CATALOG_BASE = "https://www.shl.com/solutions/products/product-catalog/view/{slug}/"
CDX = "https://web.archive.org/cdx/search/cdx"
RAW_CACHE = DATA_DIR / "raw_cache"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (compatible; shl-catalog-scraper/1.0)"})

# Section labels used to bound free-text extraction on a detail page.
LABELS = [
    "Description", "Job levels", "Job Levels", "Languages",
    "Assessment length", "Assessment Length", "Test Type", "Remote Testing",
    "Adaptive", "Downloads", "Product Fact Sheet", "Completion Time",
]


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def is_job_solution(slug: str) -> bool:
    """True for Pre-packaged Job Solutions (excluded from scope)."""
    s = slug.lower()
    return (
        s.endswith("-solution")
        or "solution" in s
        or "job-focused-assessment" in s
        or s.endswith("short-form")
        or "-short-form" in s
        or re.search(r"-[78]-[01]($|-)", s) is not None  # versioned bundles (7-0/7-1/8-0)
    )


# --------------------------------------------------------------------------- #
# Wayback harvesting
# --------------------------------------------------------------------------- #
def _cdx_view_slugs(date_from: str, date_to: str) -> dict[str, str]:
    """Return {slug: timestamp} for archived /view/ detail pages in a window."""
    r = SESSION.get(
        CDX,
        params={
            "url": "shl.com/solutions/products/product-catalog/view/",
            "matchType": "prefix", "filter": "statuscode:200",
            "collapse": "urlkey", "output": "json",
            "from": date_from, "to": date_to,
        },
        timeout=120,
    )
    out: dict[str, str] = {}
    for ts, original in ((row[1], row[2]) for row in r.json()[1:]):
        m = re.search(r"/view/([a-z0-9\-]+)/?", original)
        if m:
            out.setdefault(m.group(1), ts)
    return out


def list_individual_slugs() -> dict[str, str]:
    """All Individual-Test-Solution slugs -> preferred capture timestamp.

    Prefer 2024 captures (assignment era); backfill anything else from all-time.
    """
    slugs = _cdx_view_slugs("2024", "2025")
    for slug, ts in _cdx_view_slugs("2010", "2026").items():
        slugs.setdefault(slug, ts)
    return {s: ts for s, ts in sorted(slugs.items()) if not is_job_solution(s)}


def _cdx_all_timestamps(slug: str) -> list[str]:
    r = SESSION.get(
        CDX,
        params={
            "url": f"shl.com/solutions/products/product-catalog/view/{slug}/",
            "filter": "statuscode:200", "output": "json", "limit": "8",
        },
        timeout=60,
    )
    return [row[1] for row in r.json()[1:]]


def fetch_archived(slug: str, ts: str) -> str | None:
    """Fetch raw archived HTML for a slug, cached on disk."""
    cache = RAW_CACHE / f"{slug}.html"
    if cache.exists() and cache.stat().st_size > 2000:
        return cache.read_text(encoding="utf-8", errors="replace")
    url = CATALOG_BASE.format(slug=slug)
    for attempt in range(3):
        try:
            r = SESSION.get(f"https://web.archive.org/web/{ts}id_/{url}", timeout=60)
            if r.status_code == 200 and len(r.text) > 2000:
                cache.write_text(r.text, encoding="utf-8")
                return r.text
        except requests.RequestException:
            pass
        time.sleep(1.5 * (attempt + 1))
    return None


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _section(text: str, label: str) -> str:
    """Return the text block following `label` up to the next known label."""
    i = text.find(label)
    if i < 0:
        return ""
    start = i + len(label)
    end = len(text)
    for other in LABELS:
        j = text.find(other, start)
        if 0 <= j < end:
            end = j
    return text[start:end].strip(" :\n\t")


def _split_list(blob: str) -> list[str]:
    parts = re.split(r"[,\n]", blob)
    return [p.strip() for p in parts if p.strip()]


def parse_detail(html: str, slug: str) -> dict | None:
    soup = BeautifulSoup(html, "lxml")

    # Name: the product <h1> is authoritative; the page also has a hidden
    # "Outdated browser detected" h1 and a generic catalog <title>, so skip those.
    name = ""
    for h1 in soup.find_all("h1"):
        t = h1.get_text(strip=True)
        if t and "outdated browser" not in t.lower():
            name = t
            break
    if not name and soup.title:
        name = re.sub(r"\s*\|\s*SHL\s*$", "", soup.title.get_text(strip=True)).strip()
    if not name or name.lower() == "product assessment catalog":
        return None

    text = re.sub(r"[ \t]+", " ", soup.get_text("\n"))
    text = re.sub(r"\n\s*\n+", "\n", text)

    description = _section(text, "Description")
    description = re.sub(r"\s+", " ", description).strip()

    job_levels = _split_list(_section(text, "Job levels") or _section(text, "Job Levels"))
    languages = _split_list(_section(text, "Languages"))

    m = (re.search(r"[Cc]ompletion [Tt]ime in minutes\s*=\s*(\d+)", text)
         or re.search(r"=\s*(\d+)\s*(?:\n|$)", _section(text, "Assessment length"))
         or re.search(r"(\d+)\s*min", _section(text, "Assessment length")))
    length_min = int(m.group(1)) if m else None

    # Test-type letters: ONLY the product's own keys, which live inside the
    # <p> labelled "Test Type:" (a separate legend lists all 8 keys page-wide).
    letters: list[str] = []
    for p in soup.select("p.product-catalogue__small-text"):
        if "test type" in p.get_text(" ", strip=True).lower():
            for sp in p.select("span.product-catalogue__key"):
                k = sp.get_text(strip=True).upper()
                if k in TEST_TYPE_NAMES and k not in letters:
                    letters.append(k)
            break

    if "this is a test" in description.lower():
        return None  # junk catalog seed entry

    # Remote-testing / adaptive flags from the labeled circle spans.
    def flag(keyword: str) -> bool:
        for p in soup.select("p.product-catalogue__small-text"):
            if keyword.lower() in p.get_text(" ", strip=True).lower():
                circle = p.select_one("span.catalogue__circle")
                if circle:
                    return "-yes" in circle.get("class", [])
        return False

    remote = flag("Remote Testing")
    adaptive = flag("Adaptive")

    if not description and not letters:
        return None  # JS-shell capture; signal caller to retry another timestamp.

    type_names = [TEST_TYPE_NAMES[k] for k in letters]
    embed_text = " — ".join(filter(None, [
        name,
        description,
        ("Test types: " + ", ".join(type_names)) if type_names else "",
        ("Job levels: " + ", ".join(job_levels)) if job_levels else "",
    ]))

    return {
        "name": name,
        "slug": slug,
        "url": CATALOG_BASE.format(slug=slug),
        "test_type": letters,
        "test_type_names": type_names,
        "remote_testing": remote,
        "adaptive_irt": adaptive,
        "description": description,
        "job_levels": job_levels,
        "languages": languages,
        "assessment_length_min": length_min,
        "text": embed_text,
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap number of products (debug)")
    args = ap.parse_args()

    RAW_CACHE.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    slug_ts = list_individual_slugs()
    if args.limit:
        slug_ts = dict(list(slug_ts.items())[: args.limit])
    print(f"Individual Test Solution slugs to scrape: {len(slug_ts)}")

    catalog: list[dict] = []
    failures: list[str] = []
    for n, (slug, ts) in enumerate(slug_ts.items(), 1):
        html = fetch_archived(slug, ts)
        rec = parse_detail(html, slug) if html else None
        if rec is None:  # retry with alternate captures
            for alt in _cdx_all_timestamps(slug):
                if alt == ts:
                    continue
                (RAW_CACHE / f"{slug}.html").unlink(missing_ok=True)
                html = fetch_archived(slug, alt)
                rec = parse_detail(html, slug) if html else None
                if rec:
                    break
        if rec is None:
            failures.append(slug)
        else:
            rec["id"] = len(catalog)
            catalog.append(rec)
        if n % 25 == 0 or n == len(slug_ts):
            print(f"  [{n}/{len(slug_ts)}] parsed={len(catalog)} failed={len(failures)}")
        time.sleep(0.15)

    CATALOG_PATH.write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(catalog)} assessments -> {CATALOG_PATH}")
    if failures:
        print(f"Unparseable ({len(failures)}): {', '.join(failures[:15])}"
              + (" ..." if len(failures) > 15 else ""))


if __name__ == "__main__":
    main()

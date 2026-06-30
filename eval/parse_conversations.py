"""Convert the GenAI_SampleConversations/*.md traces into eval/traces.json.

Each sample conversation is a labeled trace: the user turns drive the replay and
the FINAL recommendations table is the expected shortlist (the relevance labels
for Recall@10). Expected names are validated against the catalog so we notice any
that don't resolve.

Usage:  python -m eval.parse_conversations [<folder>]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.catalog import get_catalog  # noqa: E402

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "GenAI_SampleConversations"
OUT_PATH = Path(__file__).resolve().parent / "traces.json"


def _user_turns(text: str) -> list[str]:
    turns: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() == "**User**":
            quote: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].lstrip().startswith(">"):
                i += 1  # skip blanks between header and quote
            while i < len(lines) and lines[i].lstrip().startswith(">"):
                quote.append(lines[i].lstrip()[1:].strip())
                i += 1
            msg = " ".join(q for q in quote if q).strip()
            if msg:
                turns.append(msg)
        else:
            i += 1
    return turns


def _last_table_names(text: str) -> list[str]:
    """Names from the last markdown recommendations table in the file."""
    # Walk all table rows; keep names from the final contiguous table.
    names: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if re.match(r"\s*\|", line) and "|" in line.strip("| "):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 2:
                continue
            name = cells[1]
            if not name or name.lower() == "name" or set(name) <= {"-", " ", ":"}:
                continue
            current.append(name)
        elif current:
            names = current            # table ended; remember the latest complete one
            current = []
    if current:
        names = current
    # strip markdown footnote markers like "_(+35 more)_"
    return [re.sub(r"\s*_\(.*?\)_", "", n).strip() for n in names]


def main() -> None:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    catalog = get_catalog()
    traces = []
    for md in sorted(folder.glob("*.md"), key=lambda p: (len(p.stem), p.stem)):
        text = md.read_text(encoding="utf-8")
        turns = _user_turns(text)
        expected = _last_table_names(text)
        resolved, missing = [], []
        for name in expected:
            r = catalog.find_by_name(name)
            (resolved if r else missing).append(r["name"] if r else name)
        traces.append({"id": md.stem, "user_turns": turns, "expected": resolved})
        flag = f"  ⚠ unmatched: {missing}" if missing else ""
        print(f"{md.stem}: {len(turns)} turns, {len(resolved)}/{len(expected)} labels resolved{flag}")

    OUT_PATH.write_text(json.dumps(traces, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(traces)} traces -> {OUT_PATH}")


if __name__ == "__main__":
    main()

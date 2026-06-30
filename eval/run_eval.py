"""Replay harness: mean Recall@10 over conversation traces.

Mirrors the grader's loop: feed the trace's scripted user turns to POST /chat
(in-process), answer any extra agent question with "no preference", and stop at
the first shortlist (the simulated user ends when the agent recommends). The
trace's expected assessment names are resolved against the live catalog via
fuzzy match, so labels stay valid even as the catalog is rebuilt.

Usage:  python -m eval.run_eval
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from app.agent import handle
from app.catalog import get_catalog

TRACES_PATH = Path(__file__).resolve().parent / "traces.json"
NO_PREF = "I don't have a strong preference."
MAX_TURNS = 8
# Light pacing keeps the free-tier Gemini quota from 429-ing during the burst.
TURN_DELAY_S = float(os.getenv("EVAL_TURN_DELAY_S", "3"))


def _resolve_relevant(expected: list[str]) -> set[int]:
    cat = get_catalog()
    ids: set[int] = set()
    for name in expected:
        r = cat.find_by_name(name)
        if r:
            ids.add(r["id"])
    return ids


def _replay(user_turns: list[str]) -> list[dict]:
    """Run a conversation; return the final (first non-empty) shortlist."""
    history: list[dict] = []
    final: list[dict] = []
    i = 0
    for _ in range(MAX_TURNS // 2):
        user = user_turns[i] if i < len(user_turns) else NO_PREF
        i += 1
        history.append({"role": "user", "content": user})
        out = handle(history)
        history.append({"role": "assistant", "content": out["reply"]})
        time.sleep(TURN_DELAY_S)
        if out["recommendations"]:
            final = out["recommendations"]
            break
    return final


def _rec_ids(recs: list[dict]) -> list[int]:
    cat = get_catalog()
    name_to_id = {r["name"]: r["id"] for r in cat.records}
    return [name_to_id[r["name"]] for r in recs if r["name"] in name_to_id]


def main() -> None:
    traces = json.loads(TRACES_PATH.read_text())
    recalls = []
    print(f"{'trace':32} {'recall@10':>9}  rel  hit")
    print("-" * 60)
    for t in traces:
        relevant = _resolve_relevant(t["expected"])
        final = _replay(t["user_turns"])
        top10 = _rec_ids(final)[:10]
        hit = len(set(top10) & relevant)
        recall = hit / len(relevant) if relevant else 0.0
        recalls.append(recall)
        print(f"{t['id']:32} {recall:9.2f}  {len(relevant):>3}  {hit:>3}")
    print("-" * 60)
    print(f"{'MEAN Recall@10':32} {sum(recalls) / len(recalls):9.3f}")


if __name__ == "__main__":
    main()

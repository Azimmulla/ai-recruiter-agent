"""Gemini reasoning layer.

Every public function returns a plain dict and never raises: on a missing key,
API error, or timeout it returns {} so the agent can fall back to pure retrieval
and still answer within the 30s budget. A hard thread timeout guarantees we
abandon a slow call rather than blow the deadline.

We request `application/json` and parse the text ourselves rather than passing a
`response_schema` — the Gemini API rejects schemas that carry default values, and
hand-parsing keeps us tolerant of minor shape drift.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from functools import lru_cache

from app import prompts
from app.config import (
    GEMINI_API_KEY,
    GEMINI_FALLBACK_MODEL,
    GEMINI_MODEL,
    LLM_TIMEOUT_S,
)

_POOL = ThreadPoolExecutor(max_workers=4)


@lru_cache(maxsize=1)
def _client():
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")
    from google import genai

    return genai.Client(api_key=GEMINI_API_KEY)


def _parse_json(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):  # strip accidental markdown fences
        text = text.strip("`")
        text = text[text.find("{"):]
    obj = json.loads(text)
    return obj if isinstance(obj, dict) else {}


def _generate_json(system: str, user: str) -> dict:
    """Call Gemini for a JSON object; try primary then fallback model.

    Returns {} on any failure (missing key, timeout, parse error).
    """
    from google.genai import types

    # Headroom for max_output_tokens: 2.5-class models spend output tokens on
    # internal reasoning, so we keep the ceiling high to avoid truncated JSON.
    cfg = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        temperature=0.2,
        max_output_tokens=4096,
    )

    def _call(model: str) -> dict:
        resp = _client().models.generate_content(model=model, contents=user, config=cfg)
        return _parse_json(resp.text)

    for model in (GEMINI_MODEL, GEMINI_FALLBACK_MODEL):
        try:
            return _POOL.submit(_call, model).result(timeout=LLM_TIMEOUT_S)
        except (FuturesTimeout, Exception):
            continue
    return {}


# --- public API ---
def route(messages: list[dict]) -> dict:
    convo = prompts.format_conversation(messages)
    user = f"Conversation so far:\n{convo}\n\nDecide the action and fill the JSON."
    return _generate_json(prompts.ROUTER_SYSTEM, user)


def rerank(query: str, candidates: list[dict]) -> dict:
    user = (
        f"User need: {query}\n\nCANDIDATES:\n{prompts.format_candidates(candidates)}\n\n"
        "Select the best fitting candidates by id."
    )
    return _generate_json(prompts.RERANK_SYSTEM, user)


def compare(question: str, records: list[dict]) -> dict:
    user = (
        f"Question: {question}\n\nCATALOG FACTS:\n"
        f"{prompts.format_compare_facts(records)}\n\nWrite the grounded comparison."
    )
    return _generate_json(prompts.COMPARE_SYSTEM, user)

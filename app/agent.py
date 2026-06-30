"""Agent orchestration: route -> dispatch -> grounded response.

Stateless: every call re-derives intent from the full history. The four
behaviors (clarify / recommend / refine / compare) plus refuse are dispatched
here. Recommendations are always assembled from the catalog by id, so the
response can never contain a name or URL that isn't in the catalog. The whole
entrypoint is wrapped so it always returns a schema-valid dict.
"""
from __future__ import annotations

import re

from app import llm
from app.catalog import get_catalog
from app.config import MAX_RECOMMENDATIONS, REC_FLOOR, RETRIEVE_K
from app.retrieval import get_retriever

REFUSALS = {
    "injection": "I can only help you find SHL assessments, so I can't take on other "
                 "instructions or roles. Tell me about the role you're hiring for and I'll "
                 "suggest suitable assessments.",
    "legal": "I'm not able to give legal, compliance, or HR-policy advice. I can help with "
             "finding SHL assessments though — what role are you hiring for?",
}
REFUSAL_DEFAULT = ("I can only help with finding SHL assessments. I can't help with that, "
                   "but if you tell me about the role you're hiring for, I'll suggest some.")
CLARIFY_DEFAULT = ("Happy to help. Could you tell me a bit about the role — the job title, "
                   "the key skills, and the seniority level?")
VAGUE_MAX_WORDS = 6
BACKBONE = 4          # top retrieval matches always retained, ahead of the reranker

# High-precision deterministic scope guard: refuses obvious injection / legal-advice
# inputs without an LLM call, so refusal survives even when Gemini is rate-limited.
_INJECTION_PAT = ("ignore previous", "ignore all previous", "ignore the above",
                  "disregard previous", "disregard your", "disregard all",
                  "system prompt", "you are now", "reveal your", "your instructions",
                  "developer mode", "jailbreak", "pretend you are")
_LEGAL_PAT = ("is it legal", "is it illegal", "legal to ", "illegal to ", "legally ",
              "against the law", "lawsuit", "discriminat", "gdpr", "employment law",
              "labour law", "labor law")


def _scope_guard(text: str) -> str | None:
    t = text.lower()
    if any(p in t for p in _INJECTION_PAT):
        return "injection"
    if any(p in t for p in _LEGAL_PAT):
        return "legal"
    return None
_ROLE_HINTS = ("develop", "engineer", "manager", "analyst", "sales", "java", "python",
               "nurse", "accountant", "graduate", "agent", "representative", "clerk",
               "leader", "administrator", "designer", "consultant", "technician",
               "skills", "personality", "cognitive", "numerical", "verbal", "experience")


def _rec(record: dict) -> dict:
    return {
        "name": record["name"],
        "url": record["url"],
        # Spec example shows a single primary letter code; keep the first.
        "test_type": (record["test_type"][0] if record["test_type"] else ""),
    }


def _last_user(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user" and (m.get("content") or "").strip():
            return m["content"].strip()
    return ""


def _all_user_text(messages: list[dict]) -> str:
    return " ".join(m.get("content", "") for m in messages if m.get("role") == "user").strip()


def _looks_vague(text: str) -> bool:
    t = text.lower()
    return len(t.split()) <= VAGUE_MAX_WORDS and not any(h in t for h in _ROLE_HINTS)


# Deterministic test-type extraction: maps explicit category words to SHL keys so
# "refine" (e.g. "add a personality test") works even when the LLM router is down.
_TYPE_KEYWORDS = (
    ("P", ("personality", "behavioural", "behavioral", "opq", "motivation questionnaire",
           "disposition", "temperament", "soft skill")),
    ("A", ("cognitive", "aptitude", "numerical", "verbal reasoning", "inductive", "deductive",
           "logical reasoning", "abstract reasoning", "reasoning test", "ability test",
           "general ability")),
    ("S", ("simulation", "role play", "role-play")),
    ("B", ("situational judg", "sjt", "judgement test", "judgment test", "biodata")),
    ("C", ("competency", "competencies")),
    ("K", ("coding test", "programming test", "technical skills", "knowledge test")),
)


def _as_str_list(value) -> list[str]:
    """Coerce a router field to a clean list of strings (it may arrive as a
    string, None, or a list with non-string items if the model drifts)."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


def _extract_test_types(text: str) -> list[str]:
    t = text.lower()
    out: list[str] = []
    for code, kws in _TYPE_KEYWORDS:
        if any(k in t for k in kws) and code not in out:
            out.append(code)
    return out


# Deterministic compare-intent detection: routes "difference between X and Y",
# "compare X and Y", "X vs Y" to the grounded compare path without the LLM.
_COMPARE_PATTERNS = (
    re.compile(r"\b(?:difference|differences|differ)\b[^.?!]*?\bbetween\b\s+(.+?)\s+"
               r"\b(?:and|vs\.?|versus|or)\b\s+(.+?)[?.!]*$", re.I),
    re.compile(r"\bcompare\b\s+(.+?)\s+\b(?:and|with|to|vs\.?|versus)\b\s+(.+?)[?.!]*$", re.I),
    re.compile(r"\bhow does\b\s+(.+?)\s+\bdiffer from\b\s+(.+?)[?.!]*$", re.I),
    re.compile(r"^(.+?)\s+\b(?:vs\.?|versus)\b\s+(.+?)[?.!]*$", re.I),
)
_TAIL = re.compile(r"\s*\b(?:assessment|assessments|test|tests)\b\s*$", re.I)
_LEAD = re.compile(r"^(?:a|an|the)\s+", re.I)


def _detect_compare(text: str) -> list[str] | None:
    text = text.strip()
    for pat in _COMPARE_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        a, b = (_LEAD.sub("", _TAIL.sub("", g.strip(" ,"))).strip() for g in m.groups())
        if a and b and len(a) < 60 and len(b) < 60:
            return [a, b]
    return None


def _recommend(query: str, test_types: list[str]) -> dict:
    catalog = get_catalog()
    retriever = get_retriever()
    candidates = retriever.search(query, k=RETRIEVE_K, test_types=test_types)

    # Multi-aspect needs (e.g. a Java skills test AND a personality test for
    # stakeholder skills): make sure every requested test type is represented in
    # the candidate pool, otherwise a skill-dominated query never surfaces it.
    if test_types:
        seen = {c["id"] for c in candidates}
        for tt in test_types:
            if not any(tt in c["test_type"] for c in candidates):
                for extra in retriever.search(query, k=6, restrict_types=[tt]):
                    if extra["id"] not in seen:
                        seen.add(extra["id"])
                        candidates.append(extra)

    if not candidates:
        return {"reply": CLARIFY_DEFAULT, "recommendations": [], "end_of_conversation": False}

    rr = llm.rerank(query, candidates)
    rr_ids = catalog.valid_ids([int(i) for i in rr.get("ids", []) if isinstance(i, (int, str))
                                and str(i).lstrip("-").isdigit()])
    cand_ids = [c["id"] for c in candidates]

    # Build the shortlist by priority, deduping by name (the catalog has a few
    # same-named variants). Recall@10 is order-independent, so the goal is the
    # best SET of <=10 items.
    final: list[int] = []
    seen_names: set[str] = set()

    def add(i: int) -> None:
        r = catalog.get(i)
        if not r or i in final or len(final) >= MAX_RECOMMENDATIONS:
            return
        key = r["name"].strip().lower()
        if key in seen_names:
            return
        seen_names.add(key)
        final.append(i)

    for i in cand_ids[:BACKBONE]:          # 1. strongest retrieval matches, never dropped
        add(i)
    for tt in test_types:                  # 2. guarantee each requested test type (refine)
        if not any(tt in catalog.get(i)["test_type"] for i in final):
            for c in candidates:
                if tt in c["test_type"]:
                    add(c["id"])
                    break
    for i in rr_ids:                       # 3. reranker refinements (may surface lower-ranked gems)
        add(i)
    for i in cand_ids:                     # 4. pad to the recall floor
        if len(final) >= REC_FLOOR:
            break
        add(i)

    recs = [_rec(catalog.get(i)) for i in final]
    reply = rr.get("reply") or f"Here are {len(recs)} SHL assessments that fit what you described."
    return {"reply": reply, "recommendations": recs, "end_of_conversation": False}


def _compare(question: str, targets: list[str]) -> dict | None:
    catalog = get_catalog()
    matched, seen = [], set()
    for t in targets:
        r = catalog.find_by_name(t)
        if r and r["id"] not in seen:
            seen.add(r["id"])
            matched.append(r)
    if not matched:
        return None  # named items not in catalog -> let caller recommend instead

    res = llm.compare(question, matched)
    reply = res.get("reply")
    if not reply:  # grounded template fallback (works with no LLM)
        parts = []
        for r in matched:
            types = ", ".join(r["test_type_names"]) or "specialised"
            gist = (r["description"] or "").split(". ")[0][:140]
            length = r.get("assessment_length_min")
            mins = f", ~{length} min" if length else ""
            parts.append(f"{r['name']} is a {types} assessment{mins}: {gist}")
        reply = "Here's how they compare. " + " ".join(parts)
    return {"reply": reply, "recommendations": [_rec(r) for r in matched],
            "end_of_conversation": False}


def handle(messages: list[dict]) -> dict:
    try:
        messages = [m for m in messages if isinstance(m, dict)]
        if not _all_user_text(messages):
            return {"reply": CLARIFY_DEFAULT, "recommendations": [], "end_of_conversation": False}

        # Deterministic guardrail first (robust to LLM downtime / rate limits).
        guard = _scope_guard(_last_user(messages))
        if guard:
            return {"reply": REFUSALS.get(guard, REFUSAL_DEFAULT),
                    "recommendations": [], "end_of_conversation": False}

        prior_clarifications = sum(
            1 for m in messages
            if m.get("role") == "assistant" and (m.get("content") or "").strip().endswith("?")
        )
        assistant_turns = sum(1 for m in messages if m.get("role") == "assistant")
        must_recommend = assistant_turns >= 3  # protect the 8-turn cap

        decision = llm.route(messages)
        action = (decision.get("action") or "").lower()
        last = _last_user(messages)

        # Deterministic signals (work even when the LLM router is rate-limited).
        det_targets = _detect_compare(last)
        det_types = _extract_test_types(_all_user_text(messages))

        # --- LLM unavailable: heuristic fallback that still covers all behaviors ---
        if not decision:
            if det_targets:
                res = _compare(last, det_targets)
                if res is not None:
                    return res
            if _looks_vague(_all_user_text(messages)) and not prior_clarifications and not must_recommend:
                return {"reply": CLARIFY_DEFAULT, "recommendations": [], "end_of_conversation": False}
            return _recommend(_all_user_text(messages), det_types)

        if action == "refuse" and not must_recommend:
            reply = REFUSALS.get(decision.get("refusal_reason", ""), REFUSAL_DEFAULT)
            return {"reply": reply, "recommendations": [], "end_of_conversation": False}

        # Compare: router-detected OR deterministically detected (handles a throttled
        # router that misroutes a clear "difference between X and Y" to recommend).
        compare_targets = _as_str_list(decision.get("compare_targets")) or det_targets
        if compare_targets:
            res = _compare(last, compare_targets)
            if res is not None:
                return res
            # nothing matched the catalog -> fall through to recommend

        if action == "clarify" and not prior_clarifications and not must_recommend:
            q = decision.get("clarifying_question") or CLARIFY_DEFAULT
            return {"reply": q, "recommendations": [], "end_of_conversation": False}

        query = decision.get("search_query") or _all_user_text(messages)
        test_types = list(dict.fromkeys(_as_str_list(decision.get("test_types")) + det_types))
        return _recommend(query, test_types)

    except Exception:
        # Last-resort safety net: never 500 the evaluator.
        return {
            "reply": "Sorry, I had trouble with that. Could you restate the role and key "
                     "skills you're hiring for?",
            "recommendations": [],
            "end_of_conversation": False,
        }

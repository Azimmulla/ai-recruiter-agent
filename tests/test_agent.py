"""Behavior + grounding tests for the agent (LLM mocked, deterministic)."""
import pytest

from app import agent, llm
from app.catalog import get_catalog

catalog = get_catalog()
pytestmark = pytest.mark.skipif(len(catalog) == 0, reason="catalog.json not built yet")

U = lambda c: {"role": "user", "content": c}       # noqa: E731
A = lambda c: {"role": "assistant", "content": c}  # noqa: E731


def test_refuse_returns_no_recommendations(monkeypatch):
    monkeypatch.setattr(llm, "route", lambda m: {"action": "refuse", "refusal_reason": "legal"})
    out = agent.handle([U("Is it legal to reject candidates over 50?")])
    assert out["recommendations"] == []
    assert out["reply"]


def test_clarify_on_vague_turn1(monkeypatch):
    monkeypatch.setattr(llm, "route",
                        lambda m: {"action": "clarify", "clarifying_question": "What role are you hiring for?"})
    out = agent.handle([U("I need an assessment")])
    assert out["recommendations"] == []
    assert out["reply"].endswith("?")


def test_does_not_over_clarify(monkeypatch):
    # Already asked a question once; even if router says clarify again, we recommend.
    monkeypatch.setattr(llm, "route", lambda m: {"action": "clarify", "clarifying_question": "And seniority?"})
    monkeypatch.setattr(llm, "rerank", lambda q, c: {"ids": [c[0]["id"]], "reply": "Here are some."})
    out = agent.handle([U("I need an assessment"), A("What role?"), U("a java developer")])
    assert len(out["recommendations"]) >= 1


def test_recommendations_are_grounded(monkeypatch):
    # rerank returns a bogus id mixed with real ones; bogus must be dropped.
    real_id = catalog.records[0]["id"]
    monkeypatch.setattr(llm, "route", lambda m: {"action": "recommend", "search_query": "developer"})
    monkeypatch.setattr(llm, "rerank", lambda q, c: {"ids": [999999, real_id], "reply": "ok"})
    out = agent.handle([U("hiring a developer")])
    urls = {r["url"] for r in catalog.records}
    assert 1 <= len(out["recommendations"]) <= 10
    for rec in out["recommendations"]:
        assert rec["url"] in urls                      # no hallucinated URL
        assert rec["name"] in {r["name"] for r in catalog.records}


def test_recommend_caps_at_10(monkeypatch):
    ids = [r["id"] for r in catalog.records[:20]]
    monkeypatch.setattr(llm, "route", lambda m: {"action": "recommend", "search_query": "test"})
    monkeypatch.setattr(llm, "rerank", lambda q, c: {"ids": ids, "reply": "ok"})
    out = agent.handle([U("hiring for a technical role")])
    assert len(out["recommendations"]) <= 10


def test_compare_is_grounded(monkeypatch):
    target = catalog.records[0]["name"]
    monkeypatch.setattr(llm, "route", lambda m: {"action": "compare", "compare_targets": [target]})
    monkeypatch.setattr(llm, "compare", lambda q, recs: {"reply": "They differ in focus."})
    out = agent.handle([U(f"what is {target}?")])
    assert any(r["name"] == target for r in out["recommendations"])


def test_llm_down_fallback_recommends(monkeypatch):
    monkeypatch.setattr(llm, "route", lambda m: {})   # simulate LLM failure
    out = agent.handle([U("hiring a python developer with sql")])
    assert len(out["recommendations"]) >= 1           # still functional via retrieval


def test_extract_test_types():
    assert agent._extract_test_types("also add a personality test") == ["P"]
    assert set(agent._extract_test_types("numerical reasoning and personality")) == {"A", "P"}
    assert agent._extract_test_types("hiring a java developer") == []


def test_detect_compare():
    assert agent._detect_compare("difference between OPQ and a numerical test") == ["OPQ", "numerical"]
    assert agent._detect_compare("OPQ32r vs Verify Numerical") == ["OPQ32r", "Verify Numerical"]
    assert agent._detect_compare("hiring a junior or senior developer") is None


def test_refine_adds_personality_without_llm(monkeypatch):
    # LLM fully down: deterministic test-type extraction must still add a P item.
    monkeypatch.setattr(llm, "route", lambda m: {})
    monkeypatch.setattr(llm, "rerank", lambda q, c: {})
    out = agent.handle([U("Hiring a Java developer"), A("Here are some."),
                        U("Actually, also add a personality test")])
    assert any("P" in r["test_type"] for r in out["recommendations"])


def test_compare_without_llm(monkeypatch):
    # LLM fully down: deterministic compare detection + grounded template fallback.
    monkeypatch.setattr(llm, "route", lambda m: {})
    monkeypatch.setattr(llm, "compare", lambda q, recs: {})
    out = agent.handle([U("What's the difference between OPQ and a numerical reasoning test?")])
    assert 1 <= len(out["recommendations"]) <= 10
    assert "compare" in out["reply"].lower()
    urls = {r["url"] for r in catalog.records}
    assert all(r["url"] in urls for r in out["recommendations"])


def test_malformed_router_shape_does_not_crash(monkeypatch):
    # Router drifts and returns test_types/compare_targets as strings, not lists.
    monkeypatch.setattr(llm, "route",
                        lambda m: {"action": "recommend", "search_query": "java", "test_types": "P"})
    monkeypatch.setattr(llm, "rerank", lambda q, c: {})
    out = agent.handle([U("hiring a java developer")])
    assert 1 <= len(out["recommendations"]) <= 10

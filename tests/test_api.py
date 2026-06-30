"""Contract tests for the HTTP surface (LLM mocked, no network)."""
from fastapi.testclient import TestClient

from app import llm
from app.main import app

client = TestClient(app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_chat_response_schema(monkeypatch):
    monkeypatch.setattr(llm, "route", lambda m: {"action": "recommend", "search_query": "java developer"})
    monkeypatch.setattr(llm, "rerank", lambda q, c: {"ids": [c[0]["id"]], "reply": "Here you go."})

    r = client.post("/chat", json={"messages": [{"role": "user", "content": "hiring a java developer"}]})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"reply", "recommendations", "end_of_conversation"}
    assert isinstance(body["reply"], str)
    assert isinstance(body["end_of_conversation"], bool)
    assert 0 <= len(body["recommendations"]) <= 10
    for rec in body["recommendations"]:
        assert {"name", "url", "test_type"} <= set(rec)
        assert rec["url"].startswith("https://www.shl.com/")


def test_chat_tolerates_messy_input():
    # extra fields, missing content, unknown role -> must not 422
    r = client.post("/chat", json={"messages": [{"role": "user", "content": "hi", "extra": 1},
                                                 {"role": "tool"}]})
    assert r.status_code == 200
    assert set(r.json()) == {"reply", "recommendations", "end_of_conversation"}


def test_chat_empty_messages():
    r = client.post("/chat", json={"messages": []})
    assert r.status_code == 200
    assert r.json()["recommendations"] == []

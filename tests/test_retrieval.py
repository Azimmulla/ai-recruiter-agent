"""Retrieval sanity tests (no LLM)."""
import pytest

from app.catalog import get_catalog
from app.retrieval import get_retriever, tokenize

catalog = get_catalog()
pytestmark = pytest.mark.skipif(len(catalog) == 0, reason="catalog.json not built yet")


def test_tokenize_keeps_tech_terms():
    toks = tokenize("ADO.NET and C++ for a developer")
    assert "net" in toks and "c++" in toks


def test_search_returns_candidates():
    res = get_retriever().search("java developer backend", k=10)
    assert 1 <= len(res) <= 10
    assert all("url" in r and "id" in r for r in res)


def test_test_type_boost_surfaces_personality():
    res = get_retriever().search("personality and behaviour assessment", k=10, test_types=["P"])
    assert any("P" in r["test_type"] for r in res)

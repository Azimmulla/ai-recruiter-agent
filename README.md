# SHL Conversational Assessment Recommender

A stateless conversational agent that takes a recruiter from a vague hiring intent to a
grounded shortlist of **SHL Individual Test Solutions**. It clarifies vague queries,
recommends 1–10 assessments, refines on new constraints, compares named assessments, and
refuses anything out of scope — and it can only ever return names/URLs that exist in the
scraped catalog.

Built for the SHL Labs take-home. Stack: **FastAPI** + **Gemini** (agent reasoning) +
local **hybrid retrieval** (fastembed `bge-small` dense + BM25 sparse, fused with RRF).

## API

`GET /health` → `{"status": "ok"}` (HTTP 200).

`POST /chat` — stateless; send the full history each call:

```json
{ "messages": [
    {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "Sure. What seniority level?"},
    {"role": "user", "content": "Mid-level, around 4 years"}
]}
```

Response:

```json
{
  "reply": "Here are some assessments that fit a mid-level Java dev with stakeholder needs.",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "Occupational Personality Questionnaire OPQ32r", "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

`recommendations` is `[]` while clarifying or refusing; 1–10 items once committed.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then put your Gemini key in .env (GEMINI_API_KEY)

# (optional) rebuild the data artifacts — they are committed, so this is not required:
python -m scraper.scrape_catalog     # -> data/catalog.json  (393 Individual Test Solutions)
python -m scraper.build_index        # -> data/embeddings.npy

uvicorn app.main:app --reload        # http://127.0.0.1:8000  (docs at /docs)
```

Get a free Gemini key at https://aistudio.google.com/apikey.

## Tests & evaluation

```bash
pytest                    # 14 offline tests (schema contract, retrieval, grounding, behaviors)
python -m eval.run_eval   # mean Recall@10 over eval/traces.json
python -m eval.probes     # behavior probe pass-rate (refuse / clarify / refine / compare / no-hallucination)
```

## Deploy (Render)

1. Push to GitHub.
2. Render → **New → Blueprint**, select the repo (uses [render.yaml](render.yaml)).
3. Set `GEMINI_API_KEY` in the dashboard. Data artifacts are committed, so the service
   boots without scraping; the first `/health` is well within the 2-minute cold-start window.

## Layout

```
app/         FastAPI service, agent orchestration, retrieval, Gemini layer, prompts
scraper/     Wayback catalog scraper + embedding index builder
data/        catalog.json + embeddings.npy (committed)
eval/        Recall@10 replay harness, behavior probes, labeled traces
tests/       pytest suite
```

See [approach.md](approach.md) for design rationale, retrieval setup, prompt design, what
didn't work, and AI-tool usage.

## Note on the data source

The live SHL site has since been restructured and the flat product-catalog table no longer
exists, so the catalog is reconstructed from the **Wayback Machine** snapshot the assignment
was written against (stable + reproducible). Scope is restricted to Individual Test Solutions
via a slug classifier validated at 100% against index-page ground-truth labels; Pre-packaged
Job Solutions are excluded.

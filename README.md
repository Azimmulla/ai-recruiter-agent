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
python -m scraper.load_official_catalog   # -> data/catalog.json (377 items, official SHL catalog)
python -m scraper.build_index             # -> data/embeddings.npy

uvicorn app.main:app --reload        # http://127.0.0.1:8000  (docs at /docs)
```

Get a free Gemini key at https://aistudio.google.com/apikey.

## Tests & evaluation

```bash
pytest                          # 19 offline tests (schema contract, retrieval, grounding, behaviors)
python -m eval.parse_conversations   # GenAI_SampleConversations/*.md -> eval/traces.json (labeled)
python -m eval.run_eval         # mean Recall@10 over the 10 official sample traces (~0.68)
python -m eval.probes           # behavior probe pass-rate, 8/8 (refuse / clarify / refine / compare / no-hallucination)
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

## Data source

The catalog is built from the **official SHL catalog JSON** provided for the task (377 items)
via [scraper/load_official_catalog.py](scraper/load_official_catalog.py) — its `link` URLs and
item set are exactly what the evaluator grades against. The raw file is cached at
`data/official_catalog.json` for a reproducible offline build.

A standalone Wayback-Machine scraper ([scraper/scrape_catalog.py](scraper/scrape_catalog.py))
is kept as a fallback for reconstructing the catalog from the live/archived site if the JSON
is unavailable.

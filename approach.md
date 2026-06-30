# Approach — SHL Conversational Assessment Recommender

## Problem framing & design choices

The task is an agent that must do four things well (clarify, recommend, refine, compare),
stay strictly in scope, and never hallucinate a name or URL — under a stateless API, an
8-turn cap, and a 30s/call budget. I optimised for the three things the brief says sink most
submissions: **a contract that never breaks, grounding that makes hallucination structurally
impossible, and real evaluation.**

**Architecture: a thin Gemini "brain" over a deterministic retrieval backbone.** Each
`POST /chat` runs: (1) a **router** — one Gemini structured-JSON call that reads the whole
history and returns an `action` (clarify / recommend / refine / compare / refuse), a distilled
`search_query`, and implied `test_types`; (2) **dispatch** — retrieve, then a Gemini
**reranker** selects/ orders the shortlist by id; (3) **assembly** — recommendations are built
from `catalog.json` **by id only**. The LLM never emits names or URLs, so a hallucinated URL
cannot reach the response. Every id is validated against the catalog and the list is capped to
1–10. The whole entrypoint is wrapped so it always returns a schema-valid object.

**Stateless** is handled by re-deriving all constraints from the full history on every call
(the router is explicitly told to combine constraints across turns), so "refine" updates the
shortlist instead of restarting.

## Data & retrieval

The live SHL catalog page has been restructured and the flat table is gone, so I rebuilt the
catalog from the **Wayback Machine** — the static per-product detail pages from the era the
assignment targets (stable, reproducible, and likely matching the grader's snapshot). I
enumerated detail pages via the Wayback CDX API (**393 Individual Test Solutions**), parsing
name, description, job levels, languages, length, test-type keys, and remote-testing flag.
**Scope** (Individual vs Pre-packaged Job Solutions) is enforced by a slug classifier that I
validated at **100% against 69 index-page ground-truth labels** before trusting it on the full
set.

**Retrieval is local and hybrid:** fastembed `bge-small-en-v1.5` dense vectors (precomputed at
build, committed) + BM25 sparse, fused with **Reciprocal Rank Fusion**. Dense captures intent
("works with stakeholders" → personality); BM25 nails exact tokens ("Java", "ADO.NET",
"Verify"). Catalog embeddings are precomputed so the server only embeds the short query per
request — fast cold start, fits Render's free tier. I chose fastembed over sentence-transformers
specifically to avoid PyTorch, which would blow the free-tier build/memory limits.

**Recall robustness:** because Recall@10 is order-independent, the shortlist is anchored on the
top deterministic retrieval matches (never droppable by the reranker), then refined by the LLM,
with each requested test-type guaranteed a slot and a pad-to-floor safety net. This keeps recall
strong even when Gemini is rate-limited (free tier) and the reranker returns nothing.

## Prompt design

Three focused prompts. The **router** encodes scope, refusal rules, the "don't recommend on a
vague turn-1 query — ask once, never over-clarify, recommend the moment a role/JD is present"
policy, the SHL test-type legend, and an explicit JSON shape. The **reranker** may only choose
from the given candidate ids and is told to include every clearly relevant item and its close
variants. The **comparator** answers strictly from supplied catalog facts and admits when an
item isn't in the catalog rather than inventing detail. Defence-in-depth: a **deterministic
scope guard** refuses obvious injection/legal inputs *before* any LLM call, so refusal holds
even if Gemini is unavailable.

## Evaluation

A replay harness mirrors the grader: it feeds a trace's scripted user turns to `/chat`, answers
extra questions with "no preference", and stops at the first shortlist. I authored 10 labelled
traces spanning all four behaviors; expected names resolve against the live catalog by fuzzy
match. Plus 8 **behavior probes** with binary assertions (refuse off-topic/legal, resist
injection, no-recommend-on-vague-turn-1, refine adds personality, grounded compare, and a
"every returned URL ∈ catalog" invariant).

Results: **mean Recall@10 = 0.77**; **probes 8/8** (7/8 before adding the deterministic legal
guard). Retrieval alone places the labelled items in the top candidates for every trace, so the
remaining recall gap is dominated by label specificity (e.g. the catalog has many OPQ "report"
variants vs. the base OPQ32r questionnaire) rather than retrieval misses.

## What didn't work / how I measured it

- **Gemini `response_schema` with default values silently 500'd** every structured call, so the
  agent ran purely on its fallback (identical Recall before/after every prompt change — the tell
  that the LLM wasn't actually steering). Fix: request `application/json` and parse the text.
- **`gemini-2.5-flash` truncated JSON** — as a "thinking" model it spent the output budget on
  reasoning (`finish_reason=MAX_TOKENS`). Switching the primary to **`gemini-2.0-flash`**
  (non-thinking, faster, higher free-tier limits) fixed it; 2.5-flash remains the fallback.
- **Reranker dropping the top retrieval match** (it once returned Core Java *Entry* over the
  rank-1 *Advanced*, plus a duplicate name). Fix: retrieval-anchored selection + name dedup,
  which lifted that trace 0.33 → 0.67 and the mean 0.73 → 0.77.
- Each change was measured with the same Recall@10 + probe harness; numbers above are the deltas.

## Stack & AI-tool usage

FastAPI/Pydantic (free, exact schema), Gemini (`google-genai`), fastembed + rank-bm25 + numpy
(local, free, no GPU), Render (free). I used an AI coding assistant (Claude) for scaffolding,
the Wayback scraping strategy, and iterating on the eval-driven fixes above; every design
decision, the data-scoping validation, and the retrieval/agent logic are my own and defensible.

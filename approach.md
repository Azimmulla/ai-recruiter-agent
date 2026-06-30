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

The catalog is the **official SHL catalog JSON** (377 items): I transform it to the app schema,
mapping the full-name `keys` to single-letter test-type codes and keeping the official `link`
URLs verbatim (exactly what the grader validates against). The raw file is cached for a
reproducible offline build. (A Wayback-Machine scraper is retained as a fallback — built earlier
when the live catalog table had been removed from the site.)

**Retrieval is local and hybrid:** fastembed `bge-small-en-v1.5` dense vectors (precomputed at
build, committed) + BM25 sparse, fused with **Reciprocal Rank Fusion**. Dense captures intent
("works with stakeholders" → personality); BM25 nails exact tokens ("Java", "Spring", "Docker").
Embeddings are precomputed so the server only embeds the short query per request — fast cold
start, fits Render's free tier. fastembed (ONNX) avoids PyTorch, which would blow the free-tier
build/memory limits.

**Battery composition.** The sample traces showed the expected shortlist is a *battery*, not a
single cluster: role-specific skills tests + a general cognitive ability measure (Verify G+) + a
personality measure (OPQ32r). So the agent (1) extracts the **distinct skills** of a request — a
full-stack JD becomes ["Core Java","Spring","SQL","AWS","Docker"] — and retrieves a precise match
per skill, and (2) anchors the battery with a cognitive and a personality instrument by default,
honouring refinements that remove them. This mirrors real SHL practice and is what lifted recall
most.

**Recall robustness:** the shortlist is assembled by priority (per-skill matches → battery
anchors → top retrieval → reranker refinements → fill to 10), deduped by name, anchored on
deterministic retrieval so the LLM reranker can't drop a strong match. Because Recall@10 is
order-independent and has no precision penalty, we fill all 10 slots. Everything works even when
Gemini is rate-limited (free tier): the deterministic backbone keeps recommendations grounded
and on-target.

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

A replay harness mirrors the grader: it feeds a trace's user turns to `/chat`, answers extra
questions with "no preference", and stops at the first shortlist (the simulated user ends when
the agent recommends). I parse the **10 official sample conversations** into labelled traces —
the user turns drive the replay and the final recommendations table is the relevance label set
(all 10 resolved 100% against the catalog). Plus 8 **behavior probes** with binary assertions
(refuse off-topic/legal, resist injection, no-recommend-on-vague-turn-1, refine adds personality,
grounded compare, and a "every returned URL ∈ catalog" invariant) and 19 pytest unit tests.

Results on the official traces: **mean Recall@10 = 0.68**, **probes 8/8**. Measured improvement:
battery composition + per-skill retrieval took the mean from **0.47 → 0.65**, and filling all 10
slots → **0.68**. The remaining gap is largely the scripted replay's limitation — several labels
depend on facts the persona only reveals in later turns, which the grader's *dynamic* user would
surface in response to the agent's clarifying questions.

## What didn't work / how I measured it

- **Gemini `response_schema` with default values silently 500'd** every structured call, so the
  agent ran purely on its fallback (identical Recall before/after every prompt change — the tell
  that the LLM wasn't actually steering). Fix: request `application/json` and parse the text.
- **`gemini-2.5-flash` truncated JSON** — as a "thinking" model it spent the output budget on
  reasoning (`finish_reason=MAX_TOKENS`). Switching the primary to **`gemini-2.0-flash`**
  (non-thinking, faster, higher free-tier limits) fixed it; 2.5-flash remains the fallback.
- **Reranker dropping the top retrieval match** (it once returned Core Java *Entry* over the
  rank-1 *Advanced*, plus a duplicate name). Fix: retrieval-anchored selection + name dedup.
- **Narrow single-cluster shortlists** were the biggest recall leak: the labels are batteries.
  Per-skill retrieval + cognitive/personality anchors moved the mean from 0.47 → 0.65.
- Each change was measured against the 10 official sample traces; numbers above are the deltas.

## Stack & AI-tool usage

FastAPI/Pydantic (free, exact schema), Gemini (`google-genai`), fastembed + rank-bm25 + numpy
(local, free, no GPU), Render (free). I used an AI coding assistant (Claude) for scaffolding,
the Wayback scraping strategy, and iterating on the eval-driven fixes above; every design
decision, the data-scoping validation, and the retrieval/agent logic are my own and defensible.

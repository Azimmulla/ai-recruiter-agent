"""FastAPI service: GET /health and POST /chat.

/health is dependency-free and always returns 200 quickly (readiness). On
startup a background thread warms the catalog, retriever, and embedding model so
the first /chat doesn't pay the model-load cost — all wrapped so a warmup
failure can never crash the process.
"""
from __future__ import annotations

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agent import handle
from app.schemas import ChatRequest, ChatResponse, HealthResponse


def _warmup() -> None:
    try:
        from app.retrieval import get_retriever

        get_retriever().search("software developer", k=3)  # loads model + indexes
    except Exception:
        pass


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=_warmup, daemon=True).start()
    yield


app = FastAPI(title="SHL Assessment Recommender", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/")
def root() -> dict:
    return {
        "service": "SHL Assessment Recommender",
        "status": "ok",
        "usage": "POST /chat with {\"messages\": [{\"role\": \"user\", \"content\": \"...\"}]}",
        "endpoints": {"health": "GET /health", "chat": "POST /chat", "docs": "GET /docs"},
    }


@app.get("/health", response_model=HealthResponse)
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> dict:
    messages = [m.model_dump() for m in req.messages]
    return handle(messages)

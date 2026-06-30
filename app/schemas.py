"""Pydantic models = the hard API contract.

Input is parsed leniently (unknown fields ignored, missing content tolerated) so
a slightly-off request never 422s the evaluator. Output shape is fixed exactly to
the spec: reply, recommendations[1..10 or empty], end_of_conversation.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: str = "user"
    content: str = ""


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    messages: list[Message] = Field(default_factory=list)


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str = ""


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation] = Field(default_factory=list, max_length=10)
    end_of_conversation: bool = False


class HealthResponse(BaseModel):
    status: str = "ok"

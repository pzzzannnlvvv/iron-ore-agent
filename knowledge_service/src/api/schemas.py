"""Pydantic 模型,对齐原版 KnowledgeSearchRequestVO / KnowledgeSearchVO。"""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel


class KnowledgeSearchRequest(BaseModel):
    content: str
    topK: int = 10
    reRank: int = 1
    mode: str = "mix"  # knn / bm25 / mix
    publishStartDate: str | None = None
    publishEndDate: str | None = None
    sourceType: int | None = None
    kbId: Any | None = None  # 智能体不关注


class SearchResponse(BaseModel):
    code: int = 200
    data: list[dict]
    timings: dict | None = None


class IndexResponse(BaseModel):
    code: int = 200
    data: dict

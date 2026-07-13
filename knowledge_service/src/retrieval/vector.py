"""向量召回:Chroma KNN(用已算好的 query 向量)。对应方案 vector.py。"""
from __future__ import annotations

from src.pipeline.store import query_knn


def search_vec(query_embedding: list[float], top_k: int = 10) -> list[dict]:
    return query_knn(query_embedding, top_k=top_k)

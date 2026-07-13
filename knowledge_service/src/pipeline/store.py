"""
Store:Chroma 持久化(cosine 空间)。metadata 对齐原版:fileName/source/sourceType/chunkIndex。
add_chunks / query_knn / get_all_chunks / count / reset。
对应方案 store.py。
"""
from __future__ import annotations

from typing import Any

import chromadb

from src.config import CHROMA_DIR, COLLECTION_NAME

_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def add_chunks(chunks: list[dict]) -> None:
    """chunks: [{id, content, embedding, metadata}]"""
    if not chunks:
        return
    col = _get_collection()
    col.add(
        ids=[c["id"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        documents=[c["content"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )


def query_knn(query_embedding: list[float], top_k: int = 10) -> list[dict]:
    col = _get_collection()
    res = col.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for i in range(len(res["ids"][0])):
        dist = res["distances"][0][i]
        hits.append({
            "id": res["ids"][0][i],
            "content": res["documents"][0][i],
            "metadata": res["metadatas"][0][i],
            "score": float(max(0.0, 1.0 - dist)),  # cosine 距离 -> 相似度
        })
    return hits


def get_all_chunks() -> list[dict]:
    """全量取回(BM25 建索引用)。"""
    col = _get_collection()
    res = col.get(include=["documents", "metadatas"])
    return [
        {"id": i, "content": d, "metadata": m}
        for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])
    ]


def count() -> int:
    try:
        return _get_collection().count()
    except Exception:
        return 0


def reset() -> None:
    """删库重建(重新入库前调)。"""
    global _client, _collection
    if _client is not None:
        try:
            _client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        _collection = None

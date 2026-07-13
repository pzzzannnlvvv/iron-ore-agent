"""
BM25 召回:rank_bm25 + jieba 中文分词。
首次用时从 Chroma 拉全量 chunk 建索引,进程内缓存。对应方案 bm25.py。
"""
from __future__ import annotations

import jieba
from rank_bm25 import BM25Okapi

from src.logger import log


def _tokenize(text: str, for_search: bool = False) -> list[str]:
    tokens = jieba.cut_for_search(text) if for_search else jieba.cut(text)
    return [w for w in tokens if w.strip()]


class BM25Index:
    def __init__(self, chunks: list[dict]):
        self.ids = [c["id"] for c in chunks]
        self.docs = [c["content"] for c in chunks]
        self.metas = [c.get("metadata") or {} for c in chunks]
        self.tokenized = [_tokenize(d) for d in self.docs]
        self.bm25 = BM25Okapi(self.tokenized)
        log.info("BM25 索引建好: %d 个 chunk", len(self.ids))

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        scores = self.bm25.get_scores(_tokenize(query, for_search=True))
        ranked = sorted(zip(self.ids, self.docs, self.metas, scores), key=lambda x: -x[3])[:top_k]
        return [{"id": i, "content": d, "metadata": m, "score": float(s)} for i, d, m, s in ranked]


_index: BM25Index | None = None


def get_index() -> BM25Index:
    """懒加载:首次从 Chroma 拉全量 chunk 建 BM25。"""
    global _index
    if _index is None:
        from src.pipeline.store import get_all_chunks

        chunks = get_all_chunks()
        if not chunks:
            raise RuntimeError("Chroma 里没数据,先入库(POST /knowledge-base/index 或 scripts/ingest.py)")
        _index = BM25Index(chunks)
    return _index


def reset_index() -> None:
    global _index
    _index = None

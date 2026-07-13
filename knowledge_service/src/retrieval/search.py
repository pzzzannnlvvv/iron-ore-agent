"""
检索主链路:对齐原版 KnowledgeSearchServiceImpl。
embed(query) -> 三模式召回(KNN / BM25 / Mix 按 RRF 融合) -> 降级(embedding 失败用纯 BM25)
-> 可选 bge-reranker 重排 -> 日期/来源过滤 -> 三段耗时日志 -> 返回 KnowledgeSearchVO 列表。
对应方案 search.py。
"""
from __future__ import annotations

import time
from typing import Any

from src.logger import log
from src.pipeline.embedder import embed
from src.pipeline.reranker import rerank as rerank_fn
from src.retrieval import bm25, hybrid, vector


def _to_vo(hit: dict) -> dict:
    """对齐原版 KnowledgeSearchVO 字段。"""
    md = hit.get("metadata") or {}
    return {
        "id": hit["id"],
        "content": hit["content"],
        "fileId": md.get("fileId"),
        "fileName": md.get("fileName"),
        "contentTags": md.get("contentTags") or [],
        "publishTime": md.get("publishTime"),
        "sourceType": md.get("sourceType", 1),
        "source": md.get("source"),
        "indexTime": md.get("indexTime"),
        "score": round(float(hit.get("score", 0.0)), 6),
    }


def _filter(
    hits: list[dict],
    publish_start: str | None,
    publish_end: str | None,
    source_type: int | None,
) -> list[dict]:
    """日期 / 来源过滤(发布日期在 Python 侧后处理,Chroma 过滤弱)。"""
    out = []
    for h in hits:
        md = h.get("metadata") or {}
        pt = md.get("publishTime")
        if publish_start and pt and pt < publish_start:
            continue
        if publish_end and pt and pt > publish_end:
            continue
        if source_type is not None and md.get("sourceType") != source_type:
            continue
        out.append(h)
    return out


def search(
    content: str,
    topK: int = 10,
    reRank: int = 1,
    mode: str = "mix",
    publishStartDate: str | None = None,
    publishEndDate: str | None = None,
    sourceType: int | None = None,
) -> tuple[list[dict], dict]:
    """
    Args:
        content: 查询文本(必填)
        topK: 返回条数
        reRank: 0=不重排 1=重排
        mode: knn / bm25 / mix
    Returns:
        (hits_vo, timings)  timings 含 embeddingTimeCost/searchTime/reRankTimeCost/totalHits
    """
    timings: dict[str, Any] = {}
    pool_k = max(topK * 3, 30)  # 融合/重排候选池,取多一些再裁

    # 1. embedding(可能失败 -> 降级纯 BM25,照搬原版降级逻辑)
    t0 = time.time()
    emb_ok = True
    qvec = None
    try:
        qvec = embed(content)
    except Exception as e:
        log.warning("embedding 失败,降级纯 BM25: %s", e)
        emb_ok = False
    timings["embeddingTimeCost"] = round(time.time() - t0, 3)

    # 2. 召回 + 融合
    t1 = time.time()
    bm_hits = bm25.get_index().search(content, top_k=pool_k)
    if not emb_ok or mode == "bm25":
        merged = bm_hits
    elif mode == "knn":
        merged = vector.search_vec(qvec, top_k=pool_k)
    else:  # mix
        vec_hits = vector.search_vec(qvec, top_k=pool_k)
        merged = hybrid.rrf([vec_hits, bm_hits])
    merged = _filter(merged, publishStartDate, publishEndDate, sourceType)
    candidates = merged[: max(topK * 2, 20)]
    timings["searchTime"] = round(time.time() - t1, 3)

    # 3. 重排
    t2 = time.time()
    if reRank == 1 and candidates:
        docs = [c["content"] for c in candidates]
        try:
            scores = rerank_fn(content, docs)
            for c, s in zip(candidates, scores):
                c["score"] = float(s)
            candidates.sort(key=lambda x: -x["score"])
        except Exception as e:
            log.warning("rerank 失败,用融合分排序: %s", e)
    candidates = candidates[:topK]
    timings["reRankTimeCost"] = round(time.time() - t2, 3)
    timings["totalHits"] = len(candidates)

    log.info("检索完成 mode=%s topK=%d timings=%s", mode, topK, timings)
    return [_to_vo(c) for c in candidates], timings

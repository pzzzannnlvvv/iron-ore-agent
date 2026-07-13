"""
RRF(Reciprocal Rank Fusion)融合:把多路召回按排名融合。
score = sum(1 / (k + rank))。对应方案 hybrid.py。
"""
from __future__ import annotations


def rrf(rankings: list[list[dict]], k: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    meta: dict[str, dict] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking):
            hid = hit["id"]
            scores[hid] = scores.get(hid, 0.0) + 1.0 / (k + rank + 1)
            if hid not in meta:
                meta[hid] = hit
    merged = sorted(scores.items(), key=lambda x: -x[1])
    return [{**meta[i], "score": s} for i, s in merged]

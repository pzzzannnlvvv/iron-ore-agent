"""
bge-reranker-v2-m3 重排封装。

不用 FlagEmbedding.FlagReranker——它在 transformers 5.x 下调用已废弃的
tokenizer.prepare_for_model 会报 AttributeError。改用 transformers 直接加载
XLMRobertaForSequenceClassification(标准 cross-encoder 写法),跨版本稳定。

本地路径优先读 BGE_RERANKER_PATH。CPU 推理。
对应 knowledge_be接入方案 第五节 reranker.py,search.py 混合召回后调 rerank()。
"""
from __future__ import annotations

import os
from functools import lru_cache

import torch

DEFAULT_BGE_RERANKER_PATH = "C:/Users/admin/Desktop/Embedding和重排模型/bge-reranker-v2-m3"


@lru_cache(maxsize=1)
def get_model():
    """懒加载 reranker,进程内只加载一次。返回 (tokenizer, model)。"""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path = os.getenv("BGE_RERANKER_PATH", DEFAULT_BGE_RERANKER_PATH)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return tokenizer, model


def rerank(
    query: str,
    documents: list[str],
    normalize: bool = True,
    max_length: int = 512,
) -> list[float]:
    """对每个 (query, doc) 打分,返回与 documents 等长的分数列表。

    normalize=True 时分数经 sigmoid 落在 (0,1),越大越相关。
    max_length:CPU 上 512 够用且快;chunk 本身就切到几百 token。
    """
    if not documents:
        return []
    tokenizer, model = get_model()
    pairs = [[query, doc] for doc in documents]
    inputs = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**inputs).logits.squeeze(-1)  # (N,)
    scores = torch.sigmoid(logits) if normalize else logits
    return scores.tolist()


if __name__ == "__main__":
    q = "铁矿石库存预测"
    docs = ["铁矿石港口库存本周下降", "今天天气不错", "库存预测模型 MDA 84%"]
    print(rerank(q, docs))

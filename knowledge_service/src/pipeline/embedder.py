"""
bge-m3 embedding 封装。

本地模型路径优先读环境变量 BGE_M3_PATH,缺省指向桌面下载位置。
用 FlagEmbedding.BGEM3FlagModel 加载(bge 官方实现,与原版 knowledge_be 语义对齐),
产出 1024 维 dense 向量。CPU 推理 use_fp16=False。

对应 knowledge_be接入方案 第五节 embedder.py,后续 store.py / search.py 调 embed()。
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

# 缺省指向你下载模型的位置(可被 BGE_M3_PATH 覆盖)
DEFAULT_BGE_M3_PATH = "C:/Users/admin/Desktop/Embedding和重排模型/bge-m3"


@lru_cache(maxsize=1)
def get_model():
    """懒加载 bge-m3,进程内只加载一次。"""
    from FlagEmbedding import BGEM3FlagModel

    path = os.getenv("BGE_M3_PATH", DEFAULT_BGE_M3_PATH)
    use_fp16 = os.getenv("BGE_USE_FP16", "0") == "1"  # 本机无 GPU,默认 fp32
    return BGEM3FlagModel(path, use_fp16=use_fp16)


def embed(text: str, normalize: bool = True) -> list[float]:
    """单条文本 -> 1024 维向量。"""
    model = get_model()
    out = model.encode(
        [text],
        batch_size=1,
        max_length=8192,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    vec = np.asarray(out["dense_vecs"][0], dtype=np.float32)
    if normalize:
        n = np.linalg.norm(vec)
        if n > 0:
            vec = vec / n
    return vec.tolist()


def embed_batch(texts: list[str], normalize: bool = True) -> list[list[float]]:
    """批量编码(入库用)。"""
    model = get_model()
    out = model.encode(
        texts,
        batch_size=8,
        max_length=8192,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    vecs = np.asarray(out["dense_vecs"], dtype=np.float32)
    if normalize:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.clip(norms, 1e-12, None)
    return vecs.tolist()


if __name__ == "__main__":
    v = embed("铁矿石港口库存周度预测")
    print(f"dim={len(v)} head={[round(x, 4) for x in v[:5]]}")

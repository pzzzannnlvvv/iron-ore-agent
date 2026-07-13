"""
冒烟测试:验证两个本地 bge 模型能加载、能出向量/分数。

运行(在 knowledge_service venv 激活后):
    python scripts/smoke_test_models.py

首次会加载 ~2.3G 权重到内存,CPU 下约 30~90 秒。
"""
import sys
import time
from pathlib import Path

import numpy as np

# 让脚本能 import src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.embedder import embed, embed_batch
from src.pipeline.reranker import rerank


def main():
    print("== 1. 加载 bge-m3 并编码单条 ==")
    t = time.time()
    v = embed("铁矿石港口库存周度预测")
    print(f"  耗时 {time.time()-t:.1f}s, 维度={len(v)}, 前5维={[round(x,4) for x in v[:5]]}")
    assert len(v) == 1024, f"期望 1024 维, 实际 {len(v)}"

    print("\n== 2. 批量编码 + 余弦相似度(向量已归一化,点积即余弦)==")
    t = time.time()
    texts = ["铁矿石港口库存下降", "钢材需求走弱", "澳巴发运量回升"]
    mat = embed_batch(texts)
    print(f"  耗时 {time.time()-t:.1f}s, 形状={len(mat)}x{len(mat[0])}")
    q = np.array(embed("铁矿石库存"))
    for text, vec in zip(texts, mat):
        print(f"  sim('{text}')={float(q @ np.array(vec)):.4f}")

    print("\n== 3. 加载 bge-reranker-v2-m3 并重排 ==")
    t = time.time()
    query = "铁矿石库存最近怎么样"
    docs = [
        "本周45个港口铁矿石库存环比下降120万吨",
        "铁矿石价格震荡运行,成交一般",
        "澳洲飓风影响发运,后期到港或减少",
        "今天适合出门散步",
    ]
    scores = rerank(query, docs)
    print(f"  耗时 {time.time()-t:.1f}s")
    for doc, sc in sorted(zip(docs, scores), key=lambda x: -x[1]):
        print(f"  {sc:.4f}  {doc}")
    assert scores[0] > scores[3], "相关文档分数应高于无关文档"

    print("\n✅ 冒烟测试通过:两个模型加载 & 推理正常")


if __name__ == "__main__":
    main()

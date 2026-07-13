"""
检索测试:验证混合召回 + 重排,打印三段耗时 + 命中片段。

运行(先跑过 ingest.py 入库):
    .venv/Scripts/python.exe scripts/test_search.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.search import search  # noqa: E402


def show(query: str, reRank: int = 1, mode: str = "mix", topK: int = 5):
    print(f"\n=== query: {query}  mode={mode} reRank={reRank} ===")
    t = time.time()
    hits, timings = search(query, topK=topK, reRank=reRank, mode=mode)
    print(f"总耗时 {time.time()-t:.2f}s  timings={timings}")
    for h in hits:
        snippet = h["content"][:90].replace("\n", " ")
        print(f"  {h['score']:.4f}  [{h['fileName']}]  {snippet}")


if __name__ == "__main__":
    queries = [
        "铁矿石港口库存预测结果怎么样",
        "LightGBM 超参优化怎么做的",
        "特征工程里时间特征有哪些",
        "MDA 方向准确率是多少",
    ]
    for q in queries:
        show(q, reRank=1, mode="mix")

    print("\n--- 对比 reRank=0 vs reRank=1(同一 query)---")
    show("铁矿石库存", reRank=0, mode="mix")
    show("铁矿石库存", reRank=1, mode="mix")

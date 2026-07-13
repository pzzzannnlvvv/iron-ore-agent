"""
简单评估:几个 query + 期望命中的文件,算 hit@5(非严谨 RAGAS,够看检索质量)。
走 HTTP 调 knowledge_service,复用服务端已加载的模型。

运行(需 knowledge_service 在 :8092 运行):
    .venv/Scripts/python.exe scripts/eval.py
"""
import sys

import httpx

URL = "http://127.0.0.1:8092/knowledge-base/report/search"

# (query, 期望命中的 fileName 子串列表;top5 里任一文件名含任一子串即算命中)
CASES = [
    ("铁矿石库存预测准确率是多少", ["report_20260707", "report_20260709", "项目完整流程总结"]),
    ("LightGBM 超参优化怎么做的", ["LGBM超参优化", "report_20260709", "07_model_training"]),
    ("时间特征怎么构造", ["时间特征", "feature_engineering_summary"]),
    ("模型过拟合怎么判断", ["report_20260709", "07_model_training", "项目完整流程总结"]),
    ("数据清洗做了什么", ["数据清洗"]),
    ("项目整体建模流程", ["项目完整流程总结", "项目执行计划"]),
    ("标的滞后特征是什么", ["标的滞后特征", "feature_engineering_summary"]),
    ("测试集有多少行", ["项目完整流程总结", "07_model_training", "report_20260709"]),
]


def do_search(query: str, top_k: int = 5) -> list[dict]:
    r = httpx.post(URL, json={"content": query, "topK": top_k, "reRank": 1, "mode": "mix"}, timeout=180)
    r.raise_for_status()
    return r.json().get("data", [])


def main():
    hit_n = 0
    for query, expected in CASES:
        results = do_search(query)
        got = [r["fileName"] or "" for r in results]
        ok = any(any(e in f for f in got) for e in expected)
        hit_n += ok
        print(f"{'✓' if ok else '✗'} {query}")
        print(f"   期望含: {expected}")
        print(f"   top5: {got}")
    rate = hit_n / len(CASES)
    print(f"\nhit@5 = {hit_n}/{len(CASES)} = {rate:.0%}")
    sys.exit(0 if rate >= 0.6 else 1)


if __name__ == "__main__":
    main()

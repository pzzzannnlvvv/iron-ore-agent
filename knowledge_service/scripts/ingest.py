"""
入库脚本:读 data/corpus -> 切片 -> bge-m3 编码 -> 写 Chroma。

运行(在 knowledge_service 下):
    .venv/Scripts/python.exe scripts/ingest.py
首次会加载 bge-m3(~10s)+ 批量编码,几百 chunk 约 30~60s。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import store  # noqa: E402
from src.pipeline.ingest import ingest  # noqa: E402

if __name__ == "__main__":
    result = ingest()
    print(f"\n入库完成: {result['docs']} 文档, {result['chunks']} chunks")
    print(f"Chroma 当前 chunk 数: {store.count()}")

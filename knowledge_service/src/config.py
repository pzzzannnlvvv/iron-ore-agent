"""集中配置:路径、切片参数、模型路径(都可用 .env 覆盖)。"""
import os
from pathlib import Path

from dotenv import load_dotenv

# knowledge_service 根目录
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# 语料与向量库
CORPUS_DIR = Path(os.getenv("CORPUS_DIR", str(ROOT / "data" / "corpus")))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(ROOT / "data" / "chroma")))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "iron_ore_kb")

# 模型
BGE_M3_PATH = os.getenv("BGE_M3_PATH", "C:/Users/admin/Desktop/Embedding和重排模型/bge-m3")
BGE_RERANKER_PATH = os.getenv("BGE_RERANKER_PATH", "C:/Users/admin/Desktop/Embedding和重排模型/bge-reranker-v2-m3")

# 切片
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# 服务
SERVICE_PORT = int(os.getenv("KNOWLEDGE_SERVICE_PORT", "8092"))

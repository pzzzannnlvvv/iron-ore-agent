# knowledge_service

项目复现2 的知识库检索服务(Python 复刻 knowledge_be 的 RAG 核心)。整体方案见 `../knowledge_be接入方案.md`,复刻对照见 `复刻笔记.md`。

## 现状(阶段 0~6 完成)

- ✅ 模型层:`pipeline/embedder.py`(bge-m3)、`pipeline/reranker.py`(bge-reranker-v2-m3)
- ✅ 入库:`loader + splitter + store(Chroma)`,15 文档 / 237 chunks
- ✅ 检索主链路:BM25(jieba)+ KNN + RRF + bge-reranker + 降级 + 三段耗时
- ✅ FastAPI:`/knowledge-base/report/search`、`/knowledge-base/index`、`/health`(:8092)
- ✅ mcp_server:`fetch_knowledge` 工具(见 `../mcp_server/server.py`)
- ✅ agent:`.env.dev` 的 DEFAULT/SYNTHESIS 的 researcher+analyst 已加 `fetch_knowledge`
- ✅ 评估:hit@5 = 88%(`scripts/eval.py`)
- ⬜ agent 联调:需启动 agent_service 跑 `generate_report.py`(见「运行」)

## 架构

```
agent_service (LangGraph,:5000)
   │ MCP streamable_http
   ▼
mcp_server (FastMCP,:17000)  —— fetch_knowledge 工具
   │ httpx POST
   ▼
knowledge_service (FastAPI,:8092)  —— /knowledge-base/report/search
   │
   ▼
Chroma(本地文件)+ bge-m3 / bge-reranker(本地推理)
```

## 目录

```
knowledge_service/
├── main.py                     # FastAPI 入口
├── pyproject.toml / .env.example
├── data/{corpus,chroma}/       # 语料 + Chroma 持久化
├── src/
│   ├── config.py / logger.py
│   ├── pipeline/{loader,splitter,embedder,reranker,store,ingest}.py
│   ├── retrieval/{bm25,vector,hybrid,search}.py
│   └── api/{schemas,routes}.py
└── scripts/{smoke_test_models,ingest,test_search,eval}.py
```

## 模型

桌面下载位置(默认路径已写进代码,可用 `BGE_M3_PATH`/`BGE_RERANKER_PATH` 覆盖):

- `C:/Users/admin/Desktop/Embedding和重排模型/bge-m3`
- `C:/Users/admin/Desktop/Embedding和重排模型/bge-reranker-v2-m3`

## 环境

本机无 GPU,CPU 推理。**Python 3.12**(系统 3.14 太新,torch 无 Windows 轮子)。3.12 基础解释器在 `AppData/Roaming/uv/python/cpython-3.12-...`。

```bash
BASE="C:/Users/admin/AppData/Roaming/uv/python/cpython-3.12-windows-x86_64-none/python.exe"
"$BASE" -m venv .venv
VPY=.venv/Scripts/python.exe
"$VPY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU 版,别装 CUDA 那个 2.5G
"$VPY" -m pip install FlagEmbedding chromadb rank-bm25 jieba "fastapi[standard]" uvicorn httpx python-dotenv
```

## 踩坑

`FlagEmbedding.FlagReranker` 在 **transformers 5.x** 下报 `XLMRobertaTokenizer has no attribute prepare_for_model`(5.x 删了该方法)。所以 `reranker.py` 不用 FlagReranker,改用 `AutoModelForSequenceClassification + AutoTokenizer` 直接跑 cross-encoder。`embedder.py` 的 `BGEM3FlagModel` 在 5.x 下正常,没动。

## 运行(4 个终端)

```bash
# 0. 入库(首次,~1min;之后不用重复,除非语料变了)
.venv/Scripts/python.exe scripts/ingest.py

# 1. knowledge_service (:8092)
.venv/Scripts/python.exe -m uvicorn main:app --port 8092 --host 127.0.0.1

# 2. mcp_server (:17000)  —— 在 mcp_server 目录
../mcp_server/.venv/Scripts/python.exe ../mcp_server/server.py

# 3. agent_service (:5000) —— 在 agent_service 目录,然后:
#    ../11_report_generation/scripts/generate_report.py
#    提问示例:"结合知识库里的铁矿石分析文档和预测数据,生成本周分析报告"
```

跑前建议设 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUTF8=1`(模型全本地、修中文乱码)。

## 检索接口

```bash
POST http://127.0.0.1:8092/knowledge-base/report/search
{"content":"铁矿石库存预测结果","topK":5,"reRank":1,"mode":"mix"}
# mode: knn / bm25 / mix(默认)
# reRank: 1=重排(更准,~10s) 0=不重排(快)
```

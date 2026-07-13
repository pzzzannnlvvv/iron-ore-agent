# knowledge_be 复刻并接入项目复现2 · 实施方案

> 目标:用 Python 复刻 `xmschain_knowledge_be` 的 RAG 检索核心,接入 `项目复现2` 已有的 `mcp_server`,让 agent 的 `researcher` 能真正检索知识库(目前 `fetch_knowledge` 工具不存在,`fetch_news`/`fetch_graph` 是占位)。
> 选型已确认:本地 sentence-transformers(bge-m3 + bge-reranker)+ Chroma + rank_bm25 + FastAPI。

---

## 一、原架构:knowledge_be 怎么连接 agent(三层)

agent **不直接调** knowledge_be,中间隔了一个 MCP 中转服务:

```
xmschain_agent (LangGraph, Python)
   │  langchain_mcp_adapters.MultiServerMCPClient
   │  transport=streamable_http,POST 到 MCP server 的 /mcp
   ▼
Xmschain_Mcp (MCP Server,独立服务,:17000)        ← 中转层,暴露 MCP 工具
   │  工具内部用 HTTP 调下游 REST
   ▼
xmschain_knowledge_be (Java,:8091) + 其他数据服务   ← 知识中台
```

### MCP 工具 -> knowledge_be 接口映射(已核实)

| MCP 工具 | knowledge_be 接口 | 入参 | 出参 | 内部逻辑 |
|---|---|---|---|---|
| `fetch_knowledge` | `POST /knowledge-base/report/search`(注释"智能体报告知识库检索") | `KnowledgeSearchRequestVO` | `List<KnowledgeSearchVO>` | embedding->KNN+BM25+Mix 混合召回->bge-reranker 重排->降级 |
| `fetch_news` | `POST /knowledge-base/search-public-opinion`(舆情检索) | `PublicOpinionSearchRequestVO` | `PublicOpinionSearchResponseVO` | 舆情语料检索 |
| `fetch_graph` | `POST /api/graph/agent/searchVertex`(注释"智能体使用") | `GraphVertexQueryVO` | `GraphResultVO` | JanusGraph 语义检索,返回实体+关系 |
| `fetch_subject_data` | `POST /knowledge-base/search-subject-data` | `SubjectSearchRequestVO` | `JSONObject` | 专题数据(反向调 chatmind) |

### 接口契约(复刻版要对齐原版)

`POST /knowledge-base/report/search` 入参(对齐 `KnowledgeSearchRequestVO`):
```json
{
  "content": "铁矿石库存最近怎么样",   // 必填
  "topK": 10,                         // 默认10
  "reRank": 1,                        // 0=不重排 1=重排
  "publishStartDate": "2026-01-01",   // 可选
  "publishEndDate": "2026-07-13",     // 可选
  "sourceType": 1,                    // 1用户上传 2政策舆情
  "kbId": null                        // 智能体不关注
}
```
出参(对齐 `KnowledgeSearchVO`,加一个 `score`):
```json
{"code":200,"data":[
  {"id":"...","content":"...","fileId":"...","fileName":"...",
   "contentTags":[...],"publishTime":"...","sourceType":1,"source":"...","indexTime":"...","score":0.87}
]}
```

### agent 侧工具配置机制(已核实)

工具启用在 `agent_service/.env.dev` 的 `MCP_*_SETTINGS` JSON 里,按 **agent 类型 × 节点** 过滤:
```json
{"servers":{"xmschain-mcp-tools":{"enabled_tools":[
  {"node":"researcher","tools":["fetch_data","fetch_news","model_choice","model_predict","draw_chart"]},
  {"node":"analyst","tools":["fetch_news","fetch_graph"]}
],"transport":"streamable_http","url":"http://127.0.0.1:17000/mcp"}}}
```
`merge_mcp_tools()`(在 `agent_service/src/agents/mcps.py`)启动时按当前 agent 的节点名加载对应工具。**接入点:把 `fetch_knowledge` 加回 researcher/analyst 的 tools 列表。**

---

## 二、项目复现2 现状盘点

| 已有 | 状态 |
|---|---|
| `agent_service/` | ✅ 完整复刻 xmschain_agent,跑通(GLM via Anthropic 协议,免 PG) |
| `mcp_server/server.py` | ✅ 自建 FastMCP(:17000),7 工具 |
| `11_report_generation/` | ✅ 客户端脚本,已生成首份报告 |

`mcp_server/server.py` 现有 7 工具的真实状态(已核实):
- `fetch_data`/`model_choice`/`model_predict`/`fetch_background` -- 读本地建模 outputs,**真实数据,工作正常**
- `fetch_news` -- 占位"无新闻数据"
- `fetch_graph` -- 占位"无图谱数据"
- `fetch_knowledge` -- **不存在**(原版 default 的 researcher 有,复刻时被去掉了)

**缺口 = 没有真正的知识库检索能力。** 接入点很干净:在 `mcp_server/server.py` 加 `fetch_knowledge` 工具,调新建的 knowledge_service。

---

## 三、复刻目标与范围

### 做(RAG 核心,Python)
Load -> Split -> Embed(bge-m3) -> Store(Chroma) -> 混合召回(KNN+BM25+RRF) -> 重排(bge-reranker) -> 降级容错 -> 检索日志 -> FastAPI 暴露 `/knowledge-base/report/search`

### 跳过(基础设施,非学习重点)
- JanusGraph 图谱(`fetch_graph` 保持占位)
- RocketMQ / EOS(S3)/ Quartz / MyBatis-Plus / 多数据源
- 多语料(先只做"用户上传"知识库,舆情/专题后期)

### 语料(用复现2 自产文档)
- `11_report_generation/outputs/*.md`(已生成的铁矿石报告)
- `docs/*.html` 转文本(学习笔记)
- `agent复刻思考与认知.md`、各阶段 outputs 的 summary
- 可选:补 2~3 篇铁矿石行业公开文档

---

## 四、接入架构(复刻版)

```
项目复现2/agent_service (LangGraph)              ← 已有,改 .env
   │  MCP streamable_http
   ▼
项目复现2/mcp_server (FastMCP,:17000)            ← 已有,加 fetch_knowledge 工具
   │  fetch_knowledge 工具内部 httpx POST
   ▼
项目复现2/knowledge_service (FastAPI,:8092)      ← 新建,Python 复刻 knowledge_be
   │  /knowledge-base/report/search
   ▼
Chroma(本地文件)+ bge-m3/reranker(本地 sentence-transformers)
```

> 端口用 8092(致敬原版 8091 但不冲突)。

---

## 五、目录结构(新建 knowledge_service/)

```
项目复现2/knowledge_service/
├── main.py                     # FastAPI 入口(:8092)
├── pyproject.toml              # fastapi/uvicorn/chroma/sentence-transformers/rank_bm25/httpx
├── .env.example
├── src/
│   ├── api/
│   │   ├── routes.py           # /knowledge-base/report/search、/index、/health
│   │   └── schemas.py          # Pydantic 模型(对齐 KnowledgeSearchRequestVO/VO)
│   ├── pipeline/
│   │   ├── loader.py           # Load:读 data/corpus/*
│   │   ├── splitter.py         # Split:RecursiveCharacterTextSplitter
│   │   ├── embedder.py         # Embed:bge-m3 本地(包成函数,可换 API)
│   │   ├── reranker.py         # Rerank:bge-reranker 本地
│   │   └── store.py            # Store:Chroma 读写
│   ├── retrieval/
│   │   ├── bm25.py             # BM25 召回(rank_bm25)
│   │   ├── vector.py           # 向量召回(Chroma KNN)
│   │   ├── hybrid.py           # RRF 融合
│   │   └── search.py           # 主链路(对齐 KnowledgeSearchServiceImpl)
│   ├── config.py
│   └── logger.py               # 检索日志(三段耗时)
├── data/
│   ├── corpus/                 # 语料原文
│   └── chroma/                 # Chroma 持久化
└── README.md
```

---

## 六、分阶段实施

### 阶段 0 · 语料准备(0.5 天)
- 收集复现2 文档 -> `knowledge_service/data/corpus/`(.md/.txt)
- 可选补 2~3 篇铁矿石行业文档
- **完成标志**:corpus 目录有 10+ 篇文档

### 阶段 1 · 服务骨架(0.5 天)
- 建 `knowledge_service/`,`pyproject.toml` 装依赖(fastapi/uvicorn/chroma/sentence-transformers/rank_bm25/httpx)
- `main.py` 起 FastAPI(:8092)
- `/knowledge-base/report/search` 先返回假数据,`/health` 可用
- **完成标志**:`curl http://127.0.0.1:8092/knowledge-base/report/search -d '{"content":"测试"}'` 返回假数据

### 阶段 2 · 入库管道 Write 链路(1 天)
- `loader.py`:读 corpus 文档
- `splitter.py`:RecursiveCharacterTextSplitter(chunk_size/overlap 参考原版 `knowledge.splitter` 配置)
- `embedder.py`:sentence-transformers 加载 bge-m3,`embed(text)->vector`
- `store.py`:Chroma 持久化到 `data/chroma/`,存 `{id, content, embedding, metadata(fileName, publishTime, source, contentTags)}`
- `POST /knowledge-base/index`:触发本地语料入库
- **完成标志**:入库后 Chroma 有数据,能做一次纯向量搜索返回相关片段

### 阶段 3 · 检索链路 Read 链路(2 天,核心)
- `bm25.py`:rank_bm25 建索引,检索 topK
- `vector.py`:Chroma KNN 检索
- `hybrid.py`:RRF 融合两路
- `reranker.py`:bge-reranker 逐对打分
- `search.py` 主链路(对齐 `KnowledgeSearchServiceImpl`):
  - `embed(query)` -> 三模式(KNN / BM25 / Mix,按 `reRank`/`topK`)
  - **降级**:embedding 失败 -> 纯 BM25(照搬原版)
  - `rerank`(reRank=1 时)
  - **日志**:`embeddingTimeCost`/`searchTime`/`reRankTimeCost`/`totalHits`
  - 返回 `List[{id, content, fileId, fileName, contentTags, publishTime, sourceType, source, indexTime, score}]`
- **完成标志**:`curl /report/search` 返回真实检索结果,日志打印三段耗时;reRank=1 比 reRank=0 结果更准

### 阶段 4 · 接入 mcp_server(0.5 天)
- `mcp_server/server.py` 加 `fetch_knowledge(query, topK=5, reRank=1)` 工具
- httpx POST `knowledge_service/knowledge-base/report/search`
- 拼文本:`【知识库检索 N 条】` + 每条 `content` + `[来源:fileName, 发布:publishTime]`
- `fetch_news`/`fetch_graph` 暂保持占位
- **完成标志**:单独调 `fetch_knowledge` 工具返回真实检索片段

### 阶段 5 · 接入 agent + 联调(0.5 天)
- `agent_service/.env.dev` 的 `MCP_DEFAULT_SETTINGS`/`MCP_PRICE_SETTINGS` 等:researcher 和 analyst 的 `tools` 加 `fetch_knowledge`
- 重启 agent_service
- `11_report_generation/scripts/generate_report.py`,提问"结合知识库里的铁矿石分析文档和预测数据,生成本周分析报告"
- 观察:researcher 调 `fetch_knowledge`(真实片段)+ `fetch_data`(预测数据),reporter 生成含知识库引用的报告
- **完成标志**:报告里出现来自知识库的内容(非纯预测数据)

### 阶段 6 · 评估 + 文档(0.5 天)
- 简单评估:5~10 个 query + 期望命中文档,算命中率@5;或接 RAGAS
- `knowledge_service/README.md`:启动、入库、检索、架构
- 复刻笔记:和原版 knowledge_be 的对照(简化了什么、为什么)

---

## 七、与原版 knowledge_be 的对照学习点

| 复刻阶段 | 读原版哪个文件 | 体会什么 |
|---|---|---|
| 阶段2 切片 | `util/FileContentSplitter.java` + `docs/文本拆分器逻辑final.md` | 段落/表格/长度多策略 |
| 阶段3 检索主链路 | `base/service/impl/KnowledgeSearchServiceImpl.java`(845 行,**精读**) | 三模式召回 + 降级 + 日志 |
| 阶段3 重排 | `api/EmbeddingRerankApi.java` | rerank 调用方式 |
| 阶段3 配置 | `application-dev.yml` 的 `splitter`/`embedding`/`rerank` | 参数怎么调 |
| 阶段2 存储 | `resources/es/mappings/kbIronOreCore.json` | 字段设计(复刻版 Chroma metadata 对齐) |

---

## 八、风险与降级

- **bge-m3 下载大/慢**:可先用 `bge-small-zh` 跑通流程,再换 bge-m3
- **CPU 推理慢**:缓存 query 向量;或接受秒级延迟(学习够用)
- **Chroma 过滤弱**:发布日期过滤在 Python 侧后处理
- **若 knowledge_service 太重**:降级为把 RAG 逻辑直接塞进 `mcp_server` 的 `fetch_knowledge`(少一层服务),但偏离原版三层架构--届时先和你确认

---

## 九、验收标准

1. `knowledge_service` 独立跑通:`/report/search` 返回真实检索结果,含混合召回 + 重排 + 降级 + 三段耗时日志
2. `mcp_server` 的 `fetch_knowledge` 工具返回真实检索片段
3. agent 联调:生成的报告含知识库引用内容
4. 文档齐全:`README.md` + 复刻笔记

---

## 十、运行顺序(完成后)

4 个终端:
1. `knowledge_service`:`uv run uvicorn main:app --port 8092`
2. `mcp_server`:`uv run python server.py`(:17000)
3. `agent_service`:`uv run uvicorn main:app --port 5000`
4. `generate_report.py`:生成报告

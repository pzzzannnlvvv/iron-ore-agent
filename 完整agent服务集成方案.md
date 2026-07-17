# 复刻方案：完整 agent 服务集成进复现2

## 目标

在 `项目复现2/` 下完整复刻 `xmschain_agent`（planner→researcher→reporter 自主研究架构），自建 MCP server 让 researcher 能检索复现2 的建模产出，最终基于预测数据生成铁矿石分析报告。LLM 用你自己的。

## 目录结构（复现2 下新增，不动现有 01~10）

```
项目复现2/
├── 01~10, multi_target... (现有建模流水线，不动)
├── agent_service/        # 完整 agent 服务（从 xmschain_agent 复刻）
│   ├── main.py, pyproject.toml, .env, src/ ...
├── mcp_server/           # 自建 MCP server，桥接复现2 数据
│   ├── server.py (streamable_http), tools/ (fetch_data 等读 outputs)
└── 11_report_generation/ # 报告产出 + 客户端脚本
    ├── scripts/generate_report.py (调 agent 服务 + 解析 SSE)
    └── outputs/
```

## 分阶段实施

### 阶段0 · 代码迁移
在 `agent_service/` 下从 `xmschain_agent` 拷贝 `src/`、`main.py`、`pyproject.toml`、`.env.example`（排除 `.venv/__pycache__/.git`）；装 Python 3.12 + uv，`uv sync`（已配清华镜像）。

### 阶段1 · LLM 配置（你的）
改 `agent_service/.env` 填 `LLM_VENDOR/BASE_URL/API_KEY/MODEL`；写最小脚本验证 LLM 调通。OpenAI 兼容接口走默认 `CustomChatOpenAI` 分支即可。

### 阶段2 · 免 Postgres
设 `LANGGRAPH_CHECKPOINT_SAVER=False`；处理 `deepresearch.py:468` 的 `graph.checkpointer = checkpointer_pool._checkpointer`——未配 Postgres 时保留图自带的 MemorySaver，不覆盖。

### 阶段3 · 自建 MCP server（核心难点）
用 `mcp` 库写 streamable_http server（端口 17000），实现工具读复现2 产出——`fetch_data` 读 07/10 预测结果与 MDA、`fetch_knowledge` 读特征体系/标的元数据、`fetch_news/fetch_background` 占位、`model_choice` 读评估；改 `.env` 的 `MCP_*_SETTINGS` 把 url 指向 `127.0.0.1:17000/mcp`；验证 agent 能经 MCP 拿到复现2 数据。

### 阶段4 · prompts 适配
`default/reporter.md`（铁矿石分析师人设）原版先跑通，再按"基于库存预测模型结果分析"微调 reporter/planner/researcher。

### 阶段5 · 联调
起 mcp_server(17000) + agent_service(5000) → `POST /agent/api/deepresearch/stream`，`agent_type=REPORT`，提问"基于本周铁矿石库存预测生成分析报告" → 观察 planner 拆步、researcher 查数据、reporter 出报告。

### 阶段6 · 衔接脚本
`11_report_generation/scripts/generate_report.py` 封装请求+SSE 解析+存报告；文档说明完整启动顺序。

## 风险与降级

- **阶段3 是最大工作量**。若自建 MCP server 太重，可降级为：把"读复现2数据"做成 langchain 本地 tool 直接注入 researcher（绕过 MCP），但会偏离"完整 MCP 架构"——届时我会先问你。
- 完整服务依赖多（langgraph/langchain/mcp/fastapi），`uv sync` 可能遇兼容问题，有镜像兜底。
- prompts 需迭代，首轮报告质量可能一般。

## 诚实提示

这是最重的方案：要新建 2 个子项目、自建 MCP server、改配置、调 prompts、联调。对复现2 这个数据建模项目而言，完整 agent 服务（常驻 FastAPI + 多节点编排）形态上偏重，核心价值在于让你体验完整的 planner→researcher→reporter 自主研究闭环。我会每阶段做完确认再继续。

---

## 现状核实与下一步：知识库端到端联调（2026-07-15）

> 背景：原方案阶段 0-6 已完成 agent↔mcp_server 联调（2026-07-07，4 份报告）。本节针对后续接入的 knowledge_service，核实真实进度并给出端到端联调方案。核实方法：直接查代码/数据/git，不依赖 memory 快照。

### 一、进度核实结论

| 断言 | 核实结果 | 证据 |
|---|---|---|
| knowledge_service 全流程已通 | ✅ 属实 | Chroma 入库 237 chunk（`knowledge_service/data/chroma/chroma.sqlite3`，3.1MB，7/14 仍有写入）；检索链路代码完整 |
| hit@5 = 88% | ✅ 数字真实，评测简陋 | `scripts/eval.py`：8 条手写 query，top5 文件名沾边即算命中，非严谨 RAGAS，仅看趋势 |
| mcp_server 的 fetch_knowledge 接好 | ✅ 属实 | `mcp_server/server.py:121` 实现；`agent_service/.env.dev:14-15` DEFAULT/SYNTHESIS 的 researcher+analyst 已配权限 |
| 仅差端到端联调 | ⚠️ 方向对，需精确化 | 见下 |

**精确化**：
- agent↔mcp_server 半条链 7/7-7/9 已跑通（4 份报告为证），但**无一引用知识库**（grep `知识库|fetch_knowledge` 全 0 命中）；knowledge_service 7/13 才提交，时序上不可能调过。
- 真正没跑通的是 **agent↔mcp_server↔knowledge_service 完整链**；`knowledge_service/README.md:14` 自标 ⬜ 待办。

**两个隐藏缺口（不补则联调调不到知识库）**：
1. `agent_service/src/prompts/deepresearch/default/researcher.md` 工具选择表（29-35 行）**无 `fetch_knowledge`**，第 53 行反限制其使用 -> LLM 不知何时该调。
2. `11_report_generation/scripts/generate_report.py:96` 默认提问"数据时间范围/训练测试划分/gap"**不引导知识库** -> researcher 只会调 `fetch_background`/`fetch_data`。

**LLM 配置确认**：`.env.dev` 仍为 `LLM_VENDOR=anthropic` + 火山方舟 GLM（`glm-5-2-260617`），key 已配，未改动（排除"配置漂移"风险）。

### 二、下一步执行方案（6 步）

**第 1 步 · 前置自检**
- `curl http://127.0.0.1:8092/health` -> `chunks=237`
- 三服务入口齐全（均已有 `__pycache__` 运行痕迹）
- `.env.dev` 的 LLM key 有效

**第 2 步 · 补缺口 1：researcher.md 加 fetch_knowledge 引导**
- 工具选择表加一行：`fetch_knowledge` | 使用场景=查过往分析报告/学习笔记/项目文档结论 | 禁止用途=验证 fetch_data 已得数据
- 调整第 53 行表述，明确"查文档类信息时应优先 fetch_knowledge"

**第 3 步 · 补缺口 2：generate_report.py 改默认提问**
- `default_q` 改为引导知识库的提问，如"结合知识库里的铁矿石分析文档和预测数据，梳理本周库存走势与分析"
- 或保留通用默认、另加知识库专用提问分支

**第 4 步 · 启动三服务 + 联调**
- 终端 1：`knowledge_service` `:8092`（`uvicorn main:app --port 8092`）
- 终端 2：`mcp_server` `:17000`（`python server.py`）
- 终端 3：`agent_service` `:5000`（`uvicorn main:app --port 5000`）
- 终端 4：`generate_report.py`（用第 3 步提问）
- 建议设 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUTF8=1`

**第 5 步 · 验收**（对照 `knowledge_be接入方案.md` 验收标准）
- `generate_report.py` 打印的 `[工具调用]` 日志出现 `fetch_knowledge`
- 报告含来自知识库的片段（非纯预测数据）
- `knowledge_service` 日志打印三段耗时（embeddingTimeCost / searchTime / reRankTimeCost）

**第 6 步 · 更新文档**
- `knowledge_service/README.md:14` ⬜ -> ✅
- `agent复刻进度.md` 追加 knowledge 联调结果
- 本文件记录联调结果

### 三、已知风险
- thinking 块 / token 截断：已有 `_content_to_text` helper + `LLM_MAX_TOKENS=16000`，若仍截断再调大
- `fetch_knowledge` 超时：`mcp_server` 设 `timeout=120`，rerank ~10s，应够
- researcher 工具预算：max 2 轮 / 每轮 3-4 个，提问要让其一轮内同时调 `fetch_knowledge` + `fetch_data`
- `eval.py` 仅 8 条 query，联调后可考虑扩充评测（非阻塞）

---

## 联调结果（2026-07-17，已跑通）

按上述 6 步执行，三服务起齐后跑 `generate_report.py`，**完整链打通、fetch_knowledge 触发、报告引用知识库**。

**改动落地（Part A）**：
- `researcher.md`：工具表加 `fetch_knowledge` 行；第53行改为「查文档类信息优先 fetch_knowledge，但不重复验证 fetch_data 数值」。
- `planner.md`：第7行项目上下文补 `fetch_knowledge`；第73行「只有口径说明才规划」拓宽为「涉及过往报告/笔记/文档结论/口径说明时应规划 fetch_knowledge 步骤」。
- `generate_report.py`：`default_q` 改为引导知识库的提问。
- `analyst.md`：补一句 fetch_knowledge 可用于综合分析补文档（轻量）。
- `mcp_server/server.py`：fetch_knowledge 工具描述已足够，未改。

**联调证据（report_20260717_091245.md，28 次工具调用）**：
- ✅ fetch_knowledge 触发：parallel_researcher 多次调用（口径定义、关键驱动因素等不同角度）。
- ✅ 报告引用知识库：报告「口径说明」明写「与知识库中3份过往分析报告保持一致」，含 KB 来源的港口明细（曹妃甸港、江阴港）、分地区（华北/华南/长江沿江/华东）、业务因果链（废钢价差↑->废钢替代铁矿↑->铁矿需求↓->库存↑）--fetch_data/fetch_background 不返回这些。
- ✅ knowledge_service 三段耗时：首查 embeddingTimeCost=28.96s（bge-m3 首次加载）、searchTime=0.68s、reRankTimeCost=14.78s；后续 embedding 降至 0.1s。BM25 索引由 Chroma 237 chunk 现建。
- ✅ 链路：agent:5000 -> mcp:17000 -> knowledge:8092，多轮 POST /knowledge-base/report/search 200 OK。

**踩坑**：mcp_server `uv run python server.py` 撞清华镜像 403（httpx-0.28.1 轮子）；改用 `.venv/Scripts/python.exe server.py` 直跑解决。

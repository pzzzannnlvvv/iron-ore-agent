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

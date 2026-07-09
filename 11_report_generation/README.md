# 11_report_generation · 报告生成

调用 `agent_service` 生成铁矿石库存分析报告，保存到 `outputs/`。

## 完整启动顺序（3 个终端）

### 终端 1 · 启动 MCP server
```bash
cd 项目复现2/mcp_server
uv run python server.py
# 等待出现：Uvicorn running on http://127.0.0.1:17000
```

### 终端 2 · 启动 agent_service
```bash
cd 项目复现2/agent_service
uv run uvicorn main:app --host 0.0.0.0 --port 5000
# 等待出现：Application started successfully
```

### 终端 3 · 生成报告
```bash
cd 项目复现2/11_report_generation
# 自定义问题
uv --directory ../agent_service run python scripts/generate_report.py "基于本周铁矿石库存预测生成分析报告"
# 或不传参数，用默认问题
uv --directory ../agent_service run python scripts/generate_report.py
```

报告保存到 `outputs/report_<时间戳>.md`。

## 前提条件
- `agent_service/.env.dev` 已填你的 LLM 配置（`LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL`）—— 见阶段1
- 复现2 的 `07_model_training/outputs`、`05_feature_engineering/outputs` 等产出文件存在（MCP server 会读取它们）

## 故障排查
- **连接被拒（agent_service 没起）**：先起终端 2
- **工具调用失败（mcp_server 没起）**：先起终端 1
- **LLM 报错 401/连接超时**：检查 `.env.dev` 的 LLM 配置是否正确
- **报告为空**：看终端 2 的 agent_service 日志，确认 planner/researcher/reporter 是否正常执行

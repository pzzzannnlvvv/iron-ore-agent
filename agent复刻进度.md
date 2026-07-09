# Agent 复刻任务进度（项目复现2）

> ✅ 全部阶段完成。本文件现在作为「如何运行 + 改动记录」参考。
> 原方案见同目录 `完整agent服务集成方案.md`。

## 目标
把 `xmschain_agent`（铁矿石研究报告生成服务，LangGraph）完整复刻进项目复现2，自建 MCP server 桥接复现2 的建模产出，用**火山方舟 GLM（Anthropic 协议）**作 LLM，生成铁矿石库存预测分析报告。

## 各阶段状态

| 阶段 | 状态 | 说明 |
|---|---|---|
| 0 搬代码+装环境 | ✅ | `agent_service/` 建好；uv 0.11.26（`C:\Users\admin\.local\bin\uv.exe`）；Python 3.12.13；依赖装好 |
| 1 配置 LLM | ✅ | 火山方舟 GLM（Anthropic 协议）；`test_llm.py` 验证 LLM_OK |
| 2 免 Postgres | ✅ | `SAVER=False` + `deepresearch.py:468` 保留 MemorySaver |
| 3 自建 MCP server | ✅ | `mcp_server/` 7 工具；agent 端能加载；fetch_data 返回真实预测数据 |
| 4 prompts 适配 | ✅ | researcher 工具表 + planner/reporter 顶部注入"库存预测"项目上下文 |
| 5 启动联调 | ✅ | 跑通完整流程 coordinator→planner→researcher→reporter，生成完整报告 `report_20260707_085712.md`（18 次工具调用）|
| 6 客户端脚本 | ✅ | `11_report_generation/scripts/generate_report.py` + README |

## 阶段5联调结果
- 完整流程跑通：coordinator(handoff_to_planner) → planner(生成4步计划) → parallel_researcher(并行调 MCP 工具 fetch_data/model_predict/model_choice) → reporter(生成报告)
- 报告含真实数据：标的 ID00186052、470训练/56测试行、LightGBM+Optuna 200次、完整超参表、MDA 99.56%→84.91% 对比、2026-02-13~05-01 的 12 周预测明细、3200 特征、过拟合分析
- 联调中发现并修复两个问题：
  1. **planner 崩溃**：ChatAnthropic 返回 content 为 thinking+text 列表，`full_response += chunk.content` 报 `TypeError: can only concatenate str (not "list") to str`。加 `_content_to_text` helper 修复（见改动第 11 条）
  2. **报告截断**：thinking 吃 max_tokens，4096 不够 → `.env.dev` 的 `LLM_MAX_TOKENS` 调到 16000（改动第 12 条）

## 如何再次生成报告
**前提**：`.env.dev` 的 `LLM_API_KEY` 是用户的火山方舟 key（ark 开头，用户手填，Claude 不写）。

开 3 个终端（命令用完整 uv 路径）：

**终端1 · MCP server**（常驻）
```
/c/Users/admin/.local/bin/uv.exe --directory "C:/Users/admin/Desktop/项目复现2/mcp_server" run python server.py
```
等 `Uvicorn running on http://127.0.0.1:17000`

**终端2 · agent_service**（常驻）
```
/c/Users/admin/.local/bin/uv.exe --directory "C:/Users/admin/Desktop/项目复现2/agent_service" run uvicorn main:app --host 0.0.0.0 --port 5000
```
等 `Application started successfully`

**终端3 · 生成报告**
```
/c/Users/admin/.local/bin/uv.exe --directory "C:/Users/admin/Desktop/项目复现2/agent_service" run python "C:/Users/admin/Desktop/项目复现2/11_report_generation/scripts/generate_report.py"
```
报告存到 `11_report_generation/outputs/report_<时间戳>.md`。改问题：编辑脚本末尾 `default_q`，或传命令行参数。

> 改了 `src/` 下代码或 `.env.dev` 要重启 agent_service（uvicorn 没开 --reload）。

## 已做的关键改动（文件级）

### agent_service/（从 xmschain_agent 拷贝，已改）
1. **pyproject.toml**：加 `langchain-anthropic>=0.3.1` 依赖（已 uv sync 装 1.4.8）
2. **.env.dev**：
   - `LLM_VENDOR=anthropic`、`LLM_BASE_URL=https://ark.cn-beijing.volces.com/api/coding`、`LLM_MODEL=glm-5-2-260617`、`LLM_API_KEY=`（火山方舟 key，完整值在 .env.dev 文件里，用户已填）
   - `LANGGRAPH_CHECKPOINT_SAVER=False`（免 Postgres）
   - 所有 `MCP_*_SETTINGS` 的 url 改成 `http://127.0.0.1:17000/mcp`（指向自建 MCP server）
3. **src/llms/llm.py**：加 `anthropic` vendor 分支，用 `ChatAnthropic`（anthropic_api_key/anthropic_api_url/model/temperature/max_tokens）
4. **src/server/deepresearch.py:468**：`if checkpointer_pool._checkpointer is not None: graph.checkpointer = ...`（未配 Postgres 时保留 MemorySaver）
5. **src/server/utils.py `_create_event_stream_message`**：content 是列表时提取 `text` 块（ChatAnthropic 返回 thinking+text 列表）
6. **src/graph/deepresearch/nodes.py `reporter_node`**：response.content 是列表时提取 text 作为 final_report
7. **src/prompts/deepresearch/default/researcher.md**：工具表改成复现2 语义（fetch_data=预测+评估+特征，fetch_background=项目背景，model_choice=评估详情，fetch_news=占位无新闻）
8. **src/prompts/deepresearch/default/planner.md**：顶部注入"库存预测项目"上下文
9. **src/prompts/deepresearch/default/reporter.md**：顶部注入"库存预测项目"上下文
10. **test_llm.py**：LLM 调通测试脚本（`uv run python test_llm.py`，已验证 LLM_OK）
11. **src/graph/deepresearch/nodes.py**（联调修复）：加模块级 `_content_to_text(content)` helper（content 是 str 直接返回；是列表则提取 `type==text` 块、跳过 thinking）；`planner_node` 流式累加改 `full_response += _content_to_text(chunk.content)`。**新增节点若 `+= content` 必须套这个 helper**
12. **.env.dev `LLM_MAX_TOKENS`**（联调修复）：4096 → 16000（thinking 吃 token，4096 会截断 reporter 报告）

### mcp_server/（自建）
- **server.py**：FastMCP，`host=127.0.0.1 port=17000`，7 工具（fetch_data/fetch_background/fetch_news/fetch_graph/model_choice/model_predict/draw_chart），读复现2 的 07/05 等 outputs
- **pyproject.toml**：mcp + pandas + uvicorn（已 uv sync）

### 11_report_generation/
- **scripts/generate_report.py**：客户端，POST `/agent/api/deepresearch/stream`，解析 SSE，提取 agent 含 "reporter" 的 message_chunk 拼成报告，存 `outputs/report_<时间戳>.md`
- **README.md**：3 终端启动顺序
- **outputs/report_20260707_085712.md**：首份完整报告（联调产物）

## 潜在问题
- **thinking 块**：已用 `_content_to_text` 处理（utils.py + nodes.py 的 planner/reporter）。新增节点若 `+= content` 必须套 helper。`LLM_ENABLE_THINKING=False` 当前 anthropic 分支没接，端点仍返回 thinking。
- **token**：thinking 吃 token，`LLM_MAX_TOKENS=16000`。若报告仍截断再调大。
- **researcher ReAct**：ChatAnthropic 工具调用联调验证正常（18 次工具调用成功）。

## 关键路径
- 项目根：`C:/Users/admin/Desktop/项目复现2/`
- uv：`/c/Users/admin/.local/bin/uv.exe`（或 `C:\Users\admin\.local\bin\uv.exe`）
- agent_service venv：`agent_service/.venv`（Python 3.12.13）
- mcp_server venv：`mcp_server/.venv`
- 原 agent 项目（参考）：`C:/Users/admin/Desktop/AI应用开发/xmschain_agent`

## 重要文件清单
- `完整agent服务集成方案.md` — 原方案
- `agent复刻进度.md` — 本文件
- `agent_service/.env.dev` — LLM 配置（含真实 key，用户手填）
- `agent_service/test_llm.py` — LLM 测试脚本
- `agent_service/src/llms/llm.py` — 加了 anthropic 分支
- `agent_service/src/server/utils.py` — content 列表提取 text
- `agent_service/src/server/deepresearch.py` — 免 Postgres 改造
- `agent_service/src/graph/deepresearch/nodes.py` — `_content_to_text` helper + planner/reporter content 提取
- `mcp_server/server.py` — 自建 MCP server
- `11_report_generation/scripts/generate_report.py` — 客户端
- `11_report_generation/README.md` — 启动说明
- `11_report_generation/outputs/report_20260707_085712.md` — 首份完整报告

# iron-ore-agent

铁矿石库存预测**智能体项目** —— 基于 LangGraph 的深度研究(Deep Research)多智能体系统,通过 MCP 协议桥接铁矿石库存预测建模产出,自动生成库存走势与分析报告。

> 本仓库只含**智能体部分**(`agent_service` + `mcp_server` + 报告生成 + 建模产出数据),不含建模脚本本身。建模主流程(数据准备→特征工程→LightGBM 训练)在另一个仓库。

## 架构

```
用户问题
  │
  ▼
coordinator(前台主管)        ── 接需求、判定是否简单查询
  │
  ▼
planner(规划师)              ── 拆解成 Plan/Step 计划单(JSON)
  │
  ▼
human_feedback(人审)         ── 自动接受(auto_accepted_plan)
  │
  ▼
research_team(调度台)        ── 空壳节点 + 路由函数,按依赖派活
  │  ┌──────────────┬──────────────┐
  ▼  ▼              ▼              ▼
researcher    parallel_researcher  analyst   ── 各自装工具箱(merge_mcp_tools)、ReAct 调 MCP
  │              │              │
  └──────────────┴──────────────┘
                 │ (每步回调度台)
                 ▼
            reporter(报告员)      ── 综合研究过程,写最终报告
```

<img width="653" height="598" alt="image" src="https://github.com/user-attachments/assets/055d775f-c42d-4889-8b1c-4295cd0e5268" />


详细架构与类比讲解见 [`agent复刻架构图.md`](agent复刻架构图.md)、[`agent复刻思考与认知.md`](agent复刻思考与认知.md)、[`agent复刻学习指南.html`](agent复刻学习指南.html)。

## 目录结构

```
iron-ore-agent/
├── agent_service/              # 智能体服务(LangGraph + FastAPI, 端口 5000)
│   ├── src/
│   │   ├── graph/deepresearch/ #   状态图: builder / nodes / planner_model
│   │   ├── agents/             #   agent 创建、merge_mcp_tools
│   │   ├── prompts/            #   各节点 prompt 模板
│   │   ├── config/             #   Configuration(读 env)
│   │   └── server/             #   FastAPI 路由
│   ├── main.py
│   └── pyproject.toml
├── mcp_server/                 # MCP 工具服务(端口 17000, 路径 /mcp)
│   └── server.py               #   7 个 @mcp.tool
├── 11_report_generation/       # 报告生成客户端
│   └── scripts/generate_report.py
├── 0X_*/outputs/               # 建模产出数据(mcp_server 读取)
├── multi_target/outputs/       # 多目标建模产出
├── agent复刻*.md/html          # 学习文档
└── .gitignore
```

## MCP 工具(mcp_server 提供 7 个)

| 工具 | 取什么 | 数据来源 |
|---|---|---|
| `fetch_background` | 项目背景(标的/时间范围/建模流程) | 硬编码文本 |
| `fetch_data` | 预测结果+评估指标+超参+特征列表 | `07/05` outputs |
| `model_choice` | 模型评估详情+过拟合判断 | `07` outputs |
| `model_predict` | 测试集预测明细 | `07` outputs |
| `fetch_news` | 占位(本项目无新闻数据) | — |
| `fetch_graph` | 占位(本项目无图谱数据) | — |
| `draw_chart` | 占位(暂不支持绘图) | — |

## 环境准备

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) 包管理器
- 一个可用的 LLM 服务(兼容 OpenAI API 格式)

## 配置 `.env.dev`

`agent_service/.env.dev` **不包含在仓库中**(含真实密钥,已 gitignore),需自行创建。关键变量:

```dotenv
# ===== LLM =====
LLM_VENDOR=your-vendor
LLM_BASE_URL=http://your-llm-host/api/v3
LLM_MODEL=your-model-id
LLM_API_KEY=your-api-key-here
LLM_MAX_TOKENS=4096
LLM_ENABLE_THINKING=False

# ===== MCP 工具权限表(指向 mcp_server) =====
MCP_DEFAULT_SETTINGS='{"servers":{"xmschain-mcp-tools":{"enabled_tools":[{"node":"background","tools":["fetch_background"]},{"node":"analyst","tools":["fetch_news","fetch_graph"]},{"node":"researcher","tools":["fetch_data","fetch_news","model_choice","model_predict","draw_chart"]}],"transport":"streamable_http","url":"http://127.0.0.1:17000/mcp"}}}'

# ===== Agent 递归限制 =====
AGENT_RECURSION_LIMIT=60
AGENT_DEEPRESEARCH_RECURSION_LIMIT=500
```

> `MCP_DEFAULT_SETTINGS` 是"工具权限表":决定每个节点(researcher/analyst/background)能调哪些 MCP 工具。`url` 指向 mcp_server 的地址。

## 启动(三个终端)

```bash
# 1. 启动 mcp_server(端口 17000)
cd mcp_server
uv run python server.py

# 2. 启动 agent_service(端口 5000)
cd agent_service
uv run uvicorn main:app --port 5000

# 3. 生成报告(用 agent_service 的 venv 运行)
cd 11_report_generation
uv --directory ../agent_service run python scripts/generate_report.py "基于铁矿石库存预测模型的结果,生成一份库存走势与分析报告,包括模型表现(MDA、MAPE)和最近的预测值"
```

报告输出到 `11_report_generation/outputs/report_*.md`。完整前提见 [`11_report_generation/README.md`](11_report_generation/README.md)。

## 数据说明

- 仓库已含建模产出 `outputs/` 数据,供 mcp_server 直接读取
- **5 个超大中间产物 CSV 已排除**(>50MB,超 GitHub 限制),如需完整可由建模脚本的 `05_feature_engineering`、`06_data_split` 重新生成:
  - `05_feature_engineering/outputs/{03_factor_lags,04_trend_volatility,08_features_merged,09_features_final}.csv`
  - `06_data_split/outputs/train.csv`
- 主标的(ID00186052 铁矿石港口库存)测试集 MDA = 84.91%

## 可测问题示例

- **模型表现**:`铁矿石库存预测模型的 MDA 是多少?训练集和测试集差距多大?是否过拟合?`
- **预测结果**:`最近 12 周的库存预测值与实际值对比`
- **特征体系**:`模型使用了哪些特征?一共多少个?`
- **背景方法**:`为什么用一阶差分 diff 作为训练目标?`
- **综合报告**:`生成一份模型可靠性评估报告`(默认问题)

> 注意:`max_step_num` 默认为 4,单个问题聚焦 1-2 个数据维度最稳,别一次问太散。需要新闻/图谱/绘图的问题会拿到占位回复(本项目无此类数据)。

## License

私有项目。

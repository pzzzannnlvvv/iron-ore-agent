# agent 复刻 · 调度架构图（与代码核对版）

> 本文档对照 `agent_service/src/graph/deepresearch/` 的真实代码，画出 planner→researcher→reporter 自主研究架构的准确流转。用于纠正手绘理解图中的偏差，**不涉及代码改动**。
> 涉及代码均以 `文件:行号` 标注，可直接跳转比对。

---

## 一、准确架构图

```
                        ┌───────────────────────────────┐
                        │  .env / Configuration         │  启动时读，每次
                        │  (LLM、MCP url、token limit)  │  from_runnable_config 解析
                        └───────────────────────────────┘
                                     │
START ──► coordinator（前台主管）─────┘
          判断 + 多轮追问(max N轮)
          │
          ├─ 闲聊/直接回答 ───────────────────────────► END
          ├─ simple_search快速 ─► tool_executor ─► summary ─► END   旁路①
          ├─ 背景调查 ─► background_investigator ─► planner           (调 merge_mcp_tools)
          └─ 规划 ─► planner（规划师）
                     输出 Plan JSON: steps[]{ step_type, need_search,
                                       association.depends_on_steps }
                     │
                     ├─ has_enough_context / 迭代超限 ─► reporter
                     └─► human_feedback（人工确认 [ACCEPTED]/[EDIT_PLAN]）
                              │ ACCEPTED
                              ▼
                       research_team（调度室）◄────────────────────┐
                       ┌──────────────────────────────┐            │ 每执行完一个/一批
                       │  本体 = 空壳 pass            │            │ step 都回到这里
                       │  真正派活的是路由函数:       │            │ 重新算就绪步骤
                       │  continue_to_running_       │            │
                       │  research_team              │            │
                       └──────────────┬──────────────┘            │
                                      │ 扫 steps：依赖已满足 &    │
                                      │   未执行 = 就绪           │
                                      │                           │
            ┌─────────────────────────┼─────────────────────────┐ │
            ▼                         ▼                         ▼ │
      就绪 research×1           就绪 research×N            就绪 analysis │
            │                         │                         │ │
            ▼                         ▼                         ▼ │
       researcher              parallel_researcher           analyst     │
       (串行，取第1个           (asyncio.gather              (纯LLM,     │
        未执行step)              +Semaphore(4)并发            tools=[])   │
            │                    共享一份MCP工具池)              │        │
            │  ┌─ merge_mcp_tools 在这里调（创建agent前） ──┐  │ │        │
            │  │  wrap_tools_with_dynamic_params 注task_id │  │ │        │
            │  └──────────────────────────────────────────┘  │ │        │
            │                         │                         │ │
            └─── goto="research_team" ─┴─── goto="research_team"┘ │
                                      │                           │
                                      │ 全部step完成              │
                                      ▼                           │
                                 reporter（出报告）──────────► END

工具室 mcp_server/server.py (@mcp.tool: fetch_data / model_choice / ...)
   ▲ 被谁调：background_investigator / researcher / parallel_researcher
   │         / analyst 无工具 / tool_executor   ← 经 merge_mcp_tools 拉取
   └─ 端口 17000，streamable-http
```

---

## 二、逐框对照手绘理解图

| 手绘图里的框 | 手绘写的职责 | 真实代码 | 偏差 |
|---|---|---|---|
| reporter | 出报告 | `reporter_node` `nodes.py:1108`，读 `plan.title/thought` + observations，`goto=END` | ✅ 对 |
| 工具室 mcp_server | `@mcp.tool` | `mcp_server/server.py`，7 个工具，端口 17000 | ✅ 对，但被**多个节点**调用，不只调度室 |
| 调度室 research_team | 研究组长 + `merge_mcp_tool` | `research_team_node` 是 `pass` 空壳（`nodes.py:1169`）；派活靠路由函数 `continue_to_running_research_team`（`builder.py:22`）；**merge 不在这里** | ❌ ①本体是空壳 ②merge 在研究员侧 |
| 研究员工具箱 | （连接研究员） | `wrap_tools_with_dynamic_params` 注入 task_id 等动态参（`tool_injection.py`） | ✅ 对 |
| 研究员 parallel_research | "看计划单决定派是 research，后续都不经过调度室" | **看计划单派活**是调度室路由函数的活；研究员只执行被派的 step；**执行后 `goto="research_team"` 每次都回调度室**（`nodes.py:2269`） | ❌ ①职责画错位 ②"不回调度室"与代码相反 |
| 规划师 planner | 拆解 + 输出 JSON 计划单 | `planner_node` `nodes.py:318`，输出 Plan；**但还有 `human_feedback` 人工确认环节**，且 `has_enough_context`/超限可直跳 reporter | ⚠️ 对，但漏了 human_feedback 和直跳 reporter 两条出口 |
| 前台主管 coordinator | 判断 + 提问 | `coordinator_node` `nodes.py:648`，含多轮追问、补全 topic、背景调查路由、闲聊 END、simple_search 跳过 | ⚠️ 对，但出口比手绘多 |
| 配置表 env | 只第一次调度时配 | `.env` 启动读；MCP client 在**每个研究员节点**创建 agent 时建一次（parallel_researcher 复用一份） | ⚠️ "只第一次"不准，是每节点建、并行时复用 |

---

## 三、最容易误解的点：调度是"反馈式轮询"，不是"一次派出"

手绘里"后续都不经过调度室"是整张图最大的偏差。真实流程是循环：

1. **算就绪**：调度室路由函数扫计划单，找出**就绪步骤** = `execution_res is None`（没执行过）且 `depends_on_steps` 全在已完成集合里（`builder.py:44-59`）。
2. **派活**：就绪 research 步骤 = 1 个 → 派 `researcher`；>1 个 → 派 `parallel_researcher` 并发；有就绪 analysis → 派 `analyst`；全完成 → `reporter`（`builder.py:68-78`）。
3. **回写**：被派的节点**只执行就绪的那一批**，把结果回写到 `step.execution_res`，然后 `goto="research_team"` 回调度室。
4. **重算**：调度室**重新**扫计划单（刚完成的 step 进了已完成集合，可能解锁新的就绪 step），再派下一批。如此循环直到全部完成 → reporter。

> 依赖关系（`depends_on_steps`）是靠"每轮回调度室重新算就绪"来兑现的。若真"不回调度室"，依赖编排、并行批次的推进就都没了着落。这是这套图最核心的机制。

---

## 四、关键文件 / 行号索引（比对代码用）

| 关注点 | 位置 |
|---|---|
| 图构建（节点 + 边 + 条件路由） | `agent_service/src/graph/deepresearch/builder.py` |
| 调度室路由函数（算就绪、决定下一跳） | `builder.py:22-78` `continue_to_running_research_team` |
| 调度室节点本体（空壳） | `nodes.py:1169-1173` `research_team_node` |
| 计划单数据结构（Step / Plan / 依赖） | `planner_model.py` |
| coordinator（前台主管） | `nodes.py:648` `coordinator_node` |
| planner（规划师，输出 Plan JSON） | `nodes.py:318` `planner_node` |
| human_feedback（人工确认计划） | `nodes.py:561` `human_feedback_node` |
| reporter（出报告） | `nodes.py:1108` `reporter_node` |
| researcher（串行，取第 1 个未执行 step） | `nodes.py:1476` `researcher_node` → `_execute_agent_step` `nodes.py:1176` |
| parallel_researcher（并发，共享 MCP 工具池） | `nodes.py:2142` `parallel_researcher_node` |
| analyst（analysis 步骤，纯 LLM 无工具） | `nodes.py:1498` `analyst_node` |
| background_investigator（背景调查） | `nodes.py:277` `background_investigation_node` |
| tool_executor + summary（simple_search 旁路） | `nodes.py:1531` / `nodes.py:1761` |
| simple_researcher（直接 React 旁路） | `nodes.py:1830` `simple_researcher_node` |
| merge_mcp_tools（按 agent_name 过滤启用工具） | `agent_service/src/agents/mcps.py:7` |
| 工具动态参数注入（task_id 等） | `agent_service/src/agents/tool_injection.py` `wrap_tools_with_dynamic_params` |
| MCP 工具室（@mcp.tool） | `mcp_server/server.py`（端口 17000） |
| State 字段定义 | `agent_service/src/graph/deepresearch/types.py` |
| agent↔LLM 映射、token 上限 | `agent_service/src/config/agents.py` |

---

## 五、手绘图里漏画的节点（补全）

- **analyst**：处理 `step_type=analysis` 的步骤，纯 LLM 推理，`tools=[]`，不挂 MCP 工具。
- **human_feedback**：planner 出计划后的人工确认环节，`[ACCEPTED]` 放行 / `[EDIT_PLAN]` 回炉。
- **background_investigator**：可选的背景调查节点，先于 planner 跑，调 `merge_mcp_tools("background", ...)`。
- **tool_executor → summary**：`simple_search` 快速模式旁路，绕过 planner，ReAct 跑工具后用 reporter 的 LLM 总结。
- **simple_researcher**：直接 React 模式旁路（向后兼容），不依赖 Plan/Step 结构，直接跑完到 END。

---

## 六、类比版架构（研究公司模型）

把整套系统想成一家"研究公司"，组件一一对照：

| 类比 | 真实组件 | 说明 |
|---|---|---|
| 档案室 | `mcp_server/server.py`（端口 17000） | 独立开门的资料室，摆着几份资料 |
| 资料 = `@mcp.tool` 函数 | `fetch_data` / `model_choice` 等 7 个 | `@mcp.tool` 把函数登记上墙，外部才看得到 |
| 制度手册 = env.dev | `.env.dev` 的 `MCP_DEFAULT_SETTINGS` | 写明档案室地址 + 每个岗位能借哪些资料 |
| 前台 | `schemas.py` | 受理任务时从手册抄一张权限表附在任务单上 |
| 调度台 = research_team | `research_team_node`（空壳 `pass`）+ 路由函数 | 只看任务单派活，不碰手册、不碰工具 |
| 研究员 = researcher / parallel_researcher | `researcher_node` 等 | 到岗后自己拿权限表去档案室领工具 |
| 工具箱 | `merge_mcp_tools` + `create_agent` 绑的工具 | 研究员到岗现装，每个 step / 批重装一次 |
| 任务单 / 白板 = state | `State` 对象 | 计划单、step 结果都写在上面，谁需要谁读 |

### 一次研究的类比流程

1. 客户提问题 -> 前台建档，交前台主管（coordinator）判断立项。
2. 规划师（planner）把任务拆成几个 step，写成计划单贴上白板。
3. （可选）客户确认计划。
4. 调度台看白板上的计划单，按就绪情况派活：单个就绪 research step 派 1 个研究员；多个就绪派并行研究员；analysis 派分析师。
5. 研究员到岗：从任务单拿出前台附的权限表 -> 按地址连档案室 -> 看墙上菜单（`get_tools`）-> 只拿自己有权借的几份 -> 配工牌（`task_id`）-> 挎上工具箱上岗。
6. 研究员大脑（LLM）决定借哪份、借几次，每次直接打档案室取数据（不回调度台）。
7. 一个 step 做完，回调度台复命；调度台重算就绪，派下一批。
8. 全部 step 完成 -> 回规划师复盘（默认放行）-> 撰稿人（reporter）出报告 -> 交付。

### 这套类比专门纠正的几个误解

- ❌"调度室准备工具箱" -> ✅ 调度台只派活，工具箱是研究员到岗自己装的。
- ❌"工具箱只装一次" -> ✅ 每个 step（串行）/ 每批（并行）重装一次，跨 step 不复用。
- ❌"`merge_mcp_tools` 在调度室" -> ✅ 在研究员节点内（`_setup_and_execute_agent_step`）。
- ❌"env.dev 是调度室读的" -> ✅ 是前台（`schemas`）受理任务时读，研究员到岗才用。
- ❌"做完 step 直接 reporter" -> ✅ 先回 planner 复盘，默认放行后才 reporter。
- **装 vs 调**：装工具箱（连档案室 + 拉菜单 + 绑工具）到岗做一次；调工具（打档案室取数据）一个 step 内可多次，不回调度台。
- **`@mcp.tool` 三件事**：登记上墙（注册成对外可见的工具）、写说明书（函数名 + docstring + 参数给 LLM 看）、代办借阅（收到调用执行函数返回结果）。没它档案室是空房子。


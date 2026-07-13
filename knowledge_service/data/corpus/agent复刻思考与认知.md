# agent 复刻的思考与认知

> 搭完这套 agent 之后的一些反思。核心想通了一件事:整个 agent 项目其实是一个**约束工程**,我们搭建 agent 的意义,就是**把一次问答的流程工具化,让它变得可调用,让非技术人员也能享受到问答的服务**。

---

## 一、想通的核心:agent 是「约束工程」,不是「更强的 AI」

很多人(包括之前的我)以为搭 agent 是为了"让 AI 更聪明、更能干"。搭完才发现恰恰相反——agent 的本质是**把 AI 约束进一条可控的流水线**。

对比一下两种思路:

- **直接用 Claude Code / 通用 AI**:把"流程"也交给 AI。AI 自己决定读哪个文件、grep 什么、跑什么命令。人只下指令。AI 既是工位,又是流程设计师。
- **自建 agent**:把"流程"攥在人手里。人预先画好流水线(coordinator→planner→researcher→reporter),AI 只在决策点填智能(选哪个工具、怎么拆步、怎么写报告)。AI 只是流水线上的智能工位。

> 一句话:**Claude Code 是放权给 AI,自建 agent 是约束 AI。** 前者无所不能但不可预测,后者只擅长预见到的任务但稳定可控。

所以搭 agent 的核心动作不是"调教模型",而是"设计约束"——把一件复杂的事,拆成固定的人控流程 + AI 填空的决策点。

---

## 二、「约束」体现在哪四个层面

这套 agent 项目里,"约束"是层层叠叠的:

| 约束层 | 怎么约束 | 代价 / 换来什么 |
|---|---|---|
| **流程约束** | LangGraph 图固定编排节点和边,有条件边控制跳转 | 流程不能即兴改 → 换来可复现、可并行、可恢复 |
| **工具约束** | MCP 只暴露几个窄工具(fetch_data 等),AI 不知文件路径 | 不能随便翻文件 → 换来数据接入可控、安全 |
| **知识约束** | 领域知识固化在 prompt 和工具 docstring 里 | 不靠临时上下文 → 换来输出稳定、专业 |
| **输出约束** | Plan 模型校验、结构化计划、reporter 人设 | 不让 AI 自由发挥格式 → 换来可标准化、可下游处理 |

每一层约束,都是用"灵活性"换"可控性"。这就是工程——**工程就是把不确定性约束掉,让结果可预期**。

---

## 三、搭建 agent 的意义:问答流程的工具化

这是我最大的收获。之前觉得"问 AI"就是打开对话框敲一句话。搭完 agent 才明白:**一次问答可以被打包成一个可调用的服务**。

### 1. 工具化:从「一次性对话」变成「可复用的能力」

直接用 Claude Code 出一份报告,是"一次性对话"——下次要再出一份,得重新问、重新探索、重新等。

自建 agent 把这个过程**固化成一个 HTTP 接口**(`/agent/api/deepresearch/stream`):任何前端、定时任务、其他系统,POST 一个问题进来,就流水线般产出报告。一次搭建,反复调用。

### 2. 可调用:常驻服务,随时能调

agent_service 是个常驻 FastAPI 服务(端口 5000)。这意味着:
- 网页前端可以调它 → 做成网页问答产品
- 定时任务可以调它 → 每周自动出铁矿石库存报告
- 其他程序可以调它 → 把"出报告"变成系统的一个零件

它不再依赖某个本地 CLI、某个人在场,而是**随时在线的服务**。

### 3. 平民化:让非技术人员也能享受问答服务

这是最关键的意义。直接用 Claude Code 出报告,需要:
- 会装环境、会用命令行
- 会看代码、能判断 AI 扫的文件对不对
- 会写 prompt 引导 AI 找对数据
- 本地有 Claude Code、有额度

**这些门槛把"AI 问答"关在了开发者的小圈子里。**

而自建 agent 把这些复杂性全吞进了服务内部:业务人员只需要打开网页,问一句"基于本周库存预测生成分析报告"——背后那套 coordinator 拆步、researcher 查 MCP、reporter 写报告的复杂流程,对他完全透明。

> 我们搭 agent,本质是**把"会用 AI"这件事的专业门槛,从用户侧搬到了开发者侧**。开发者吃下复杂度,换用户一个简单的输入框。

---

## 四、约束 vs 自由的权衡

不是所有场景都该自建 agent。这是个权衡:

**约束换来**:稳定、可复现、可并行、可产品化、领域固化、输出标准化、给非技术用户用、可控不跑偏。

**约束的代价**:灵活性(只擅长预见到的任务)、搭建成本(要画图、写工具、调 prompt)、维护成本(流程一变要改代码)。

所以:

| 场景 | 该用什么 |
|---|---|
| 个人偶尔看一次报告 / 探索陌生代码 / 调 bug / 跨领域未知问题 | **直接用 Claude Code**(零搭建,灵活) |
| 给别人用 / 重复定时跑 / 嵌入产品网页 / 多用户 / 输出要稳定标准 | **自建 agent**(约束化,可产品化) |

一句话:**Claude Code 是通用瑞士军刀,自建 agent 是专用机器。** 前者灵活万能,后者稳定可复制。不是替代关系,是不同场景的工具。

---

## 五、从「用 AI」到「造 AI 产品」的跃迁

这次复刻最大的价值,不是"多了一个能出铁矿石报告的工具"(那个 Claude Code 也能干),而是完成了一次认知跃迁:

**从"用 AI"跨到了"造 AI 产品"。**

- 以前:AI 是个工具,我用它。
- 现在:AI 是个零件,我把它组织成产品给别人用。

而且在这个过程中,摸到了"造任何 agent"的通用积木:
- **LangGraph** — 怎么用节点+边+state 编排流程
- **MCP** — 怎么把工具/数据解耦,让 AI 可调用
- **state** — 怎么让节点间通过共享白板流转数据
- **SSE** — 怎么把生成过程流式推给用户
- **条件边** — 怎么让流程根据状态动态路由、并行

这些能力不绑定铁矿石,不绑定库存预测——换个领域,同样的积木能搭出另一个 agent。这才是真正可迁移的收获。

---

## 六、一次请求的完整旅程:从类比到实际链路

前面几节讲的都是「为什么」,这一节补上「实际怎么跑」——把一次请求从进来到出报告的完整路径串一遍,先类比,再对照真实代码。第二节那四层约束到底是哪句话、哪个函数在执行,看完这条链路就落地了。

### 1. 类比:一家研究咨询公司接客户委托

```
客户提问题
  │
  ▼
①前台登记建档(thread_id)→ 交给前台主管 coordinator
  │   主管判断:要不要立项?走哪个业务线(agent_type)?
  ▼
②规划师 planner 写研究计划
  │   拆成几个可并发的 research 步骤,输出一份 JSON 计划单
  ▼
③(可选)给客户确认计划 human_feedback
  │
  ▼
④研究员到岗后,自己调 merge_mcp_tools 装工具箱(不在调度室):
  │   查配置表 env.dev → 连楼下资料室 mcp server
  │   → 从服务菜单 @mcp.tool 里挑授权工具(如 fetch_data)
  │   → 工具上挂着说明牌 docstring,一起塞进研究员工具箱
  ▼
⑤研究组长 research_team 看计划单:
  │   哪些步骤前置依赖已满足?能并行的派给 parallel_researcher,单个的派给 researcher
  ▼
⑥研究员(大脑=LLM)干活:
  │   读工具箱里的 docstring,判断"这步该调 fetch_data"
  │   → 调用经 MCP 协议发到 mcp server → 执行 fetch_data → 拿回预测/评估数据
  ▼
⑦回 research_team 复命,看还有没有就绪步骤 → 有则回 ⑤循环
  │
  ▼
⑧全部完成 → 撰稿人 reporter 出报告 → 交付客户(SSE 流结束)
```

### 2. 实际链路(带文件位置)

```
HTTP POST /agent/api/... (server/deepresearch.py)
  │  请求进来,分配 thread_id
  ▼
StreamingResponse → graph.astream (LangGraph)
  │
  ▼
[START → coordinator]  coordinator_node (nodes.py)
  │  判断是否研究、locale、agent_type;可能插入 background_investigation
  ▼
background_investigator → planner   (builder.py:103)
  │
  ▼
planner_node (nodes.py:318)
  │  apply_prompt_template 渲染 deepresearch/{agent_type}/planner.md (prompts/template.py:47)
  │  llm.astream(...) 流式输出 → full_response 拼成 JSON 文本
  │  json.loads(repair_json_output(...)) → dict
  │  Plan.model_validate(dict) → Plan 对象 (planner_model.py)
  ▼
human_feedback_node(可选中断,等用户确认/修改计划)
  ▼
research_team_node + continue_to_running_research_team (builder.py:22-108)
  │  扫描 Plan.steps:依赖已满足且未执行的 → ready
  │  多个 research 步骤 → parallel_researcher;单个 → researcher;analysis → analyst
  │  全完成 → reporter
  ▼
researcher_node / parallel_researcher_node (nodes.py:1476 / 1625)
  │  上岗前先调 merge_mcp_tools(agent_name, mcp_settings, default_tools) (agents/mcps.py:7)
  │    └─ MultiServerMCPClient 连各 mcp server → get_tools()
  │       按 env.dev/请求里的 enabled_tools 过滤 → loaded_tools
  │       工具 description 来自 mcp_server 里函数的 docstring
  │  create_agent(LLM, tools=loaded_tools) → LLM 拿到工具清单(含 docstring)
  │  LLM 决定调 fetch_data → langchain_mcp_adapters 经 MCP 协议发请求
  ▼
mcp_server/server.py (端口 17000, FastMCP)
  │  @mcp.tool() 注册的 fetch_data(query) 执行
  │  读 07_model_training/outputs、05_feature_engineering/outputs 下的文件
  │  返回:标的 + 模型评估 + 最优超参 + 测试集预测 + 特征体系
  ▼
结果回到 researcher → 写入 step.execution_res → 回 research_team 循环 (builder.py:110)
  │
  ▼ 全部 step.execution_res 非空
reporter_node → 渲染 reporter.md → 最终报告
  ▼
graph END → SSE event 流回前端 (server/utils.py:_make_event)
```

### 3. 这条链路里,第二节的四层约束正在动

- **流程约束**:coordinator→planner→research_team→researcher→reporter 的边是 LangGraph 在 `builder.py` 里画死的,请求只能这么走,不能即兴改道。
- **工具约束**:researcher 拿不到文件路径,只能调 `merge_mcp_tools` 装箱进来的 `fetch_data` 等窄工具,数据接入被锁死在 mcp server 里。
- **知识约束**:planner 该怎么拆步、`fetch_data` 该怎么用,都固化在 `planner.md` 和 docstring 里,不靠临时上下文。
- **输出约束**:planner 的 JSON 必须能过 `Plan.model_validate`,格式跑偏就被拦下来。

> 把这条链路和第二节对照着看就明白:所谓「约束工程」不是抽象口号,而是这条路径上每一处节点、边、工具、校验,都在替 AI 把不确定性约束掉。

### 4. 调度机制复盘:几个被类比对齐过的点

第1小节的类比初稿里,曾把"备工具"写成调度室的活,深抠代码后发现几处偏差,在此纠偏(也是最容易混的点):

- **调度室是空壳,派活的是路由函数**:`research_team_node` 本体就一句 `pass`(nodes.py:1169),真正看计划单决定派谁的是挂在它出边上的路由函数 `continue_to_running_research_team`(builder.py:22)。节点只负责"中转",决策在边上。
- **工具箱是研究员到岗自己装的,不在调度室**:`merge_mcp_tools` 在 `researcher_node -> _setup_and_execute_agent_step`(nodes.py:1439)里调,不在 research_team。调度台只"喊人",研究员到岗后自己凭权限表去档案室领工具。
- **工具箱不是只装一次**:串行每个 step 重装一次;并行每批装一次共享给本批并发子任务(nodes.py:2176)。跨 step / 批不复用。
- **env.dev 是前台读的,不是调度室**:`.env.dev` 的 `MCP_DEFAULT_SETTINGS` 在请求进来时由 `schemas.py:67` 解析成 `mcp_settings` 附在任务单上,跟着请求流进研究员节点。调度台从头到尾不碰 env.dev。
- **@mcp.tool 三件事**:登记上墙(注册成对外可见的工具)、写说明书(函数名 + docstring + 参数给 LLM 看)、代办借阅(收到调用执行函数返回结果)。没它档案室是空房子。
- **装 vs 调要分清**:装工具箱(连档案室 + 拉菜单 + 绑工具)到岗做一次;调工具(打档案室取数据)一个 step 内可多次,直接 researcher↔mcp_server,不回调度台。
- **做完 step 不是直接 reporter**:所有 step 完成后路由函数派的是 planner(builder.py:35),让 planner 复盘"够不够出报告";默认 `max_plan_iterations=1` 到限即放行给 reporter,所以看起来像直接出报告,但中间过了 planner 这一站。

---

## 七、回到项目复现2:这套 agent 给了它什么

项目复现2 本来是个**数据建模项目**:LightGBM 预测铁矿石库存,产出的是 CSV、JSON、评估指标。这些产出对懂技术的人有意义,但对业务人员是"看不懂的数字"。

加这套 agent 之后:
- 建模产出被 MCP 包成了"可自然语言查询"的工具
- 非技术人员问一句话,就能拿到一份**有人话、有结论、有建议**的分析报告
- 预测结果从"躺在 outputs 文件夹里的数据",变成了"随时能问答的服务"

> 这套 agent 给项目复现2 加的不是"预测能力"(那个 LightGBM 已经有了),而是**把预测结果服务化、平民化的能力**。让冰冷的模型产出,变成业务人员用得起的咨询。

---

## 八、最后的一句话

搭 agent 这件事,表面上是在写 LangGraph、写 MCP、调 prompt,骨子里是在做一道选择题:

**你要 AI 的自由,还是 AI 的可控?**

- 选自由 → 用 Claude Code,享受万能但不可预测
- 选可控 → 自建 agent,用约束换来稳定、可产品化、能普惠到非技术用户

两者都该会。而当你开始选"可控"的那一刻,你就从 AI 的**使用者**,变成了 AI 产品的**建造者**。这次复刻,就是跨过那道门。

---

*配套文件:`agent复刻进度.md`(改动与运行命令)、`agent复刻学习指南.html`(7 个问题搞懂原理)、`agent复刻架构图.md`(架构图与类比核对)、`完整agent服务集成方案.md`(原始方案)。*

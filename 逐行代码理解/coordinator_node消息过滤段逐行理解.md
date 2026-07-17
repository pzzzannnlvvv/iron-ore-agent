# coordinator_node 消息过滤段 —— 逐行代码理解

> 代码位置：`agent_service/src/graph/deepresearch/nodes.py` 约 677–707 行
> 所属函数：`coordinator_node`

---

## 一、这段代码的整体目的

把白板（`state["messages"]`）里所有消息**过滤一遍**，挑出 coordinator 该看的，扔掉内部噪音（规划思路、路由指令、占位消息等），得到一个干净的上下文列表 `user_role_messages`，供后面 LLM 使用。

---

## 二、关键前置概念：白板是累积历史

`state["messages"]` **不是这一次提问的内容，而是整个对话历史的累积**。

流程是这样的：

```
你提问 ──> HumanMessage（你的话）写进白板
   │
   └─> coordinator 是个 LLM，它要回应
          └─> AIMessage（coordinator 的话）也写进白板
```

你每问一次，coordinator 就回一次，一来一回两条都留在白板上。下一轮再问时，白板里就同时有用户消息和 AI 消息了。

举例（澄清场景）：

| 轮次 | 谁说的 | 白板新增 | 白板累积 |
|---|---|---|---|
| 1 | 你 | HumanMessage「研究铁矿石价格」 | [你的问题] |
| 1 | coordinator | AIMessage「看哪个时间段？」 | [你的问题, coordinator 的问题] |
| 2 | 你 | HumanMessage「近一年」 | [你的问题, coordinator 的问题, 你的回答] |

白板里的 AI 消息来源不止 coordinator，还有：
- **planner** 的规划输出（name="planner"）
- **deepresearch** 的中间思考（name="deepresearch"）
- **researcher** 的研究结果等

这些都是之前轮次各节点干完活后写进白板的。所以需要过滤。

---

## 三、逐行讲解

### 1. 读三个标记备用

```python
is_simple_search = state.get("is_simple_search", False)
enable_clarification = state.get("enable_clarification", False)
initial_topic = state.get("research_topic", "")
clarified_topic = initial_topic
```

- `is_simple_search`：是不是快速搜索模式
- `enable_clarification`：要不要追问澄清
- `initial_topic`：用户原始问题
- `clarified_topic`：澄清后的话题，先默认等于原始问题

### 2. 准备过滤

```python
user_role_messages = []
for i in state["messages"]:
```

建空列表放过滤结果，然后遍历白板里每条消息 `i`。

### 3. 用户消息分支（HumanMessage）

```python
    if type(i) is HumanMessage:
        if "Original" in i.content and "Topic" in i.content:
            continue
```
是用户消息。先跳过系统塞的 "Original Topic" 占位消息（不是真用户说的话）。

```python
        if 'Here is a summary of the conversation to date' in i.content:
            continue
```
跳过历史摘要消息（也是系统生成的，不是用户原话）。

```python
        if i.name and i.name == 'coordinator':
            continue
        user_role_messages.append(i)
```
跳过 coordinator 自己冒充 user 发的消息。其余真用户消息保留。

### 4. AI 消息分支（AIMessage）

```python
    elif type(i) is AIMessage:
        if i.name and (i.name == 'planner' or i.name == 'deepresearch'):
            continue
```
是 AI 消息。跳过 planner 和 deepresearch 的消息——它们是规划思路/中间思考，coordinator 不用看。

```python
        if i.name and i.name == 'coordinator' and i.tool_calls:
            continue
        user_role_messages.append(i)
```
跳过 coordinator **带工具调用**的消息（那是内部路由指令，比如「转给 planner」）。
但保留 coordinator **不带工具调用**的消息（比如闲聊回复）。

### 5. dict 格式消息分支

```python
    elif type(i) is dict and 'role' in i and (i.get("role", "") == "user" or i.get("role", "") == "system"):
        if 'name' in i and i.get("role", "") == "coordinator":
            continue
        user_role_messages.append(i)
```
有些消息是 dict 格式（不是 HumanMessage/AIMessage 对象）。只处理 user/system 角色的，跳过 coordinator 的，其余保留。

---

## 四、一句话总结

这段就是个**筛子**：从整段对话的累积流水账里，挑出 coordinator 这次该看的（真用户发言 + 该看的 AI 回复），扔掉不该看的（占位、摘要、规划思路、路由指令），给后面 LLM 一个干净的上下文。

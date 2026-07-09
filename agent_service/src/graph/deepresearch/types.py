from langgraph.graph import MessagesState

from .planner_model import Plan


class State(MessagesState):
    """State for the agent system, extends MessagesState with next field."""

    locale: str
    research_topic: str
    # Complete/final clarified topic with all clarification rounds
    clarified_research_topic: str
    observations: list[str]
    plan_iterations: int
    current_plan: Plan | str
    final_report: str
    auto_accepted_plan: bool
    enable_background_investigation: bool
    background_investigation_results: str

    # Clarification state tracking (disabled by default)
    # Enable/disable clarification feature (default: False)
    enable_clarification: bool
    clarification_rounds: int
    clarification_history: list[str]
    is_clarification_complete: bool
    # Default: 3 rounds (only used when enable_clarification=True)
    max_clarification_rounds: int

    # Workflow control
    # Default next node
    goto: str

    # Simple search mode flags
    is_simple_search: (
        bool  # 标识是否是 simple_search 模式（不管 simple_search_prompt 有没有值）
    )
    simple_search_with_prompt: bool  # 标识是否是 simple_search_prompt 有值的情况
    is_simple_deepresearch: bool  # 标识是否是旧的 simple_deepresearch 模式（走 planner -> researcher -> reporter）

    # 多轮对话历史上下文（不与 messages 字段合并，由 simple_researcher_node 单独读取）
    conversation_history: list[dict]

    # Coordinator 标记的下一个节点类型，供 SSE 层 / 前端区分路由目标
    # 可选值: "tool_executor" | "planner" | ""
    next_agent: str

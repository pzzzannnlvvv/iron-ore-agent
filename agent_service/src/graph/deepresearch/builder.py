from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .planner_model import Plan, StepType
from .nodes import (
    analyst_node,
    coordinator_node,
    human_feedback_node,
    planner_node,
    reporter_node,
    research_team_node,
    researcher_node,
    parallel_researcher_node,
    background_investigation_node,
    simple_researcher_node,
    tool_executor_node,
    summary_node,
)
from .types import State


def continue_to_running_research_team(state: State):
    """决定 research_team 节点的下一步跳转目标"""
    current_plan = state.get("current_plan")

    # 1. 确保 current_plan 是 Plan 对象
    if not current_plan or not isinstance(current_plan, Plan) or not current_plan.steps:
        return "planner"

    is_simple_search = state.get("is_simple_search", False)

    # 2. 检查是否整体全部完成
    all_completed = all(step.execution_res is not None for step in current_plan.steps)
    if all_completed:
        return "reporter" if is_simple_search else "planner"

    # 核心逻辑：动态计算当前所有步骤的【就绪状态】
    # 提取已完成步骤的 title 集合
    completed_step_titles = {
        step.title for step in current_plan.steps if step.execution_res is not None
    }

    # 扫描当前【所有前置依赖已完成】且【自身未执行】的就绪步骤 (Ready Steps)
    ready_research_steps = []
    ready_analysis_steps = []

    for step in current_plan.steps:
        if step.execution_res is not None:
            continue  # 已经执行过了，跳过

        # 解析模型生成的字符串数组依赖关系
        depends_steps = []
        if hasattr(step, "association") and step.association:
            depends_steps = getattr(step.association, "depends_on_steps", [])

        # 核心判定：当前步骤的所有前置依赖，是否都在已完成集合中
        is_dependency_satisfied = all(
            dep_title in completed_step_titles for dep_title in depends_steps
        )

        if is_dependency_satisfied:
            if step.step_type == StepType.RESEARCH:
                ready_research_steps.append(step)
            elif step.step_type == StepType.ANALYSIS:
                ready_analysis_steps.append(step)

    # 存在可以执行的 research 步骤
    if ready_research_steps:
        # 如果当前互不依赖、 research 步骤超过 1 个，并行
        if len(ready_research_steps) > 1:
            return "parallel_researcher"
        # 只有一个 research 步骤，串行
        return "researcher"

    if ready_analysis_steps:
        return "analyst"

    return "reporter" if is_simple_search else "planner"

def _build_base_graph():
    """Build and return the base state graph with all nodes and edges."""
    builder = StateGraph(State)
    builder.add_edge(START, "coordinator")
    builder.add_node("coordinator", coordinator_node)
    builder.add_node("background_investigator", background_investigation_node)
    builder.add_node("planner", planner_node)
    builder.add_node("reporter", reporter_node)
    builder.add_node("research_team", research_team_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("parallel_researcher", parallel_researcher_node)     # 负责多步并发处理 Research
    builder.add_node("analyst", analyst_node)
    builder.add_node("human_feedback", human_feedback_node)
    # ...existing code...
    # Add simple_researcher node for direct React mode (keeping for backward compatibility)
    builder.add_node("simple_researcher", simple_researcher_node)
    builder.add_edge("simple_researcher", END)
    # Add tool_executor and summary nodes (split mode)
    builder.add_node("tool_executor", tool_executor_node)
    builder.add_node("summary", summary_node)
    builder.add_edge("tool_executor", "summary")
    builder.add_edge("summary", END)
    # ...existing code...
    builder.add_edge("background_investigator", "planner")
    builder.add_conditional_edges(
        "research_team",
        continue_to_running_research_team,
        ["planner", "researcher", "parallel_researcher", "analyst", "reporter"],
    )

    builder.add_edge("parallel_researcher", "research_team")
    builder.add_edge("reporter", END)
    return builder


def build_graph_with_memory():
    """Build and return the agent workflow graph with memory."""
    # use persistent memory to save conversation history
    # TODO: be compatible with SQLite / PostgreSQL
    memory = MemorySaver()

    # build state graph
    builder = _build_base_graph()
    return builder.compile(checkpointer=memory)


def build_graph():
    """Build and return the agent workflow graph without memory."""
    # build state graph
    builder = _build_base_graph()
    return builder.compile()


graph = build_graph()

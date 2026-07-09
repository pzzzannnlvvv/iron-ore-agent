from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph

from .nodes import predict_reasoning_node, predict_report_node
from .types import State


def _build_base_graph():
    """Build and return the base state graph with all nodes and edges."""
    builder = StateGraph(State)
    builder.add_edge(START, "predict_reasoning")
    builder.add_node("predict_reasoning", predict_reasoning_node)
    builder.add_node("predict_report", predict_report_node)
    return builder


def build_graph_with_memory():
    """Build and return the agent workflow graph with memory."""
    # use persistent memory to save conversation history
    memory = MemorySaver()

    # build state graph
    builder = _build_base_graph()
    return builder.compile(checkpointer=memory)


def build_graph():
    """Build and return the agent workflow graph without memory."""
    # build state graph
    builder = _build_base_graph()
    return builder.compile()

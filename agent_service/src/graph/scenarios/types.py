from langgraph.graph import MessagesState


class State(MessagesState):
    """State for the agent system, extends MessagesState with next field."""

    # Workflow control
    # Default next node
    goto: str

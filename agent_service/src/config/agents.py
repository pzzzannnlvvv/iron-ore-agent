from typing import Literal

from src.config.loader import get_int_env

# Define available LLM types
LLMType = Literal["streaming", "block"]
AgentName = Literal[
    "coordinator",
    "planner",
    "researcher",
    "analyst",
    "reporter",
    "predict_reasoning",
    "predict_report",
    "suggested",
    "simple_researcher",
    "tool_executor",
    "summary",
]

# Define agent-LLM mapping
AGENT_LLM_MAP: dict[AgentName, LLMType] = {
    "coordinator": "streaming",
    "planner": "streaming",
    "researcher": "block",
    "analyst": "block",
    "reporter": "streaming",
    "predict_reasoning": "block",
    "predict_report": "streaming",
    "suggested": "block",
    "simple_researcher": "block",
    "tool_executor": "streaming",
    "summary": "streaming",
}

AGENT_TOKEN_LIMIT: dict[AgentName, int] = {
    "coordinator": get_int_env("TOKEN_LIMIT_COORDINATOR", 8000),
    "planner": get_int_env("TOKEN_LIMIT_PLANNER", 5000),
    "researcher": get_int_env("TOKEN_LIMIT_RESEARCHER"),
    "analyst": get_int_env("TOKEN_LIMIT_ANALYST"),
    "reporter": get_int_env("TOKEN_LIMIT_REPORTER"),
    "predict_reasoning": get_int_env("TOKEN_LIMIT_PREDICT_REASONING", 5000),
    "predict_report": get_int_env("TOKEN_LIMIT_PREDICT_REPORT", 5000),
    "suggested": get_int_env("TOKEN_LIMIT_SUGGESTED"),
    "simple_researcher": get_int_env("TOKEN_LIMIT_SIMPLE_RESEARCHER"),
    "tool_executor": get_int_env("TOKEN_LIMIT_SIMPLE_RESEARCHER"),
    "summary": get_int_env("TOKEN_LIMIT_SUMMARY"),
}

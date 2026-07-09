from typing import Any, Dict, List, Generator
from contextlib import contextmanager

from langchain_core.outputs import LLMResult
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
    BaseMessage,
    SystemMessage,
)
from langchain_core.callbacks import BaseCallbackHandler
from langchain_community.callbacks.manager import openai_callback_var


def tool_to_fact_message(tool_msg: ToolMessage) -> AIMessage:
    return AIMessage(content=(f"（已获得的工具结果）:\n{tool_msg.content}"))


def extract_replayable_messages(
    messages: list[BaseMessage],
    *,
    keep_system: bool = True,
) -> list[BaseMessage]:
    """
    将 ReAct / LangGraph state 中的 messages
    转换为可再次 llm.invoke 的对话消息
    """

    replay_messages: list[BaseMessage] = []

    for msg in messages:
        # 1. SystemMessage
        if isinstance(msg, SystemMessage):
            if keep_system:
                replay_messages.append(msg)
        # 2. HumanMessage：原样保留
        elif isinstance(msg, HumanMessage):
            replay_messages.append(msg)
        # 3. AIMessage
        elif isinstance(msg, AIMessage):
            # 跳过 tool_call 型 AIMessage
            if msg.tool_calls:
                continue

            # 只保留自然语言 content
            if msg.content:
                replay_messages.append(AIMessage(content=msg.content))
        # 4. ToolMessage：转写
        elif isinstance(msg, ToolMessage):
            replay_messages.append(tool_to_fact_message(msg))

    return replay_messages


class ToolCallbackHandler(BaseCallbackHandler):
    """Callback Handler that tracks OpenAI info."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[BaseMessage] = []

    @property
    def always_verbose(self) -> bool:
        """Whether to call verbose callbacks even if verbose is False."""
        return True

    def on_chain_start(self, *args, **kwargs):
        if args[1].get("__type") == "tool_call_with_context":
            self.messages = args[1].get("state", {}).get("messages", [])

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Print out the prompts."""
        pass

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Print out the token."""
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Collect token usage."""
        # Check for usage_metadata (langchain-core >= 0.2.2)
        pass

    def __copy__(self) -> "ToolCallbackHandler":
        """Return a copy of the callback handler."""
        return self

    def __deepcopy__(self, memo: Any) -> "ToolCallbackHandler":
        """Return a deep copy of the callback handler."""
        return self


@contextmanager
def get_tool_callback() -> Generator[ToolCallbackHandler, None, None]:
    cb = ToolCallbackHandler()
    openai_callback_var.set(cb)
    yield cb
    openai_callback_var.set(None)

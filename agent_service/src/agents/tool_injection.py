import uuid
import contextvars
from contextvars import ContextVar

from loguru import logger

from langchain_core.tools import StructuredTool
from langgraph.config import get_stream_writer


# Context variable to carry LLM's tool call ID from CaptureToolCallIdMiddleware
# to the tool wrapper. Key: tool_name -> LLM's tool_call_id (e.g., "call_xxx")
_tool_call_id_context: ContextVar[dict[str, str]] = ContextVar(
    "tool_call_id_context", default={}
)


def wrap_tools_with_dynamic_params(
    tool: StructuredTool, dynamic_params: dict = {}
) -> StructuredTool:
    async def wrapped(**kwargs):
        # 这里注入的参数，一定要跟MCP侧对齐
        for k, v in dynamic_params.items():
            kwargs[k] = v
        kwargs["task_id"] = str(uuid.uuid4())
        logger.debug(f"injection params: {kwargs}")

        # 获取LLM生成的原始tool_call ID（由CaptureToolCallIdMiddleware设置）
        current_ids = _tool_call_id_context.get()
        llm_tool_call_id = current_ids.get(tool.name)

        writer = get_stream_writer()
        if writer:
            writer(
                {
                    "__tools_injection__": True,
                    "name": tool.name,
                    "args": kwargs,
                    "type": "tool_call_injection",
                    "llm_tool_call_id": llm_tool_call_id,
                }
            )

        return await tool.ainvoke(kwargs)

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        coroutine=wrapped,
        return_direct=tool.return_direct,
    )


def wrap_tools_without_events(tool: StructuredTool) -> StructuredTool:
    """Wrap a tool without sending injection events.

    Useful for background investigation where we want to execute tools directly
    without triggering streaming events that require agent context.
    """

    async def wrapped(**kwargs):
        # Directly invoke the original tool without any event injection
        return await tool.ainvoke(kwargs)

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        coroutine=wrapped,
        return_direct=tool.return_direct,
    )

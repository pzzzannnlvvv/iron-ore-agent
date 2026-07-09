import asyncio
import inspect
from typing import List, Optional, Callable, Any

from loguru import logger
from langchain.agents import create_agent as create_react_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
)
from langgraph.runtime import Runtime

from src.config.agents import AGENT_LLM_MAP
from src.config.loader import get_int_env
from src.llms.llm import get_llm_by_type
from src.prompts import apply_prompt_template, render_system_prompt


class DynamicPromptMiddleware(AgentMiddleware):
    """Middleware to apply dynamic prompt template before model invocation.

    This middleware prepends a system message with the rendered prompt template
    to the messages list before the model is called.
    """

    def __init__(self, prompt_template: str, extra_variables: Optional[dict] = None):
        self.prompt_template = prompt_template
        self.extra_variables = extra_variables

    def before_model(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        """Apply prompt template and prepend system message to messages."""
        try:
            # Get the rendered messages including system prompt from template
            rendered_messages = apply_prompt_template(
                self.prompt_template, state, self.extra_variables
            )
            # The first message is the system prompt, extract it
            if rendered_messages and len(rendered_messages) > 0:
                system_message = rendered_messages[0]
                # Prepend system message to existing messages
                return {"messages": [system_message]}
            return None
        except Exception as e:
            logger.error(
                f"Failed to apply prompt template in before_model: {e}", exc_info=True
            )
            return None

    async def abefore_model(
        self, state: Any, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Async version of before_model."""
        return self.before_model(state, runtime)


class CaptureToolCallIdMiddleware(AgentMiddleware):
    """Middleware to capture LLM's tool call IDs after model invocation.

    在模型调用后从LLM的响应中提取tool_call ID，
    存入contextvar供wrap_tools_with_dynamic_params读取。
    确保tool_calls_injection事件中的id与tool_call_result中的tool_call_id一致。
    """

    def after_model(self, state, runtime):
        messages = state.get("messages", [])
        if not messages:
            return None
        last_msg = messages[-1]
        ids = {}

        # 尝试从完整tool_calls中提取ID（block模式）
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            for tc in last_msg.tool_calls:
                ids[tc["name"]] = tc["id"]

        # 尝试从tool_call_chunks中提取ID（streaming模式）
        if not ids and hasattr(last_msg, "tool_call_chunks") and last_msg.tool_call_chunks:
            for chunk in last_msg.tool_call_chunks:
                try:
                    chunk_dict = (
                        chunk if isinstance(chunk, dict) else chunk.model_dump()
                    )
                except (AttributeError, TypeError):
                    chunk_dict = {}
                name = chunk_dict.get("name", "") or ""
                _id = chunk_dict.get("id", "") or ""
                if name and _id:
                    ids[name] = _id

        if ids:
            from src.agents.tool_injection import _tool_call_id_context

            _tool_call_id_context.set(ids)
            logger.debug(f"Captured tool call IDs from model output: {ids}")

        return None

    async def aafter_model(self, state, runtime):
        return self.after_model(state, runtime)


class PreModelHookMiddleware(AgentMiddleware):
    """Middleware to execute a pre-model hook before model invocation.

    This middleware wraps the legacy pre_model_hook callable and executes it
    as part of the middleware chain.
    """

    def __init__(self, pre_model_hook: Callable):
        self._pre_model_hook = pre_model_hook

    def before_model(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        """Execute the pre-model hook."""
        if not self._pre_model_hook:
            return None

        try:
            result = self._pre_model_hook(state, runtime)
            return result
        except Exception as e:
            logger.error(
                f"Pre-model hook execution failed in before_model: {e}", exc_info=True
            )
            return None

    async def abefore_model(
        self, state: Any, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Async version of before_model."""
        if not self._pre_model_hook:
            return None

        try:
            # Check if the hook is async
            if inspect.iscoroutinefunction(self._pre_model_hook):
                result = await self._pre_model_hook(state, runtime)
            else:
                # Run synchronous hook in thread pool to avoid blocking event loop
                result = await asyncio.to_thread(self._pre_model_hook, state, runtime)
            return result
        except Exception as e:
            logger.error(
                f"Pre-model hook execution failed in abefore_model: {e}", exc_info=True
            )
            return None


# Create agents using configured LLM types
def create_agent(
    graph_name: str,
    agent_name: str,
    tools: list,
    prompt_template: str,
    pre_model_hook: Optional[Callable] = None,
    interrupt_before_tools: Optional[List[str]] = None,
    extra_variables: Optional[dict] = None,
):
    """Factory function to create agents with consistent configuration.

    Args:
        agent_name: Name of the agent
        agent_type: Type of agent (researcher, etc.)
        tools: List of tools available to the agent
        prompt_template: Name of the prompt template to use
        pre_model_hook: Optional hook to preprocess state before model invocation
        interrupt_before_tools: Optional list of tool names to interrupt before execution
        locale: Language locale for prompt template selection (e.g., en-US, zh-CN)

    Returns:
        A configured agent graph
    """
    logger.debug(
        f"Creating agent '{graph_name}' of type '{agent_name}' "
        f"with {len(tools)} tools and template '{prompt_template}'"
    )

    llm_type = AGENT_LLM_MAP.get(agent_name, "streaming")
    # middleware = [DynamicPromptMiddleware(prompt_template, extra_variables)]
    middleware = [
        SummarizationMiddleware(
            model=get_llm_by_type(llm_type),
            trigger=("tokens", get_int_env("AGENT_REACT_SUMMARY_TIGGER", 20000)),
            keep=("tokens", get_int_env("AGENT_REACT_SUMMARY_KEEP", 10000)),
            trim_tokens_to_summarize=get_int_env("AGENT_REACT_TRIM_TOKENS", 15000),
            summary_prompt=render_system_prompt("common/summary"),
        ),
        ToolCallLimitMiddleware(run_limit=get_int_env("AGENT_REACT_TOOLS_LIMIT", 5)),
    ]

    # Add middleware to capture LLM tool call IDs for injection matching
    middleware.append(CaptureToolCallIdMiddleware())

    # Add pre-model hook middleware if provided
    if pre_model_hook:
        middleware.append(PreModelHookMiddleware(pre_model_hook))

    agent = create_react_agent(
        name=graph_name,
        model=get_llm_by_type(llm_type),
        tools=tools,
        middleware=middleware,
        system_prompt=render_system_prompt(prompt_template, extra_variables),
    )
    logger.info(f"Agent '{graph_name}' created successfully")

    return agent

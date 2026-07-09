import json
import asyncio
import dataclasses
import os
from typing import Annotated, Any, Literal, cast
from langchain_core.tools import BaseTool

from loguru import logger
from pydantic_core import ValidationError
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END
from langgraph.graph.state import CompiledStateGraph
from langgraph.errors import GraphRecursionError
from langgraph.types import Command, interrupt

from src.agents import create_agent, merge_mcp_tools
from src.agents.utils import extract_replayable_messages, get_tool_callback
from src.config.agents import AGENT_LLM_MAP
from src.config.configuration import Configuration
from src.llms.llm import get_llm_by_type, get_llm_token_limit_by_agent
from src.prompts.template import (
    apply_prompt_template,
    load_prompts_tree,
    render_system_prompt,
)
from src.utils.context_manager import ContextManager, validate_message_content
from src.utils.json_utils import repair_json_output, sanitize_tool_response
from src.agents.tool_injection import (
    wrap_tools_with_dynamic_params,
    wrap_tools_without_events,
)
from src.graph.utils import (
    build_clarified_topic_from_history,
    reconstruct_clarification_history,
)
from .planner_model import Plan, StepType
from .types import State
import asyncio
from copy import deepcopy
from langgraph.types import Command


graph_name = "deepresearch"

init_prompts = load_prompts_tree()


def _content_to_text(content: Any) -> str:
    """把 LLM 返回的 content 统一成纯文本。

    ChatAnthropic（火山方舟 GLM）会返回 content 为列表（thinking + text 块），
    而本图多处按字符串处理（如 planner 流式累加、JSON 解析）。这里提取 text 块、
    跳过 thinking 块，保证下游拿到干净字符串。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def get_agent_prompt(agent_name: str, agent_type: str) -> str:
    if agent_name in init_prompts.get(graph_name, {}).get(agent_type, []):
        prompt_name = f"{graph_name}/{agent_type}/{agent_name}"
    else:
        prompt_name = f"{graph_name}/default/{agent_name}"

    return prompt_name


@tool
def handoff_to_planner(
    research_topic: Annotated[str, "The topic of the research task to be handed off."],
    locale: Annotated[str, "The user's detected language locale (e.g., en-US, zh-CN)."],
):
    """Handoff to planner agent to do plan."""
    # This tool is not returning anything: we're just using it
    # as a way for LLM to signal that it needs to hand off to planner agent
    return


@tool
def handoff_to_tool_executor(
    research_topic: Annotated[str, "The research topic for the tool executor to search."],
    locale: Annotated[str, "The user's detected language locale (e.g., en-US, zh-CN)."],
):
    """Handoff to tool executor for quick search. Use this for research questions in fast mode."""
    return


@tool
def handoff_after_clarification(
    locale: Annotated[str, "The user's detected language locale (e.g., en-US, zh-CN)."],
    research_topic: Annotated[
        str, "The clarified research topic based on all clarification rounds."
    ],
):
    """Handoff to planner after clarification rounds are complete. Pass all clarification history to planner for analysis."""
    return


@tool
def direct_response(
    message: Annotated[str, "The response message to send directly to user."],
    locale: Annotated[str, "The user's detected language locale (e.g., en-US, zh-CN)."],
):
    """Respond directly to user for greetings, small talk, or polite rejections. Do NOT use this for research questions - use handoff_to_planner instead."""
    return


def needs_clarification(state: dict) -> bool:
    """
    Check if clarification is needed based on current state.
    Centralized logic for determining when to continue clarification.
    """
    if not state.get("enable_clarification", False):
        return False

    clarification_rounds = state.get("clarification_rounds", 0)
    is_clarification_complete = state.get("is_clarification_complete", False)
    max_clarification_rounds = state.get("max_clarification_rounds", 3)

    # Need clarification if: enabled + has rounds + not complete + not exceeded max
    # Use <= because after asking the Nth question, we still need to wait for the Nth answer
    return (
        clarification_rounds > 0
        and not is_clarification_complete
        and clarification_rounds <= max_clarification_rounds
    )


def preserve_state_meta_fields(state: State) -> dict:
    """
    Extract meta/config fields that should be preserved across state transitions.

    These fields are critical for workflow continuity and should be explicitly
    included in all Command.update dicts to prevent them from reverting to defaults.

    Args:
        state: Current state object

    Returns:
        Dict of meta fields to preserve
    """
    return {
        "locale": state.get("locale", "zh-CN"),
        "research_topic": state.get("research_topic", ""),
        "clarified_research_topic": state.get("clarified_research_topic", ""),
        "clarification_history": state.get("clarification_history", []),
        "enable_clarification": state.get("enable_clarification", False),
        "max_clarification_rounds": state.get("max_clarification_rounds", 3),
        "clarification_rounds": state.get("clarification_rounds", 0),
        "is_simple_search": state.get("is_simple_search", False),
        "simple_search_with_prompt": state.get("simple_search_with_prompt", False),
        "is_simple_deepresearch": state.get("is_simple_deepresearch", False),
        "conversation_history": state.get("conversation_history", []),
        "final_report": state.get("final_report", ""),
    }


def validate_and_fix_plan(plan: dict, enforce_web_search: bool = False) -> dict:
    """
    Validate and fix a plan to ensure it meets requirements.

    Args:
        plan: The plan dict to validate
        enforce_web_search: If True, ensure at least one step has need_search=true

    Returns:
        The validated/fixed plan dict
    """
    if not isinstance(plan, dict):
        return plan

    steps = plan.get("steps", [])

    # ============================================================
    # SECTION 1: Repair missing step_type fields (Issue #650 fix)
    # ============================================================
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            continue

        # Check if step_type is missing or empty
        if "step_type" not in step or not step.get("step_type"):
            # Infer step_type based on need_search value
            # Default to "analysis" for non-search steps (Issue #677: not all processing needs code)
            inferred_type = "research" if step.get("need_search", False) else "analysis"
            step["step_type"] = inferred_type
            logger.info(
                f"Repaired missing step_type for step {idx} ({step.get('title', 'Untitled')}): "
                f"inferred as '{inferred_type}' based on need_search={step.get('need_search', False)}"
            )

    # ============================================================
    # SECTION 2: Enforce web search requirements
    # Skip enforcement if web search is disabled
    # ============================================================
    if enforce_web_search:
        # Check if any step has need_search=true (only check dict steps)
        has_search_step = any(
            step.get("need_search", False) for step in steps if isinstance(step, dict)
        )

        if not has_search_step and steps:
            # Ensure first research step has web search enabled
            for idx, step in enumerate(steps):
                if isinstance(step, dict) and step.get("step_type") == "research":
                    step["need_search"] = True
                    logger.info(f"Enforced web search on research step at index {idx}")
                    break
            else:
                # Fallback: If no research step exists, convert the first step to a research step with web search enabled.
                # This ensures that at least one step will perform a web search as required.
                if isinstance(steps[0], dict):
                    steps[0]["step_type"] = "research"
                    steps[0]["need_search"] = True
                    logger.info(
                        "Converted first step to research with web search enforcement"
                    )
        elif not has_search_step and not steps:
            # Add a default research step if no steps exist
            logger.warning("Plan has no steps. Adding default research step.")
            plan["steps"] = [
                {
                    "need_search": True,
                    "title": "Initial Research",
                    "description": "Gather information about the topic",
                    "step_type": "research",
                }
            ]

    return plan


def build_researcher_tool_background(mcp_settings: dict | None) -> str:
    """Build lightweight tool context for planner without connecting MCP servers."""
    if not mcp_settings or not isinstance(mcp_settings, dict):
        return ""

    researcher_tools: list[str] = []
    for server_config in mcp_settings.get("servers", {}).values():
        for enabled_group in server_config.get("enabled_tools", []):
            if enabled_group.get("node") != "researcher":
                continue
            for tool_name in enabled_group.get("tools", []):
                if tool_name not in researcher_tools:
                    researcher_tools.append(tool_name)

    if not researcher_tools:
        return ""

    lines = [
        "当前 researcher 启用工具（仅用于规划，不要在 planner 阶段调用工具）：",
    ]
    for tool_name in researcher_tools:
        lines.append(f"- {tool_name}")
    return "\n".join(lines)


async def background_investigation_node(state: State, config: RunnableConfig):
    logger.info("background investigation node is running.")
    query = state.get("clarified_research_topic") or state.get("research_topic")
    background_investigation_results = []

    # Parse mcp_settings to find tools bound to "background" node
    configurable = Configuration.from_runnable_config(config)

    enable_mcp_servers, loaded_tools = await merge_mcp_tools(
        "background", configurable.mcp_settings, []
    )

    if enable_mcp_servers and query:
        if loaded_tools:
            # Wrap tools without events for direct execution
            wrapped_tools = [wrap_tools_without_events(t) for t in loaded_tools]

            # Execute each tool directly with the query
            for tool in wrapped_tools:
                try:
                    logger.info(f"Executing background tool: {tool.name}")
                    # Try to invoke tool with query parameter
                    # First check if tool accepts 'query' parameter
                    result = await tool.ainvoke({"query": query, "task_id": ""})
                    background_investigation_results.append(
                        {"tool": tool.name, "result": str(result)}
                    )
                except Exception as tool_error:
                    logger.error(f"Error executing tool {tool.name}: {tool_error}")

    logger.info(
        f"Background investigation completed with {len(background_investigation_results)} results"
    )

    return {
        "background_investigation_results": json.dumps(
            background_investigation_results, ensure_ascii=False
        )
    }


async def planner_node(
    state: State, config: RunnableConfig
) -> Command[Literal["human_feedback", "reporter"]]:
    """Planner node that generate the full plan."""
    logger.info(
        f"Planner generating full plan with locale: {state.get('locale', 'zh-CN')}"
    )
    configurable = Configuration.from_runnable_config(config)
    plan_iterations = state["plan_iterations"] if state.get("plan_iterations", 0) else 0

    # ============================================================
    # 检查是否是simple_search快速模式（无预设prompt）
    # ============================================================
    is_simple_search_mode = state.get("is_simple_deepresearch", False) and not state.get(
        "simple_search_with_prompt", False
    )

    if is_simple_search_mode:
        logger.info(
            "[planner_node] Simple search quick mode: using restricted prompt for 2-step plan"
        )

    # 统一使用get_agent_prompt，prompt内容通过Jinja2根据is_simple_search_mode区分
    prompt_template = get_agent_prompt("planner", configurable.agent_type.value)
    logger.info(
        f"[planner_node] max_step_num before prompt render: {configurable.max_step_num}"
    )
    researcher_tool_background = build_researcher_tool_background(
        configurable.mcp_settings
    )

    # For clarification feature: use the clarified research topic (complete history)
    if state.get("enable_clarification", False) and state.get(
        "clarified_research_topic"
    ):
        # Modify state to use clarified research topic instead of full conversation
        modified_state = state.copy()
        modified_state["messages"] = [
            {"role": "user", "content": state["clarified_research_topic"]}
        ]
        modified_state["research_topic"] = state["clarified_research_topic"]
        extra_vars = {
            **dataclasses.asdict(configurable),
            "is_simple_search_mode": is_simple_search_mode,
            "researcher_tool_background": researcher_tool_background,
        }
        messages = apply_prompt_template(
            prompt_template,
            modified_state,
            extra_vars,
        )

        logger.info(
            f"Clarification mode: Using clarified research topic: {state['clarified_research_topic']}"
        )
    else:
        # Normal mode: use full conversation history
        extra_vars = {
            **dataclasses.asdict(configurable),
            "is_simple_search_mode": is_simple_search_mode,
            "researcher_tool_background": researcher_tool_background,
        }
        messages = apply_prompt_template(
            prompt_template,
            state,
            extra_vars,
        )

    if state.get("enable_background_investigation") and state.get(
        "background_investigation_results"
    ):
        messages += [
            {
                "role": "user",
                "content": (
                    "background investigation results of user query:\n"
                    + state["background_investigation_results"]
                    + "\n"
                ),
            }
        ]

    llm = get_llm_by_type(AGENT_LLM_MAP["planner"])

    # if the plan iterations is greater than the max plan iterations, return the reporter node
    if plan_iterations >= configurable.max_plan_iterations:
        return Command(update=preserve_state_meta_fields(state), goto="reporter")

    user_role_messages = []
    for i in messages:
        if type(i) is HumanMessage:
            if "Original" in i.content and "Topic" in i.content:
                # logger.warning(f"Invalid user message content detected and skipped: {i.content}")
                continue
            if 'Here is a summary of the conversation to date' in i.content:
                # logger.warning(f"Invalid user message content detected and skipped: {i.content}")
                continue
            if i.name and i.name == 'coordinator':
                continue
            user_role_messages.append(i)
        elif type(i) is AIMessage:
            # 允许 coordinator 的工具调用消息通过（包含 tool_calls 且 name='coordinator'）
            if i.name and i.name == 'coordinator':
                # 跳过 coordinator 的非工具调用消息
                continue
            if i.name and i.name == 'deepresearch':
                # 跳过 deepresearch 的消息
                continue

            user_role_messages.append(i)
        elif type(i) is dict and 'role' in i and (i.get("role", "") == "user" or i.get("role", "") == "system"):
            if 'name' in i and i.get("role", "") == "coordinator":
                continue
            user_role_messages.append(i)


    # 检测是否存在2个以上的user消息且都不包含"Original Topic"
    user_message_indices = []
    for i, msg in enumerate(user_role_messages):
        if isinstance(msg, HumanMessage):
            user_message_indices.append(i)

    if len(user_message_indices) >= 2:
        # 为除最后一个外的user消息添加提示
        for idx in user_message_indices[:-1]:
            current_content = user_role_messages[idx].content
            if 'User history questions' not in current_content:
                new_content = f"{current_content}(User history questions, avoid redundant research.)"
                user_role_messages[idx].content = new_content
                logger.info(f"为user消息添加提示: {idx}")
    llm_token_limit = get_llm_token_limit_by_agent("planner")
    compressed_state = ContextManager(llm_token_limit, 3).compress_messages(
        {"messages": user_role_messages}
    )
    full_response = ""
    async for chunk in llm.astream(compressed_state["messages"]):
        full_response += _content_to_text(chunk.content)
    logger.debug(f"Current state messages: {state['messages']}")
    logger.info(f"Planner response: {full_response}")

    # Validate explicitly that response content is valid JSON before proceeding to parse it
    if (
        not full_response.strip().startswith("{")
        and not full_response.strip().startswith("[")
        and not full_response.strip().startswith("```json")
    ):
        logger.warning("Planner response does not appear to be valid JSON")
        if plan_iterations > 0:
            return Command(update=preserve_state_meta_fields(state), goto="reporter")
        else:
            return Command(update=preserve_state_meta_fields(state), goto=END)

    try:
        curr_plan = json.loads(repair_json_output(full_response))
        # Need to extract the plan from the full_response
        curr_plan_content = extract_plan_content(curr_plan)
        # load the current_plan
        curr_plan = json.loads(repair_json_output(curr_plan_content))
    except json.JSONDecodeError:
        logger.warning("Planner response is not a valid JSON")
        if plan_iterations > 0:
            return Command(update=preserve_state_meta_fields(state), goto="reporter")
        else:
            return Command(update=preserve_state_meta_fields(state), goto=END)

    # Validate and fix plan to ensure web search requirements are met
    if isinstance(curr_plan, dict):
        curr_plan = validate_and_fix_plan(curr_plan, configurable.enforce_web_search)

    # Validate Plan
    if isinstance(curr_plan, dict):
        try:
            curr_plan = Plan.model_validate(curr_plan)
        except ValidationError:
            logger.warning(f"Planner response validation error: {curr_plan}")
            return Command(update=preserve_state_meta_fields(state), goto=END)

    if curr_plan.has_enough_context:
        logger.info("Planner response has enough context.")
        return Command(
            update={
                "messages": [AIMessage(content=full_response, name="planner")],
                "current_plan": curr_plan,
                **preserve_state_meta_fields(state),
            },
            goto="reporter",
        )
    return Command(
        update={
            "messages": [AIMessage(content=full_response, name="planner")],
            "current_plan": full_response,
            **preserve_state_meta_fields(state),
        },
        goto="human_feedback",
    )


def extract_plan_content(plan_data: str | dict | Any) -> str:
    """
    Safely extract plan content from different types of plan data.

    Args:
        plan_data: The plan data which can be a string, AIMessage, or dict

    Returns:
        str: The plan content as a string (JSON string for dict inputs, or
    extracted/original string for other types)
    """
    if isinstance(plan_data, str):
        # If it's already a string, return as is
        return plan_data
    elif hasattr(plan_data, "content") and isinstance(plan_data.content, str):
        # If it's an AIMessage or similar object with a content attribute
        logger.debug(
            f"Extracting plan content from message object of type {type(plan_data).__name__}"
        )
        return plan_data.content
    elif isinstance(plan_data, dict):
        # If it's already a dictionary, convert to JSON string
        # Need to check if it's dict with content field (AIMessage-like)
        if "content" in plan_data:
            if isinstance(plan_data["content"], str):
                logger.debug("Extracting plan content from dict with content field")
                return plan_data["content"]
            if isinstance(plan_data["content"], dict):
                logger.debug("Converting content field dict to JSON string")
                return json.dumps(plan_data["content"], ensure_ascii=False)
            else:
                logger.warning(
                    f"Unexpected type for 'content' field in plan_data dict: {type(plan_data['content']).__name__}, converting to string"
                )
                return str(plan_data["content"])
        else:
            logger.debug("Converting plan dictionary to JSON string")
            return json.dumps(plan_data)
    else:
        # For any other type, try to convert to string
        logger.warning(
            f"Unexpected plan data type {type(plan_data).__name__}, attempting to convert to string"
        )
        return str(plan_data)


def human_feedback_node(
    state: State, config: RunnableConfig
) -> Command[Literal["planner", "research_team", "reporter", END]]:
    current_plan = state.get("current_plan", "")
    # check if the plan is auto accepted
    auto_accepted_plan = state.get("auto_accepted_plan", False)
    if not auto_accepted_plan:
        feedback = interrupt("确认计划.")

        # Handle None or empty feedback
        if not feedback:
            logger.warning(
                f"Received empty or None feedback: {feedback}. Returning to planner for new plan."
            )
            return Command(update=preserve_state_meta_fields(state), goto="planner")

        # Normalize feedback string
        feedback_normalized = str(feedback).strip().upper()

        # if the feedback is not accepted, return the planner node
        if feedback_normalized.startswith("[EDIT_PLAN]"):
            logger.info(f"Plan edit requested by user: {feedback}")
            return Command(
                update={
                    "messages": [
                        HumanMessage(content=feedback, name="feedback"),
                    ],
                    **preserve_state_meta_fields(state),
                },
                goto="planner",
            )
        elif feedback_normalized.startswith("[ACCEPTED]"):
            logger.info("Plan is accepted by user.")
        else:
            logger.warning(
                f"Unsupported feedback format: {feedback}. Please use '[ACCEPTED]' to accept or '[EDIT_PLAN]' to edit."
            )
            return Command(update=preserve_state_meta_fields(state), goto="planner")

    # if the plan is accepted, run the following node
    plan_iterations = state["plan_iterations"] if state.get("plan_iterations", 0) else 0
    goto = "research_team"
    try:
        # Safely extract plan content from different types (string, AIMessage, dict)
        original_plan = current_plan

        # Repair the JSON output
        current_plan = repair_json_output(current_plan)
        # parse the plan to dict
        current_plan = json.loads(current_plan)
        current_plan_content = extract_plan_content(current_plan)

        # increment the plan iterations
        plan_iterations += 1
        # parse the plan
        new_plan = json.loads(repair_json_output(current_plan_content))
        # Validate and fix plan to ensure web search requirements are met
        configurable = Configuration.from_runnable_config(config)
        new_plan = validate_and_fix_plan(new_plan, configurable.enforce_web_search)
    except (json.JSONDecodeError, AttributeError) as e:
        logger.warning(
            f"Failed to parse plan: {str(e)}. Plan data type: {type(current_plan).__name__}"
        )
        if isinstance(current_plan, dict) and "content" in original_plan:
            logger.warning("Plan appears to be an AIMessage object with content field")
        if plan_iterations > 1:  # the plan_iterations is increased before this check
            return Command(update=preserve_state_meta_fields(state), goto="reporter")
        else:
            return Command(update=preserve_state_meta_fields(state), goto=END)

    # Build update dict with safe locale handling
    update_dict = {
        "current_plan": Plan.model_validate(new_plan),
        "plan_iterations": plan_iterations,
        **preserve_state_meta_fields(state),
    }

    # Only override locale if new_plan provides a valid value, otherwise use preserved locale
    if new_plan.get("locale"):
        update_dict["locale"] = new_plan["locale"]

    return Command(
        update=update_dict,
        goto=goto,
    )


async def coordinator_node(
    state: State, config: RunnableConfig
) -> Command[
    Literal[
        "planner",
        "background_investigator",
        "coordinator",
        "research_team",
        "researcher",
        "tool_executor",
        END,
    ]
]:
    """Coordinator node that communicate with customers and handle clarification."""
    logger.info("Coordinator talking.")
    configurable = Configuration.from_runnable_config(config)

    # 检查 simple_search_prompt 有值的情况：这是预设好的计划，messages 是预设的，不会出现用户输入"你好"的情况
    # 可以直接根据 goto 字段跳转到 researcher 或 research_team
    if state.get("simple_search_with_prompt", False):
        goto_field = state.get("goto", "researcher")
        logger.info(
            f"Simple search mode with prompt: skipping coordinator logic, going directly to {goto_field}"
        )
        return Command(
            update=preserve_state_meta_fields(state),
            goto=goto_field,
        )

    # Check if clarification is enabled
    is_simple_search = state.get("is_simple_search", False)
    enable_clarification = state.get("enable_clarification", False)
    initial_topic = state.get("research_topic", "")
    clarified_topic = initial_topic
    # 构建 user_role_messages，先完整过滤一遍
    user_role_messages = []
    for i in state["messages"]:
        if type(i) is HumanMessage:
            if "Original" in i.content and "Topic" in i.content:
                continue
            if 'Here is a summary of the conversation to date' in i.content:
                continue
            if i.name and i.name == 'coordinator':
                continue
            user_role_messages.append(i)
        elif type(i) is AIMessage:
            # 过滤掉 planner 和 deepresearch 的消息（coordinator 不需要看到规划思路和中间思考过程）
            if i.name and (i.name == 'planner' or i.name == 'deepresearch'):
                continue
            # 过滤掉 coordinator 的工具调用消息（带 tool_calls 的），
            # coordinator 的工具调用是内部路由指令，不应暴露给下一轮 LLM 上下文
            # 但保留 coordinator 的直接回答（无 tool_calls），如闲聊或基于上下文的直接回复
            if i.name and i.name == 'coordinator' and i.tool_calls:
                continue

            user_role_messages.append(i)
        elif type(i) is dict and 'role' in i and (i.get("role", "") == "user" or i.get("role", "") == "system"):
            if 'name' in i and i.get("role", "") == "coordinator":
                continue
            user_role_messages.append(i)

    # 只保留最近 MAX_CONVERSATION_ROUNDS 轮 user 消息及之间的内容
    # 对 user_role_messages 本身遍历找 user 数量，不依赖 state["messages"]
    MAX_CONVERSATION_ROUNDS = 20
    _user_count = 0
    _trim_idx = None
    for _i in range(len(user_role_messages) - 1, -1, -1):
        if isinstance(user_role_messages[_i], HumanMessage):
            _user_count += 1
            if _user_count >= MAX_CONVERSATION_ROUNDS:
                _trim_idx = _i
                break
    if _trim_idx is not None:
        user_role_messages = user_role_messages[_trim_idx:]

    # Context compression
    llm_token_limit = get_llm_token_limit_by_agent("coordinator")
    compressed_state = ContextManager(llm_token_limit, 5).compress_messages(
        {"messages": user_role_messages}
    )
    # ============================================================
    # BRANCH 1: Clarification DISABLED (Legacy Mode)
    # ============================================================
    if not enable_clarification:
        # Use appropriate prompt based on mode
        prompt_name = "coordinator_fast" if is_simple_search else "coordinator"
        messages = apply_prompt_template(
            get_agent_prompt(prompt_name, configurable.agent_type.value),
            compressed_state,
        )

        # Bind the appropriate handoff tool based on mode
        tools = [handoff_to_tool_executor] if is_simple_search else [handoff_to_planner]

        response = await (
            get_llm_by_type(AGENT_LLM_MAP["coordinator"])
            .bind_tools(tools)
            .ainvoke(messages)
        )

        goto = END
        locale = state.get("locale", "zh-CN")
        logger.info(f"Coordinator locale: {locale}")
        research_topic = state.get("research_topic", "")

        # Process tool calls for legacy mode
        if response.tool_calls:
            try:
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("args", {})

                    if tool_name == "handoff_to_tool_executor":
                        logger.info("Handing off to tool_executor")
                        if tool_args.get("research_topic"):
                            research_topic = tool_args.get("research_topic")
                        goto = "tool_executor"
                        break

                    if tool_name == "handoff_to_planner":
                        logger.info("Handing off to planner")
                        goto = "planner"

                        # Extract research_topic if provided
                        if tool_args.get("research_topic"):
                            research_topic = tool_args.get("research_topic")

                        # 检查 simple_search 模式：只有当 simple_search_with_prompt 为 True 时才直接跳转
                        goto_field = state.get("goto", "")
                        if state.get(
                            "simple_search_with_prompt", False
                        ) and goto_field in ["research_team", "researcher"]:
                            logger.info(
                                f"Simple search mode with prompt: using goto field to jump to {goto_field}"
                            )
                            goto = goto_field

                        break

            except Exception as e:
                logger.error(f"Error processing tool calls: {e}")
                goto = "tool_executor" if is_simple_search else "planner"

        # Do not return early - let code flow to unified return logic below
        # Set clarification variables for legacy mode
        clarification_rounds = 0
        clarification_history = []
        clarified_topic = research_topic

    # ============================================================
    # BRANCH 2: Clarification ENABLED (New Feature)
    # ============================================================
    else:
        # Load clarification state
        clarification_rounds = state.get("clarification_rounds", 0)
        clarification_history = list(state.get("clarification_history", []) or [])
        clarification_history = [item for item in clarification_history if item]
        max_clarification_rounds = state.get("max_clarification_rounds", 3)

        # Prepare the messages for the coordinator
        state_messages = list(state.get("messages", []))
        messages = apply_prompt_template(
            get_agent_prompt("coordinator", configurable.agent_type.value),
            compressed_state,
        )

        clarification_history = reconstruct_clarification_history(
            state_messages, clarification_history, initial_topic
        )
        clarified_topic, clarification_history = build_clarified_topic_from_history(
            clarification_history
        )
        logger.debug("Clarification history rebuilt: %s", clarification_history)

        if clarification_history:
            initial_topic = clarification_history[0]
            latest_user_content = clarification_history[-1]
        else:
            latest_user_content = ""

        # Add clarification status for first round
        if clarification_rounds == 0:
            messages.append(
                {
                    "role": "system",
                    "content": "Clarification mode is ENABLED. Follow the 'Clarification Process' guidelines in your instructions.",
                }
            )

        current_response = latest_user_content or "No response"
        logger.info(
            "Clarification round %s/%s | topic: %s | current user response: %s",
            clarification_rounds,
            max_clarification_rounds,
            clarified_topic or initial_topic,
            current_response,
        )

        clarification_context = f"""Continuing clarification (round {clarification_rounds}/{max_clarification_rounds}):
            User's latest response: {current_response}
            Ask for remaining missing dimensions. Do NOT repeat questions or start new topics."""

        messages.append({"role": "system", "content": clarification_context})

        # Bind both clarification tools - let LLM choose the appropriate one
        tools = [handoff_to_planner, handoff_after_clarification]

        # Check if we've already reached max rounds
        if clarification_rounds >= max_clarification_rounds:
            # Max rounds reached - force handoff by adding system instruction
            logger.warning(
                f"Max clarification rounds ({max_clarification_rounds}) reached. Forcing handoff to planner. Using prepared clarified topic: {clarified_topic}"
            )
            # Add system instruction to force handoff - let LLM choose the right tool
            messages.append(
                {
                    "role": "system",
                    "content": f"MAX ROUNDS REACHED. You MUST call handoff_after_clarification (not handoff_to_planner) with the appropriate locale based on the user's language and research_topic='{clarified_topic}'. Do not ask any more questions.",
                }
            )

        response = await (
            get_llm_by_type(AGENT_LLM_MAP["coordinator"])
            .bind_tools(tools)
            .ainvoke(messages)
        )
        logger.debug(f"Current state messages: {state['messages']}")

        # Initialize response processing variables
        goto = END
        locale = state.get("locale", "zh-CN")
        research_topic = (
            clarification_history[0]
            if clarification_history
            else state.get("research_topic", "")
        )
        if not clarified_topic:
            clarified_topic = research_topic

        # --- Process LLM response ---
        # No tool calls - LLM is asking a clarifying question
        if not response.tool_calls and response.content:
            # Check if we've reached max rounds - if so, force handoff to planner
            if clarification_rounds >= max_clarification_rounds:
                logger.warning(
                    f"Max clarification rounds ({max_clarification_rounds}) reached. "
                    "LLM didn't call handoff tool, forcing handoff to planner."
                )
                goto = "planner"
                # Continue to final section instead of early return
            else:
                # Continue clarification process
                clarification_rounds += 1
                # Do NOT add LLM response to clarification_history - only user responses
                logger.info(
                    f"Clarification response: {clarification_rounds}/{max_clarification_rounds}: {response.content}"
                )

                # Append coordinator's question to messages
                updated_messages = list(state_messages)
                if response.content:
                    updated_messages.append(
                        HumanMessage(content=response.content, name="coordinator")
                    )

                return Command(
                    update={
                        "messages": updated_messages,
                        "locale": locale,
                        "research_topic": research_topic,
                        "clarification_rounds": clarification_rounds,
                        "clarification_history": clarification_history,
                        "clarified_research_topic": clarified_topic,
                        "is_clarification_complete": False,
                        "goto": goto,
                        "__interrupt__": [("coordinator", response.content)],
                    },
                    goto=goto,
                )
        else:
            # LLM called a tool (handoff) or has no content - clarification complete
            if response.tool_calls:
                logger.info(
                    f"Clarification completed after {clarification_rounds} rounds. LLM called handoff tool."
                )
            else:
                logger.warning("LLM response has no content and no tool calls.")
            # goto will be set in the final section based on tool calls

    # ============================================================
    # Final: Build and return Command
    # ============================================================
    # CRITICAL: Do NOT use state["messages"] directly - it contains ALL history including duplicates
    # Instead, build fresh messages from compressed_state to ensure clean context
    messages = list(compressed_state.get("messages", []) or [])
    if response.content and not response.tool_calls:
        messages.append(AIMessage(content=response.content, name="coordinator"))

    # Process tool calls for BOTH branches (legacy and clarification)
    if response.tool_calls:
        # Add the original LLM response (with tool calls) to the messages history.
        # Use the response object directly (not a new AIMessage) to avoid duplicate
        # SSE events, since LangGraph's stream_mode="messages" already captures
        # the same message from the LLM interceptor.
        response.name = "coordinator"
        messages.append(response)
        try:
            for tool_call in response.tool_calls:
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("args", {})

                if tool_name == "handoff_to_tool_executor":
                    logger.info("Handing off to tool_executor")
                    goto = "tool_executor"
                    if tool_args.get("research_topic"):
                        research_topic = tool_args["research_topic"]
                    break

                if tool_name in ["handoff_to_planner", "handoff_after_clarification"]:
                    logger.info("Handing off to planner")
                    goto = "planner"

                    if not enable_clarification and tool_args.get("research_topic"):
                        research_topic = tool_args["research_topic"]

                    if enable_clarification:
                        logger.info(
                            "Using prepared clarified topic: %s",
                            clarified_topic or research_topic,
                        )
                    else:
                        logger.info(
                            "Using research topic for handoff: %s", research_topic
                        )

                    # 检查 simple_search 模式：只有当 simple_search_with_prompt 为 True 时才直接跳转
                    goto_field = state.get("goto", "")
                    if state.get("simple_search_with_prompt", False) and goto_field in [
                        "research_team",
                        "researcher",
                    ]:
                        logger.info(
                            f"Simple search mode with prompt: using goto field to jump to {goto_field}"
                        )
                        goto = goto_field

                    break

        except Exception as e:
            logger.error(f"Error processing tool calls: {e}")
            goto = "planner"
    else:
        # No tool calls detected
        if enable_clarification:
            # BRANCH 2: Fallback to planner to ensure research proceeds
            logger.warning(
                "LLM didn't call any tools. This may indicate tool calling issues with the model. "
                "Falling back to planner to ensure research proceeds."
            )
            logger.debug(f"Coordinator response content: {response.content}")
            logger.debug(f"Coordinator response object: {response}")
            goto = "planner"
        else:
            # BRANCH 1: No tool calls means end workflow gracefully (e.g., greeting handled)
            logger.info("No tool calls in legacy mode - ending workflow gracefully")

    # Apply background_investigation routing if enabled (unified logic)
    # 注意：simple_search 模式下已经设置了 goto，不需要再应用 background_investigation
    if goto == "planner" and state.get("enable_background_investigation"):
        goto = "background_investigator"

    # Set default values for state variables (in case they're not defined in legacy mode)
    if not enable_clarification:
        clarification_rounds = 0
        clarification_history = []

    clarified_research_topic_value = clarified_topic or research_topic

    # 注意：coordinator 直接调 LLM（不是 ReAct agent），框架不会自动追加 messages。
    # 必须手动将带 tool_calls 的 AIMessage 写回 state.messages，否则下一轮恢复时
    # 看不到 coordinator 的历史回复。
    update_dict = {
        "locale": locale,
        "research_topic": research_topic,
        "clarified_research_topic": clarified_research_topic_value,
        "clarification_rounds": clarification_rounds,
        "clarification_history": clarification_history,
        "is_clarification_complete": goto != "coordinator",
        "goto": goto,
        "final_report": state.get("final_report", ""),
    }

    # 将 coordinator 带 tool_calls 的回复写回 state.messages，
    # 供后续轮次的三轮截断逻辑使用（L915 已设 name="coordinator"）
    if response.tool_calls:
        update_dict["messages"] = [response]

    # 问候/闲聊/直接回答场景（goto=END，无 tool_calls）：
    # 直接用 response 对象写回 messages（response 是 LLM ainvoke 直接返回的，
    # id 与 streaming chunks id 一致，reducer 会合并而非 append，不会二次 emit）。
    if goto == END and not response.tool_calls:
        update_dict = {
            **preserve_state_meta_fields(state),
            "goto": goto,
            "messages": [response],
        }

    # 如果是 simple_search 模式且跳转到 researcher/research_team，更新 step 的 description
    is_simple_deepresearch = state.get("is_simple_deepresearch", False)
    simple_search_with_prompt = state.get("simple_search_with_prompt", False)
    if (
        is_simple_deepresearch
        and simple_search_with_prompt
        and goto in ["researcher", "research_team"]
    ):
        current_plan = state.get("current_plan")
        if current_plan and isinstance(current_plan, Plan) and current_plan.steps:
            # 找到第一个 research step，更新其 description
            for step in current_plan.steps:
                # 检查是否是 research step（使用枚举值或字符串比较）
                is_research_step = (
                    step.step_type == StepType.RESEARCH
                    or (
                        isinstance(step.step_type, str)
                        and step.step_type.lower() == "research"
                    )
                    or (hasattr(step, "need_search") and step.need_search)
                )
                if is_research_step:
                    # 使用 coordinator 处理后的内容更新 description
                    step.description = clarified_research_topic_value
                    logger.info(
                        f"[coordinator_node] Simple search mode: updated research step description with coordinator output: {clarified_research_topic_value}"
                    )
                    # 同时更新 analysis step 的 description（如果存在）
                    for analysis_step in current_plan.steps:
                        is_analysis_step = (
                            analysis_step.step_type == StepType.ANALYSIS
                            or (
                                isinstance(analysis_step.step_type, str)
                                and analysis_step.step_type.lower() == "analysis"
                            )
                        )
                        if is_analysis_step and analysis_step != step:
                            analysis_step.description = (
                                f"基于研究结果分析：{clarified_research_topic_value}"
                            )
                            logger.info(
                                f"[coordinator_node] Simple search mode: updated analysis step description"
                            )
                    break
            update_dict["current_plan"] = current_plan

    # clarified_research_topic: Complete clarified topic with all clarification rounds
    return Command(
        update=update_dict,
        goto=goto,
    )


async def reporter_node(state: State, config: RunnableConfig):
    """Reporter node that write a final report."""
    logger.info("Reporter write final report")
    configurable = Configuration.from_runnable_config(config)
    current_plan = state.get("current_plan")
    input_ = {
        "messages": [
            HumanMessage(
                f"# Research Requirements\n\n## Task\n\n{current_plan.title}\n\n## Description\n\n{current_plan.thought}"
            )
        ],
        "locale": state.get("locale", "zh-CN"),
    }
    invoke_messages = apply_prompt_template(
        get_agent_prompt("reporter", configurable.agent_type.value),
        input_,
        dataclasses.asdict(configurable),
    )
    observations = state.get("observations", [])

    # Add a reminder about the new report format, citation style, and table usage
    # invoke_messages.append(
    #     HumanMessage(
    #         content="IMPORTANT: Structure your report according to the format in the prompt. Remember to include:\n\n1. Key Points - A bulleted list of the most important findings\n2. Overview - A brief introduction to the topic\n3. Detailed Analysis - Organized into logical sections\n4. Survey Note (optional) - For more comprehensive reports\n5. Key Citations - List all references at the end\n\nFor citations, DO NOT include inline citations in the text. Instead, place all citations in the 'Key Citations' section at the end using the format: `- [Source Title](URL)`. Include an empty line between each citation for better readability.\n\nPRIORITIZE USING MARKDOWN TABLES for data presentation and comparison. Use tables whenever presenting comparative data, statistics, features, or options. Structure tables with clear headers and aligned columns. Example table format:\n\n| Feature | Description | Pros | Cons |\n|---------|-------------|------|------|\n| Feature 1 | Description 1 | Pros 1 | Cons 1 |\n| Feature 2 | Description 2 | Pros 2 | Cons 2 |",
    #         name="system",
    #     )
    # )

    observation_messages = []
    for observation in observations:
        observation_messages.append(
            HumanMessage(
                content=f"Below are some observations for the research task:\n\n{observation}",
                name="observation",
            )
        )

    # Context compression
    llm_token_limit = get_llm_token_limit_by_agent("reporter")
    compressed_state = ContextManager(llm_token_limit).compress_messages(
        {"messages": observation_messages}
    )
    invoke_messages += compressed_state.get("messages", [])

    logger.debug(f"Current invoke messages: {invoke_messages}")
    response = await get_llm_by_type(AGENT_LLM_MAP["reporter"]).ainvoke(invoke_messages)
    response_content = response.content
    # ChatAnthropic 可能返回 content 列表（thinking + text 块），提取 text 作为报告
    if isinstance(response_content, list):
        text_parts = []
        for block in response_content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        response_content = "".join(text_parts)
    logger.debug(f"reporter response: {response_content}")

    return {"final_report": response_content, "messages": response}


def research_team_node(state: State):
    """Research team node that collaborates on tasks."""
    logger.info("Research team is collaborating on tasks.")
    logger.debug("Entering research_team_node - coordinating research agents")
    pass


async def _execute_agent_step(
    state: State, agent, agent_name: str, config: RunnableConfig = None
) -> Command[Literal["research_team"]]:
    """Helper function to execute a step using the specified agent."""
    logger.debug(f"[_execute_agent_step] Starting execution for agent: {agent_name}")

    current_plan = state.get("current_plan")
    if not current_plan:
        logger.error("[_execute_agent_step] current_plan is None, cannot execute step")
        return Command(update=preserve_state_meta_fields(state), goto="research_team")

    plan_title = current_plan.title
    observations = state.get("observations", [])
    logger.debug(
        f"[_execute_agent_step] Plan title: {plan_title}, observations count: {len(observations)}"
    )

    # Find the first unexecuted step
    current_step = None
    completed_steps = []
    for idx, step in enumerate(current_plan.steps):
        # The default value of step.execution_res is None, and sometimes it may be assigned to ""
        if step.execution_res is None:
            current_step = step
            logger.debug(
                f"[_execute_agent_step] Found unexecuted step at index {idx}: {step.title}"
            )
            break
        else:
            completed_steps.append(step)

    if not current_step:
        logger.warning(
            f"[_execute_agent_step] No unexecuted step found in {len(current_plan.steps)} total steps"
        )
        return Command(update=preserve_state_meta_fields(state), goto="research_team")

    logger.info(
        f"[_execute_agent_step] Executing step: {current_step.title}, agent: {agent_name}"
    )
    logger.debug(
        f"[_execute_agent_step] Completed steps so far: {len(completed_steps)}"
    )

    # Format completed steps information
    completed_steps_info = ""
    if completed_steps:
        completed_steps_info = "# Completed Research Steps\n\n"
        for i, step in enumerate(completed_steps):
            completed_steps_info += f"## Completed Step {i + 1}: {step.title}\n\n"
            completed_steps_info += f"<finding>\n{step.execution_res}\n</finding>\n\n"

    # Prepare the input for the agent with completed steps info
    agent_input = {
        "messages": [
            HumanMessage(
                content=f"# Original Topic\n\n{state['research_topic']}\n\n# Research Topic\n\n{plan_title}\n\n{completed_steps_info}# Current Step\n\n## Title\n\n{current_step.title}\n\n## Description\n\n{current_step.description}\n\n## Locale\n\n{state.get('locale', 'en-US')}"
            )
        ]
    }

    # Add citation reminder for researcher agent
    # if agent_name == "researcher":
    #     agent_input["messages"].append(
    #         HumanMessage(
    #             content="IMPORTANT: DO NOT include inline citations in the text. Instead, track all sources and include a References section at the end using link reference format. Include an empty line between each citation for better readability. Use this format for each reference:\n- [Source Title](URL)\n\n- [Another Source](URL)",
    #             name="system",
    #         )
    #     )

    # Invoke the agent
    default_recursion_limit = 25
    try:
        env_value_str = os.getenv("AGENT_RECURSION_LIMIT", str(default_recursion_limit))
        parsed_limit = int(env_value_str)

        if parsed_limit > 0:
            recursion_limit = parsed_limit
            logger.info(f"Recursion limit set to: {recursion_limit}")
        else:
            logger.warning(
                f"AGENT_RECURSION_LIMIT value '{env_value_str}' (parsed as {parsed_limit}) is not positive. "
                f"Using default value {default_recursion_limit}."
            )
            recursion_limit = default_recursion_limit
    except ValueError:
        raw_env_value = os.getenv("AGENT_RECURSION_LIMIT")
        logger.warning(
            f"Invalid AGENT_RECURSION_LIMIT value: '{raw_env_value}'. "
            f"Using default value {default_recursion_limit}."
        )
        recursion_limit = default_recursion_limit

    logger.debug(f"Agent input: {agent_input}")

    # Validate message content before invoking agent
    try:
        validated_messages = validate_message_content(agent_input["messages"])
        agent_input["messages"] = validated_messages
    except Exception as validation_error:
        logger.error(f"Error validating agent input messages: {validation_error}")

    # Apply context compression to prevent token overflow (Issue #721)
    llm_token_limit = get_llm_token_limit_by_agent(agent_name)
    if llm_token_limit:
        token_count_before = sum(
            len(str(msg.content).split())
            for msg in agent_input.get("messages", [])
            if hasattr(msg, "content")
        )
        compressed_state = ContextManager(
            llm_token_limit, preserve_prefix_message_count=3
        ).compress_messages({"messages": agent_input["messages"]})
        agent_input["messages"] = compressed_state.get("messages", [])
        token_count_after = sum(
            len(str(msg.content).split())
            for msg in agent_input.get("messages", [])
            if hasattr(msg, "content")
        )
        logger.info(
            f"Context compression for {agent_name}: {len(compressed_state.get('messages', []))} messages, "
            f"estimated tokens before: ~{token_count_before}, after: ~{token_count_after}"
        )

    result = {}
    agent = cast(CompiledStateGraph, agent)
    with get_tool_callback() as cb:
        try:
            result = await agent.ainvoke(
                input=agent_input,
                config={
                    "recursion_limit": recursion_limit,
                },
            )
        except GraphRecursionError:
            recursion_model = get_llm_by_type(
                AGENT_LLM_MAP.get(agent_name, "streaming")
            )
            recursion_msg = extract_replayable_messages(cb.messages)
            system_prompt = SystemMessage(
                render_system_prompt("common/summary").format(
                    messages=[
                        i.content if isinstance(i.content, str)
                        else json.dumps(i.content, ensure_ascii=False)
                        for i in recursion_msg
                    ]
                )
            )
            # BUG FIX: Pass both system prompt and history messages so the model
            # sees structured conversation context, preserving complete context
            # for subsequent state updates.
            recursion_result = await recursion_model.ainvoke([
                system_prompt,
                HumanMessage(content="请根据以上指令和消息内容，生成结构化总结报告。"),
            ])
            # BUG FIX: Preserve input messages alongside the summary result
            # so downstream processing can properly reconstruct context.
            result["messages"] = list(agent_input.get("messages", [])) + [recursion_result]
            logger.warning(f"ReAct rollback execution succeeded: {recursion_result}")
        except Exception as e:
            import traceback

            error_traceback = traceback.format_exc()
            error_message = f"Error executing {agent_name} agent for step '{current_step.title}': {str(e)}"
            logger.exception(error_message)
            logger.error(f"Full traceback:\n{error_traceback}")

            # Enhanced error diagnostics for content-related errors
            if "Field required" in str(e) and "content" in str(e):
                logger.error("Message content validation error detected")
                for i, msg in enumerate(agent_input.get("messages", [])):
                    logger.error(
                        f"Message {i}: type={type(msg).__name__}, "
                        f"has_content={hasattr(msg, 'content')}, "
                        f"content_type={type(msg.content).__name__ if hasattr(msg, 'content') else 'N/A'}, "
                        f"content_len={len(str(msg.content)) if hasattr(msg, 'content') and msg.content else 0}"
                    )

            detailed_error = f"[ERROR] {agent_name.capitalize()} Agent Error\n\nStep: {current_step.title}\n\nError Details:\n{str(e)}\n\nPlease check the logs for more information."
            current_step.execution_res = detailed_error

            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content=detailed_error,
                            name=agent_name,
                        )
                    ],
                    "observations": observations + [detailed_error],
                    "current_plan": current_plan,
                    **preserve_state_meta_fields(state),
                },
                goto="research_team",
            )

    # Process the result
    response_content = result["messages"][-1].content

    # Sanitize response to remove extra tokens and truncate if needed
    response_content = sanitize_tool_response(str(response_content))

    logger.debug(f"{agent_name.capitalize()} full response: {response_content}")

    # Update the step with the execution result
    current_step.execution_res = (
        response_content if response_content != "" else "未产生最终结果"
    )
    logger.info(f"Step '{current_step.title}' execution completed by {agent_name}")

    # Include all messages from agent result to preserve intermediate tool calls/results
    # This ensures multiple web_search calls all appear in the stream, not just the final result
    agent_messages = result.get("messages", [])
    logger.debug(
        f"{agent_name.capitalize()} returned {len(agent_messages)} messages. "
        f"Message types: {[type(msg).__name__ for msg in agent_messages]}"
    )

    # Count tool messages for logging
    tool_message_count = sum(
        1 for msg in agent_messages if isinstance(msg, ToolMessage)
    )
    if tool_message_count > 0:
        logger.info(
            f"{agent_name.capitalize()} agent made {tool_message_count} tool calls. "
            f"All tool results will be preserved and streamed to frontend."
        )

    return Command(
        update={
            "messages": agent_messages,
            "observations": observations + [response_content],
            "current_plan": current_plan,
            **preserve_state_meta_fields(state),
        },
        goto="research_team",
    )


async def _setup_and_execute_agent_step(
    state: State,
    config: RunnableConfig,
    agent_name: str,
    default_tools: list,
) -> Command[Literal["research_team"]]:
    """Helper function to set up an agent with appropriate tools and execute a step.

    This function handles the common logic forresearcher_node:
    1. Configures MCP servers and tools based on agent type
    2. Creates an agent with the appropriate tools or uses the default agent
    3. Executes the agent on the current step

    Args:
        state: The current state
        config: The runnable config
        agent_name: The type of agent ("researcher")
        default_tools: The default tools to add to the agent

    Returns:
        Command to update state and go to research_team
    """
    configurable = Configuration.from_runnable_config(config)

    enable_mcp_servers, loaded_tools = await merge_mcp_tools(
        agent_name, configurable.mcp_settings, default_tools
    )

    prompt_template = get_agent_prompt(agent_name, configurable.agent_type.value)

    if enable_mcp_servers:
        # llm_token_limit = get_llm_token_limit_by_agent(agent_name)
        # pre_model_hook = partial(ContextManager(llm_token_limit, 3).compress_messages)

        # MCP的tools统一注入task_id, 内部的tools不做注入
        injection_tools = [wrap_tools_with_dynamic_params(t) for t in loaded_tools]
        agent = create_agent(
            graph_name,
            agent_name,
            injection_tools,
            prompt_template,
            # pre_model_hook,
            interrupt_before_tools=configurable.interrupt_before_tools,
        )
        return await _execute_agent_step(state, agent, agent_name, config)
    else:
        # Use default tools if no MCP servers are configured
        # llm_token_limit = get_llm_token_limit_by_agent(agent_name)
        # pre_model_hook = partial(ContextManager(llm_token_limit, 3).compress_messages)

        agent = create_agent(
            graph_name,
            agent_name,
            default_tools,
            prompt_template,
            # pre_model_hook,
            interrupt_before_tools=configurable.interrupt_before_tools,
        )
        return await _execute_agent_step(state, agent, agent_name, config)


async def researcher_node(
    state: State, config: RunnableConfig
) -> Command[Literal["research_team"]]:
    """Researcher node that do research"""
    logger.info("Researcher node is researching.")
    logger.debug("[researcher_node] Starting researcher agent")

    # configurable = Configuration.from_runnable_config(config)
    # logger.debug(
    #     f"[researcher_node] Max search results: {configurable.max_search_results}"
    # )

    # Build tools list based on configuration
    tools = []
    return await _setup_and_execute_agent_step(
        state,
        config,
        "researcher",
        tools,
    )


async def analyst_node(
    state: State, config: RunnableConfig
) -> Command[Literal["research_team"]]:
    """Analyst node that performs reasoning and analysis without code execution.

    This node handles tasks like:
    - Cross-validating information from multiple sources
    - Synthesizing research findings
    - Comparative analysis
    - Pattern recognition and trend analysis
    - General reasoning tasks that don't require code
    """
    logger.info("Analyst node is analyzing.")
    logger.debug("[analyst_node] Starting analyst agent for reasoning/analysis tasks")

    # Analyst uses no tools - pure LLM reasoning
    # Build tools list based on configuration
    tools = []
    return await _setup_and_execute_agent_step(state, config, "analyst", tools)


async def _execute_dict_tool(tool_def: dict, args: dict) -> Any:
    """Execute a dict-style tool definition."""
    # Dict tools are wrapped by wrap_tools_with_dynamic_params which returns
    # BaseTool instances, so this is a fallback.
    from langchain_core.tools import tool as langchain_tool

    func = tool_def.get("func")
    if func:
        return await func(**args) if asyncio.iscoroutinefunction(func) else func(**args)
    raise ValueError(f"Cannot execute dict tool: {tool_def.get('name', 'unknown')}")


async def tool_executor_node(
    state: State, config: RunnableConfig
) -> Command[Literal["summary"]]:
    """Tool Executor node that executes tool calls via ReAct agent, collects raw results.

    用 create_react_agent 实现多轮工具调用，提取 ToolMessage 内容到 observations。
    SSE 流通过 messages 自动发出 tool_calls / tool_results 事件。
    """
    logger.info("[tool_executor] Starting ReAct agent for tool execution")

    configurable = Configuration.from_runnable_config(config)

    # Step 1: Prepare input context
    research_topic = (
        state.get("clarified_research_topic")
        or state.get("research_topic", "")
    )
    locale = state.get("locale", "zh-CN")
    observations = state.get("observations", [])

    # Clean up merged/combined queries
    if research_topic and " - " in research_topic:
        parts = research_topic.split(" - ")
        original_topic = research_topic
        research_topic = parts[-1].strip()
        if research_topic != original_topic:
            logger.info(
                f"[tool_executor] Cleaned merged research topic. "
                f"Original: {original_topic}, Extracted: {research_topic}"
            )

    logger.debug(f"[tool_executor] Research topic: {research_topic}, Locale: {locale}")

    # Step 2: Build conversation context from state["messages"]
    # 只保留当前轮的 user 消息，不回溯之前轮次的历史上下文
    context_messages = []
    state_messages = state.get("messages", [])

    if not state_messages:
        logger.debug("[tool_executor] No messages in state")
    else:
        # 找到最近的一条 user 消息（当前轮）
        current_msg_index = -1
        for idx in range(len(state_messages) - 1, -1, -1):
            msg = state_messages[idx]
            is_user_msg = False
            msg_content = ""
            msg_name = ""

            if isinstance(msg, HumanMessage):
                is_user_msg = True
                msg_name = msg.name or ""
                msg_content = str(msg.content) if msg.content else ""
            elif isinstance(msg, dict) and msg.get("role") == "user":
                is_user_msg = True
                msg_name = msg.get("name", "") or ""
                msg_content = str(msg.get("content", ""))

            if is_user_msg and msg_name != "coordinator":
                if "Original" in msg_content and "Topic" in msg_content:
                    continue
                if "summary of the conversation to date" in msg_content:
                    continue
                current_msg_index = idx
                break

        if current_msg_index >= 0:
            # 只取当前轮 user 消息，不回溯之前轮次的 messages
            msg = state_messages[current_msg_index]
            msg_content = ""
            if isinstance(msg, HumanMessage):
                msg_content = str(msg.content) if msg.content else ""
            elif isinstance(msg, dict):
                msg_content = str(msg.get("content", ""))
            if msg_content:
                # 如果 coordinator 补全了 query（research_topic 与原始内容不同），
                # 用补全后的 query 替换原始 user 消息，让 tool_executor 搜正确的内容
                if research_topic and research_topic != msg_content:
                    logger.info(
                        f"[tool_executor] Replacing original user message with coordinator-refined query: "
                        f"original='{msg_content[:100]}' -> refined='{research_topic[:100]}'"
                    )
                    context_messages.append(
                        HumanMessage(
                            content=f"用户原始问题：{msg_content}\n\n统筹分析后确定的完整查询意图：{research_topic}"
                        )
                    )
                else:
                    context_messages.append(HumanMessage(content=msg_content))

        if context_messages:
            logger.debug(f"[tool_executor] Built context from {len(context_messages)} message(s) (current round only)")

    # Step 3: Load MCP tools (same configuration as researcher node)
    enable_mcp_servers, loaded_tools = await merge_mcp_tools(
        "researcher",
        configurable.mcp_settings,
        [],
    )

    logger.info(f"[tool_executor] MCP servers enabled: {enable_mcp_servers}, tools count: {len(loaded_tools)}")

    # Step 4: Build agent input
    agent_messages = []
    agent_messages.extend(context_messages)
    agent_input = {"messages": agent_messages}

    # Context compression
    llm_token_limit = get_llm_token_limit_by_agent("simple_researcher")
    if llm_token_limit:
        compressed_state = ContextManager(
            llm_token_limit, preserve_prefix_message_count=3
        ).compress_messages({"messages": agent_input["messages"]})
        agent_input["messages"] = compressed_state.get("messages", [])

    input_message_count = len(agent_input["messages"])
    logger.debug(f"[tool_executor] Input contains {input_message_count} messages")

    # Step 5: Create ReAct agent with tool_executor prompt
    prompt_template = get_agent_prompt("tool_executor", configurable.agent_type.value)

    if enable_mcp_servers:
        injection_tools = [wrap_tools_with_dynamic_params(t) for t in loaded_tools]
        agent = create_agent(
            graph_name,
            "tool_executor",
            injection_tools,
            prompt_template,
            interrupt_before_tools=configurable.interrupt_before_tools,
        )
    else:
        agent = create_agent(
            graph_name,
            "tool_executor",
            [],
            prompt_template,
            interrupt_before_tools=configurable.interrupt_before_tools,
        )

    # Step 6: Execute agent
    default_recursion_limit = 25
    recursion_limit = int(os.getenv("AGENT_RECURSION_LIMIT", str(default_recursion_limit)))
    if recursion_limit <= 0:
        recursion_limit = default_recursion_limit

    logger.info(f"[tool_executor] Starting agent execution with recursion_limit: {recursion_limit}")

    result = {}
    try:
        result = await agent.ainvoke(
            input=agent_input,
            config={"recursion_limit": recursion_limit},
        )
        logger.info("[tool_executor] Agent execution completed successfully")
    except GraphRecursionError:
        logger.warning("[tool_executor] GraphRecursionError: recursion limit exceeded")
        recursion_model = get_llm_by_type(AGENT_LLM_MAP.get("researcher", "streaming"))
        recursion_msg = extract_replayable_messages(agent_input.get("messages", []))
        system_prompt = SystemMessage(
            render_system_prompt("common/summary").format(
                messages=[i.content if isinstance(i.content, str) else json.dumps(i.content, ensure_ascii=False) for i in recursion_msg]
            )
        )
        recursion_result = await recursion_model.ainvoke([
            system_prompt,
            HumanMessage(content="请根据以上指令和消息内容，生成结构化总结报告。"),
        ])
        result["messages"] = list(agent_input.get("messages", [])) + [recursion_result]
        logger.warning(f"[tool_executor] Fallback execution succeeded")
    except Exception as e:
        logger.exception(f"[tool_executor] Execution error: {e}")
        error_message = f"[ERROR] 工具执行失败: {str(e)}"
        return Command(
            update={
                "observations": observations + [error_message],
                **preserve_state_meta_fields(state),
            },
            goto="summary",
        )

    # Step 7: Extract ToolMessage content into observations
    # Only keep raw tool results, discard AIMessage thinking text
    result_messages = result.get("messages", [])
    tool_observations = []
    for m in result_messages:
        if isinstance(m, ToolMessage):
            content = str(m.content or "")
            if content.strip():
                tool_observations.append(content)

    tool_obs_content = "\n\n---\n\n".join(tool_observations) if tool_observations else "未产生有效结果"
    logger.info(f"[tool_executor] Extracted {len(tool_observations)} tool results")

    # Step 8: Build Command.update
    # Include messages (for SSE streaming) + observations (for summary_node)
    # Find current user message for idempotent add_messages merging
    current_user_msg = None
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            if msg.name == "coordinator":
                continue
            content = str(msg.content or "")
            if "Original Topic" in content or "summary of the conversation to date" in content:
                continue
            current_user_msg = msg
            break

    if current_user_msg is not None and len(result_messages) > input_message_count:
        new_messages = [current_user_msg] + result_messages[input_message_count:]
    elif current_user_msg is not None:
        new_messages = [current_user_msg] + result_messages
    else:
        new_messages = result_messages if result_messages else []

    logger.debug(
        f"[tool_executor] Agent returned {len(result_messages)} total messages, "
        f"input_message_count={input_message_count}, "
        f"returning {len(new_messages)} messages"
    )

    return Command(
        update={
            "messages": new_messages,
            "observations": observations + [tool_obs_content],
            **preserve_state_meta_fields(state),
        },
        goto="summary",
    )


async def summary_node(
    state: State, config: RunnableConfig
) -> dict:
    """Summary node that produces the final answer from raw observations.

    从 simple_researcher_node 拆分而来，职责是：
    1. 接收 tool_executor_node 收集的原始结果（observations）
    2. 用 reporter 的 LLM 做归纳总结
    3. 输出最终正文（name="deepresearch"）
    4. 结束到 END
    """
    logger.info("[summary_node] Producing final summary from observations")

    configurable = Configuration.from_runnable_config(config)
    research_topic = (
        state.get("clarified_research_topic")
        or state.get("research_topic", "")
    )
    locale = state.get("locale", "zh-CN")
    observations = state.get("observations", [])

    # 如果没有 observations，直接返回错误
    if not observations:
        error_msg = "没有可用的研究结果，无法生成总结。"
        return Command(
            update={
                **preserve_state_meta_fields(state),
            },
            goto=END,
        )

    # 构建 reporter 的输入：用原始结果做总结
    # 拆分为多条 message，不把原始结果拼到 system prompt 后面
    observation_content = "\n\n".join([
        f"### 搜索结果 {i+1}\n\n{obs}"
        for i, obs in enumerate(observations)
    ])

    input_ = {
        "messages": [
            {"role": "user", "content": f"# 研究需求\n\n## 任务\n\n{research_topic}"},
            {"role": "user", "content": f"## 原始研究结果\n\n{observation_content}"},
        ],
        "locale": locale,
    }

    prompt_template = get_agent_prompt("summary", configurable.agent_type.value)
    invoke_messages = apply_prompt_template(
        prompt_template,
        input_,
        dataclasses.asdict(configurable),
    )

    response = await get_llm_by_type(AGENT_LLM_MAP["reporter"]).ainvoke(invoke_messages)
    response_content = sanitize_tool_response(str(response.content))

    logger.info(f"[summary_node] Summary produced, length: {len(response_content)}")

    # 返回 LLM 直接返回的 response 对象，而非手动构造的 AIMessage。
    # response 的 id 与 streaming chunks 的 id 一致，LangGraph reducer 会合并而非 append，
    # 不会产生 checkpoint_id="" 的二次 emit。
    # 同时保留 final_report 作为快速引用。
    # summary_node 不用控制 goto，用 return dict 即可。
    return {
        **preserve_state_meta_fields(state),
        "messages": [response],
    }


async def simple_researcher_node(
    state: State, config: RunnableConfig
) -> Command[Literal[END]]:
    """Simple Researcher node that executes direct React mode.

    This is a completely independent execution path:
    - Does not depend on Plan/Step structure
    - Does not go through planner/analyst/reporter nodes
    - Directly uses user input to create a React Agent
    - Uses enhanced prompt to guide agent for multi-turn search, analysis, synthesis and self-review
    - Returns final result directly to END

    Responsibilities:
    1. Extract user question and context from state, including conversation history
    2. Filter and compress conversation history for relevant context (like coordinator_node)
    3. Load MCP tools (same as researcher node)
    4. Create React Agent with enhanced prompt (with self-review instructions)
    5. Execute agent and stream results
    6. Return agent's final output as final_report
    """
    logger.info("[simple_researcher] Direct React mode: executing research")

    configurable = Configuration.from_runnable_config(config)

    # Step 1: Prepare input context
    research_topic = (
        state.get("clarified_research_topic")
        or state.get("research_topic", "")
    )
    locale = state.get("locale", "zh-CN")
    observations = state.get("observations", [])

    # Clean up merged/combined queries: if research_topic contains " - " pattern
    # indicating multiple questions merged together, extract only the last one
    if research_topic and " - " in research_topic:
        # Split by " - " and take the last segment (most recent query)
        parts = research_topic.split(" - ")
        original_topic = research_topic
        research_topic = parts[-1].strip()
        if research_topic != original_topic:
            logger.info(
                f"[simple_researcher] Cleaned merged research topic. "
                f"Original: {original_topic}, Extracted: {research_topic}"
            )

    logger.debug(f"[simple_researcher] Research topic: {research_topic}")
    logger.debug(f"[simple_researcher] Locale: {locale}")

    # Step 2: Build conversation context from state["messages"]
    # Strategy:
    # 1. Find the most recent user message (this is the current research question)
    # 2. Collect all messages up to and including that message
    # 3. Return ALL of them as context (no need to add separately)

    context_messages = []
    state_messages = state.get("messages", [])

    if not state_messages:
        logger.debug("[simple_researcher] No messages in state")
    else:
        # First, find the index of the most recent user message that should be our current query
        # We iterate backwards to find the latest unprocessed user message
        current_msg_index = -1
        for idx in range(len(state_messages) - 1, -1, -1):
            msg = state_messages[idx]

            # Check if this could be a user message
            is_user_msg = False
            msg_content = ""
            msg_name = ""

            if isinstance(msg, HumanMessage):
                is_user_msg = True
                msg_name = msg.name or ""
                msg_content = str(msg.content) if msg.content else ""
            elif isinstance(msg, dict) and msg.get("role") == "user":
                is_user_msg = True
                msg_name = msg.get("name", "") or ""
                msg_content = str(msg.get("content", ""))

            if is_user_msg and msg_name != "coordinator":
                # Skip system messages
                if "Original" in msg_content and "Topic" in msg_content:
                    continue
                if "summary of the conversation to date" in msg_content:
                    continue
                # Found a valid user message - this is our current query position
                current_msg_index = idx
                break

        if current_msg_index < 0:
            logger.warning("[simple_researcher] Could not find current user message in state messages")
        else:
            logger.debug(f"[simple_researcher] Found current query at message index {current_msg_index}")

            # Strategy: Collect messages BACKWARDS from current_msg_index to ensure
            # the most recent user query is ALWAYS included (it's the first one processed).
            # This prevents the new question from being dropped when the conversation
            # exceeds max_context_messages. Then reverse for chronological order.
            max_context_messages = 20  # Limit total messages to prevent overflow

            for idx in range(current_msg_index, -1, -1):  # iterate backwards from current_msg_index
                if len(context_messages) >= max_context_messages:
                    break

                msg = state_messages[idx]
                msg_name = ""
                msg_content = ""
                msg_type = None

                if isinstance(msg, HumanMessage):
                    msg_name = msg.name or ""
                    msg_content = str(msg.content) if msg.content else ""
                    msg_type = "human"
                elif isinstance(msg, AIMessage):
                    msg_name = msg.name or ""
                    msg_content = str(msg.content) if msg.content else ""
                    msg_type = "ai"
                elif isinstance(msg, dict):
                    msg_name = msg.get("name", "") or ""
                    msg_content = str(msg.get("content", ""))
                    msg_type = msg.get("role", "")

                # Skip unwanted message types and sources
                if msg_type == "human" or msg_type == "user":
                    if msg_name == "coordinator" or ("Original" in msg_content and "Topic" in msg_content):
                        continue
                    if "summary of the conversation to date" in msg_content:
                        continue
                    display_content = msg_content if len(msg_content) <= 500 else msg_content[:500] + "..."
                    context_messages.append(HumanMessage(content=display_content))

                elif msg_type == "ai" or msg_type == "assistant":
                    # 不要过滤 "deepresearch" - 这是所有 agent 共享的 graph_name，
                    # 过滤它会导致前几轮 simple_researcher 的 assistant 回复被跳过
                    if msg_name in ("planner", "coordinator"):
                        continue
                    # 过滤掉内容为空的 assistant 消息（例如仅包含 tool_calls 的中间消息）
                    if not msg_content or not msg_content.strip():
                        continue
                    display_content = msg_content if len(msg_content) <= 500 else msg_content[:500] + "..."
                    context_messages.append(AIMessage(content=display_content))

            # Reverse to chronological order (oldest first, current query last)
            # This is required because we collected backwards from current_msg_index
            context_messages.reverse()

        if context_messages:
            logger.debug(f"[simple_researcher] Built context from {len(context_messages)} messages (including current query)")

    # Step 3: Load MCP tools (same configuration as researcher node)
    enable_mcp_servers, loaded_tools = await merge_mcp_tools(
        "researcher",
        configurable.mcp_settings,
        [],  # No default tools
    )

    logger.info(f"[simple_researcher] MCP servers enabled: {enable_mcp_servers}, tools count: {len(loaded_tools)}")

    # Step 4: Build agent input with proper message array
    # All messages (including the current research question) are already in context_messages
    agent_messages = []
    agent_messages.extend(context_messages)

    agent_input = {"messages": agent_messages}
    logger.debug(f"[simple_researcher] Built agent input with {len(agent_messages)} messages")

    # Apply context compression to prevent token overflow (same pattern as _execute_agent_step)
    # This is critical for multi-turn conversations where accumulated messages can exceed token limits
    llm_token_limit = get_llm_token_limit_by_agent("simple_researcher")
    if llm_token_limit:
        compressed_state = ContextManager(
            llm_token_limit, preserve_prefix_message_count=3
        ).compress_messages({"messages": agent_input["messages"]})
        agent_input["messages"] = compressed_state.get("messages", [])

    # Record input message count for reference
    # We'll use this to identify which messages are newly generated vs input context
    input_message_count = len(agent_input["messages"])
    logger.debug(f"[simple_researcher] Input contains {input_message_count} messages")

    # Step 5: Create React Agent with simple_researcher prompt (NOT researcher prompt)
    # simple_researcher has its own comprehensive prompt designed for single-node complete execution
    prompt_template = get_agent_prompt("simple_researcher", configurable.agent_type.value)

    if enable_mcp_servers:
        injection_tools = [wrap_tools_with_dynamic_params(t) for t in loaded_tools]
        agent = create_agent(
            graph_name,
            "simple_researcher",
            injection_tools,
            prompt_template,
            interrupt_before_tools=configurable.interrupt_before_tools,
        )
    else:
        agent = create_agent(
            graph_name,
            "simple_researcher",
            [],
            prompt_template,
            interrupt_before_tools=configurable.interrupt_before_tools,
        )

    # Step 6: Execute agent
    default_recursion_limit = 25
    recursion_limit = int(os.getenv("AGENT_RECURSION_LIMIT", str(default_recursion_limit)))
    if recursion_limit <= 0:
        recursion_limit = default_recursion_limit

    logger.info(f"[simple_researcher] Starting agent execution with recursion_limit: {recursion_limit}")

    result = {}
    try:
        result = await agent.ainvoke(
            input=agent_input,
            config={"recursion_limit": recursion_limit},
        )
        logger.info("[simple_researcher] Agent execution completed successfully")
    except GraphRecursionError:
        # Fallback for recursion limit exceeded
        logger.warning("[simple_researcher] GraphRecursionError: recursion limit exceeded, using fallback")
        recursion_model = get_llm_by_type(
            AGENT_LLM_MAP.get("researcher", "streaming")
        )
        recursion_msg = extract_replayable_messages(agent_input.get("messages", []))
        system_prompt = SystemMessage(
            render_system_prompt("common/summary").format(
                messages=[i.content if isinstance(i.content, str) else json.dumps(i.content, ensure_ascii=False) for i in recursion_msg]
            )
        )
        # BUG FIX: Pass both system prompt and history messages so the model
        # sees structured conversation context, not just the textified {messages}.
        recursion_result = await recursion_model.ainvoke([
            system_prompt,
            HumanMessage(content="请根据以上指令和消息内容，生成结构化总结报告。"),
        ])
        # BUG FIX: Preserve input messages so result_messages count stays meaningful
        # and the caller's message-slicing logic works correctly.
        result["messages"] = list(agent_input.get("messages", [])) + [recursion_result]
        logger.warning(f"[simple_researcher] Fallback execution succeeded: {recursion_result}")
    except Exception as e:
        logger.exception(f"[simple_researcher] Execution error: {e}")
        error_message = f"[ERROR] 研究执行失败: {str(e)}"
        return Command(
            update={
                "final_report": error_message,
                "observations": observations + [error_message],
                **preserve_state_meta_fields(state),
            },
            goto=END,
        )

    # Step 7: Extract and process final result
    if result.get("messages"):
        response_content = result["messages"][-1].content
        response_content = sanitize_tool_response(str(response_content))
    else:
        response_content = "未产生最终结果"

    logger.info(f"[simple_researcher] Final result extracted, length: {len(response_content)}")

    final_result = response_content if response_content else "未产生最终结果"

    logger.info("[simple_researcher] Direct React mode completed, returning to END")

    # Step 8: Return result and go to END
    # IMPORTANT: Return only the original current user message (preserving its ID)
    # plus the NEW messages generated by the agent execution.
    # Do NOT return the rebuilt input messages (context_messages) - they were created
    # with new IDs during Step 2, and returning them would cause duplicates when
    # add_messages merges them with the checkpoint's existing messages.
    result_messages = result.get("messages", [])

    # Find the current user message from state (preserves original ID for add_messages dedup)
    current_user_msg = None
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            if msg.name == "coordinator":
                continue
            content = str(msg.content or "")
            if "Original Topic" in content or "summary of the conversation to date" in content:
                continue
            current_user_msg = msg
            break

    if current_user_msg is not None and len(result_messages) > input_message_count:
        # Return original current user message + only the newly generated messages
        new_messages = [current_user_msg] + result_messages[input_message_count:]
    elif current_user_msg is not None:
        # Fallback: agent returned fewer/equal messages than input (e.g., recursion error)
        new_messages = [current_user_msg] + result_messages
    else:
        # First turn or no user message found: return as-is
        new_messages = result_messages if result_messages else []

    logger.debug(
        f"[simple_researcher] Agent returned {len(result_messages)} total messages, "
        f"input_message_count={input_message_count}, "
        f"returning {len(new_messages)} messages (current_user_msg={current_user_msg is not None})"
    )

    return Command(
        update={
            "messages": new_messages,
            "observations": observations + [final_result],
            "final_report": final_result,
            **preserve_state_meta_fields(state),
        },
        goto=END,
    )

## 并行节点
async def parallel_researcher_node(
    state: State,
    config: RunnableConfig,
):
    logger.info("开始并行执行 Research 步骤...")
    current_plan = state["current_plan"]

    # 从路由函数注入的就绪列表中获取本次应执行的 RESEARCH 步骤
    research_steps = state.get("_parallel_tasks", [])
    if not research_steps and isinstance(current_plan, Plan):
        completed_step_titles = {
            step.title
            for step in current_plan.steps
            if step.execution_res is not None
        }
        research_steps = []
        for step in current_plan.steps:
            if step.execution_res is not None:
                continue
            depends_steps = []
            if hasattr(step, "association") and step.association:
                depends_steps = getattr(step.association, "depends_on_steps", [])
            if all(dep_title in completed_step_titles for dep_title in depends_steps):
                if step.step_type == StepType.RESEARCH:
                    research_steps.append(step)

    if not research_steps:
        logger.warning("当前没有找到满足并发执行条件（所有前置依赖已完成）的 Research 步骤")
        return Command(update=preserve_state_meta_fields(state), goto="research_team")

    logger.info(f"本次将并行调度执行以下 {len(research_steps)} 个就绪任务: {[s.title for s in research_steps]}")

    # 统一初始化一次 MCP 工具
    configurable = Configuration.from_runnable_config(config)
    enable_mcp_servers, shared_loaded_tools = await merge_mcp_tools(
        "researcher", configurable.mcp_settings, []
    )
    from src.agents.tool_injection import wrap_tools_with_dynamic_params
    if enable_mcp_servers:
        shared_tools = [wrap_tools_with_dynamic_params(t) for t in shared_loaded_tools]
    else:
        shared_tools = []

    semaphore = asyncio.Semaphore(4)

    async def run_step(step_target):
        async with semaphore:
            sub_state = deepcopy(state)
            sub_plan = deepcopy(current_plan)

            sub_plan.steps = [deepcopy(step_target)]
            sub_state["current_plan"] = sub_plan

            # 创建 Agent 并运行
            prompt_template = get_agent_prompt("researcher", configurable.agent_type.value)

            # 直接创建临时的独立运行 Agent，但共享底层工具池
            sub_agent = create_agent(
                graph_name,
                "researcher",
                shared_tools,
                prompt_template,
                interrupt_before_tools=configurable.interrupt_before_tools,
            )

            # 直接调用底层的真实执行函数
            cmd_result = await _execute_agent_step(sub_state, sub_agent, "researcher", config)

            execution_res = "未产生最终结果"
            agent_messages = []

            if cmd_result and cmd_result.update:
                agent_messages = cmd_result.update.get("messages", [])

                # 优先从子生命周期的 sub_plan 中提取回写的 execution_res
                if sub_plan.steps and sub_plan.steps[0].execution_res:
                    execution_res = sub_plan.steps[0].execution_res
                elif cmd_result.update.get("observations"):
                    execution_res = cmd_result.update.get("observations")[-1]

            return {
                "title": step_target.title,
                "execution_res": execution_res,
                "messages": agent_messages
            }

    # 并发组装执行与数据结构化合并
    tasks = [run_step(step) for step in research_steps]
    task_results = await asyncio.gather(*tasks, return_exceptions=True)

    merged_messages = []
    new_observations = []

    # 浅拷贝一份原有的 plan 步骤列表用于更新
    updated_steps = list(current_plan.steps)

    for res in task_results:
        if isinstance(res, Exception):
            logger.error(f"并行 Research 步骤执行异常: {res}")
            continue

        # 精准回写原 plan 中对应 title 的步骤结果
        for original_step in updated_steps:
            if original_step.title == res["title"]:
                original_step.execution_res = res["execution_res"]
                new_observations.append(f"【步骤完成: {res['title']}】\n{res['execution_res']}")
                break

        if res["messages"]:
            merged_messages.extend(res["messages"])

    # 更新 plan 对象的 steps
    current_plan.steps = updated_steps

    logger.info(f"所有并行 Research 步骤执行完毕，成功合并 {len(task_results)} 个任务结果")

    # 清理用完的临时并发标记字段，防止污染下一次判定循环
    state_updates = {
        "messages": merged_messages,
        "observations": state.get("observations", []) + new_observations,
        "current_plan": current_plan,
        "_parallel_tasks": [],  # 显式重置
        **preserve_state_meta_fields(state),
    }

    return Command(
        update=state_updates,
        goto="research_team",
    )

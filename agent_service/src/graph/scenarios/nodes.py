import json
import re
from typing import cast, Literal

from loguru import logger
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

from src.config.agents import AGENT_LLM_MAP
from src.llms.llm import get_llm_by_type, get_llm_token_limit_by_agent
from src.utils.context_manager import ContextManager, validate_message_content
from src.prompts.template import (
    load_prompts_tree,
    apply_prompt_template,
    render_system_prompt,
)
from src.agents import create_agent, merge_mcp_tools
from src.agents.utils import extract_replayable_messages, get_tool_callback
from src.agents.tool_injection import wrap_tools_with_dynamic_params
from .types import State


graph_name = "scenarios"
init_prompts = load_prompts_tree()


def extract_tables_from_markdown(markdown_text):
    """使用正则表达式提取Markdown表格"""
    table_pattern = r"(\|.*\|\s*\n\|[-:\s|]+\|\s*\n(?:\|.*\|\s*\n)*)"
    tables = re.findall(table_pattern, markdown_text)
    return tables


def markdown_table_to_json(md: str):
    lines = [line.strip() for line in md.strip().splitlines() if line.strip()]

    if len(lines) < 2:
        return []

    # 表头
    headers = [h.strip() for h in lines[0].strip("|").split("|")]
    headers = ["data_date" if i == "日期" else i for i in headers]

    # 数据行（跳过分隔线）
    rows = []
    for line in lines[2:]:
        values = [v.strip() for v in line.strip("|").split("|")]
        rows.append(dict(zip(headers, values)))

    return rows


def get_agent_prompt(agent_name: str) -> str:
    return f"{graph_name}/{agent_name}"


async def predict_reasoning_node(
    state: State, config: RunnableConfig
) -> Command[Literal["predict_report"]]:
    logger.info("predict reasoning node is running.")
    agent_name = "predict_reasoning"

    _, loaded_tools = await merge_mcp_tools(
        agent_name, config["configurable"]["mcp_settings"], []
    )
    # llm_token_limit = get_llm_token_limit_by_agent(agent_name)
    # pre_model_hook = partial(ContextManager(llm_token_limit, 3).compress_messages)

    # MCP的tools统一注入task_id, 内部的tools不做注入
    injection_tools = [wrap_tools_with_dynamic_params(t) for t in loaded_tools]
    agent = create_agent(
        graph_name,
        agent_name,
        injection_tools,
        # 目前与DeepResearch共同使用default的researcher
        "deepresearch/default/researcher",
        # pre_model_hook,
    )

    # Invoke the agent
    agent_input = {"messages": state["messages"]}
    logger.info(f"Agent input: {agent_input}")

    # Validate message content before invoking agent
    try:
        validated_messages = validate_message_content(agent_input["messages"])
        agent_input["messages"] = validated_messages
    except Exception as validation_error:
        logger.error(f"Error validating agent input messages: {validation_error}")

    # Apply context compression to prevent token overflow
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

    goto = "predict_report"
    agent = cast(CompiledStateGraph, agent)
    result = {}
    with get_tool_callback() as cb:
        try:
            result = await agent.ainvoke(
                input=agent_input,
                config={"recursion_limit": 50},
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
            recursion_result = await recursion_model.ainvoke([
                system_prompt,
                HumanMessage(content="请根据以上指令和消息内容，生成结构化总结报告。"),
            ])
            result["messages"] = list(agent_input.get("messages", [])) + [recursion_result]
            logger.warning(f"ReAct rollback execution succeeded: {recursion_result}")
        except Exception as e:
            return Command(
                update={
                    "messages": [
                        # TODO 优化话术
                        HumanMessage(
                            content=str(e),
                            name=agent_name,
                        )
                    ],
                },
                goto=goto,
            )

    return Command(
        update={
            "messages": result.get("messages", []),
        },
        goto=goto,
    )


async def predict_report_node(state: State, config: RunnableConfig):
    is_predict = config.get("configurable", {}).get("is_predict", False)
    predict_date = config["configurable"]["predict_date"]
    data_ids = config["configurable"]["data_ids"]
    agent_name = "predict_report"
    llm = get_llm_by_type(AGENT_LLM_MAP["predict_report"])

    # Apply context compression to prevent token overflow
    llm_token_limit = get_llm_token_limit_by_agent(agent_name)

    compressed_state = ContextManager(
        llm_token_limit, preserve_prefix_message_count=3
    ).compress_messages(state)
    messages = apply_prompt_template(
        get_agent_prompt("predict_report"),
        compressed_state,
        extra_variables={
            "is_predict": is_predict,
            "predict_date": predict_date,
            "data_ids": data_ids,
        },
    )
    response = await llm.ainvoke(messages)
    logger.debug(f"reporter response: {response.content}")

    if is_predict:
        # TODO 数据预测严格格式校验
        try:
            logger.info(
                markdown_table_to_json(
                    extract_tables_from_markdown(response.content)[-1]
                )
            )
        except Exception:
            logger.warning("predict data extra error")
        # TODO 回调业务侧存储数据

    return {"messages": response}

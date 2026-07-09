from typing import List, Optional
from uuid import uuid4

from loguru import logger
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
import json

from src.config.loader import get_bool_env, get_int_env, get_str_env
from src.graph.deepresearch.builder import build_graph_with_memory
from src.graph.deepresearch.planner_model import Plan, Step, StepType
from src.graph.utils import (
    build_clarified_topic_from_history,
    reconstruct_clarification_history,
    get_message_content,
    is_user_message,
)
from src.server.utils import ToolsInjectionInfo
from . import checkpointer_pool
from .schemas import AgentType, ChatRequest
from .utils import SSE_DONE, _stream_graph_events, _make_event


router = APIRouter(prefix="/agent/api", tags=["deepresearch"])

in_memory_store = InMemoryStore()
graph = build_graph_with_memory()


@router.post("/deepresearch/stream")
async def chat_stream(request: ChatRequest):

    # 价格预测任务需要特殊处理 MCP 设置
    if request.agent_type == AgentType.PRICE:
       request.mcp_settings = json.loads(get_str_env("MCP_PRICE_SETTINGS"))
    elif request.agent_type == AgentType.SD_BALANCE:
        request.mcp_settings = json.loads(get_str_env("MCP_SD_BALANCE_SETTINGS"))
    elif request.agent_type == AgentType.SUPPLY:
        request.mcp_settings = json.loads(get_str_env("MCP_SUPPLY_SETTINGS"))
    elif request.agent_type == AgentType.DEMAND:
        request.mcp_settings = json.loads(get_str_env("MCP_DEMAND_SETTINGS"))
    # Check if MCP server configuration is enabled
    mcp_enabled = get_bool_env("ENABLE_MCP_SERVER_CONFIGURATION", False)

    # Validate MCP settings if provided
    if request.mcp_settings and not mcp_enabled:
        raise HTTPException(
            status_code=403,
            detail="MCP server configuration is disabled. Set ENABLE_MCP_SERVER_CONFIGURATION=true to enable MCP features.",
        )

    # 处理 analysis_thread_id：从 /analysis/chart 接口继续对话
    messages = request.model_dump()["messages"]
    simple_search = request.simple_search if request.simple_search else False
    simple_deepresearch = request.simple_deepresearch if request.simple_deepresearch else False
    request.max_plan_iterations = 1
    if request.analysis_thread_id:
        # 当传入 analysis_thread_id 时，不允许重新生成模式
        request.checkpoint_id = None
        # 当传入 analysis_thread_id 时，强制开启 simple_search 模式
        simple_search = False
        simple_deepresearch = True

        # 不管 thread_id 有没有传入，直接生成新的 thread_id
        thread_id = str(uuid4())
        logger.info(f"analysis_thread_id 模式：生成新 thread_id: {thread_id}")

        # 从 analysis_thread_id 对应的 checkpoint 中获取 state['messages']
        try:
            if checkpointer_pool._checkpointer:
                # 获取最新的 checkpoint
                cp_tuple = await checkpointer_pool._checkpointer.aget_tuple(
                    {
                        "configurable": {
                            "thread_id": request.analysis_thread_id,
                        }
                    }
                )
                checkpoint_data = cp_tuple.checkpoint
                # LangGraph checkpoint 结构：messages 存储在 channel_values 中
                channel_values = checkpoint_data.get("channel_values", {})
                analysis_messages = channel_values.get("messages")
                # 如果获取到了 messages，进行转换
                if analysis_messages:
                    # 将 state 中的 messages 转换为字典格式
                    converted_messages = []
                    for msg in analysis_messages:
                        content = get_message_content(msg)
                        # 如果content存在且是str类型，才加入
                        if content and isinstance(content, str):
                            # 判断是 user 还是 assistant
                            role = "user" if is_user_message(msg) else "assistant"
                            converted_messages.append(
                                {"role": role, "content": content}
                            )
                    # 将分析接口的 messages 作为上下文，合并到当前 messages 前面
                    messages = converted_messages + messages
                    logger.info(
                        f"[{thread_id}] 从 analysis_thread_id={request.analysis_thread_id} 获取到 {len(converted_messages)} 条历史消息作为上下文"
                    )
                else:
                    logger.warning(
                        f"[{thread_id}] analysis_thread_id={request.analysis_thread_id} 对应的 checkpoint 中没有 messages"
                    )
            else:
                logger.warning(
                    f"[{thread_id}] checkpointer 未初始化，无法获取 analysis_thread_id 的历史消息"
                )
        except Exception as e:
            logger.error(
                f"[{thread_id}] 获取 analysis_thread_id={request.analysis_thread_id} 的历史消息失败: {e}"
            )
            # 即使获取失败，也继续执行，只是没有历史上下文

    # 重新生成参数处理
    checkpoint_id = request.checkpoint_id
    regenerate_mode = bool(checkpoint_id)
    if request.simple_search and request.agent_type == AgentType.REPORT and not regenerate_mode:
        simple_search = False
        simple_deepresearch = True
    # 确定thread_id（如果还没有生成）
    if not request.analysis_thread_id:
        thread_id = request.thread_id
        # 如果提供了checkpoint_id但thread_id是默认值，需要前端同时提供thread_id
        if regenerate_mode and thread_id == "__default__":
            raise HTTPException(
                status_code=400,
                detail="重试模式需要同时提供checkpoint_id和对应的thread_id",
            )
        # 如果thread_id仍然是默认值且非重新生成模式，生成新ID
        if not thread_id or thread_id == "__default__":
            thread_id = str(uuid4())
        logger.debug(f"current request thread_id: {thread_id}")

    return StreamingResponse(
        _astream_workflow_generator(
            messages,
            thread_id,
            request.max_plan_iterations,
            request.max_step_num,
            # request.max_search_results,
            request.auto_accepted_plan,
            request.interrupt_feedback,
            request.mcp_settings if mcp_enabled and request.mcp_settings else {},
            request.enable_background_investigation,
            # request.report_style,
            request.enable_clarification,
            request.max_clarification_rounds,
            request.locale,
            request.interrupt_before_tools,
            checkpoint_id,
            request.agent_type,
            simple_search,
            simple_deepresearch,
            None,
        ),
        media_type="text/event-stream",
    )


async def _astream_workflow_generator(
    messages: List[dict],
    thread_id: str,
    max_plan_iterations: int,
    max_step_num: int,
    auto_accepted_plan: bool,
    interrupt_feedback: str,
    mcp_settings: dict,
    enable_background_investigation: bool,
    enable_clarification: bool,
    max_clarification_rounds: int,
    locale: str = "zh-CN",
    interrupt_before_tools: Optional[List[str]] = None,
    checkpoint_id: Optional[str] = None,
    agent_type: Optional[AgentType] = AgentType.DEFAULT,
    simple_search: bool = False,
    simple_deepresearch: bool = False,
    simple_search_prompt: Optional[str] = None,
):
    # 重新生成模式判断
    regenerate_mode = bool(checkpoint_id)

    # In simple_search mode, restore previous messages from checkpoint into
    # conversation_history local variable instead of merging into messages.
    # This prevents LangGraph's add_messages reducer from creating duplicates
    # when astream applies workflow_input on top of checkpoint-restored state.
    # simple_researcher_node reads only from state["messages"] naturally.
    conversation_history: list[dict] = []
    # 无论 simple_search=True/False，都尝试从 checkpoint 恢复 final_report
    channel_values: dict = {}
    if not regenerate_mode and checkpointer_pool._checkpointer:
        try:
            cp_tuple = await checkpointer_pool._checkpointer.aget_tuple(
                {"configurable": {"thread_id": thread_id}}
            )
            if cp_tuple:
                checkpoint_data = cp_tuple.checkpoint
                channel_values = checkpoint_data.get("channel_values", {})
                if simple_search:
                    previous_messages = channel_values.get("messages", [])

                    if previous_messages:
                        for msg in previous_messages:
                            content = get_message_content(msg)
                            if content and isinstance(content, str):
                                role = "user" if is_user_message(msg) else "assistant"
                                conversation_history.append(
                                    {"role": role, "content": content}
                                )

                        if conversation_history:
                            logger.info(
                                f"[{thread_id}] Restored {len(conversation_history)} previous messages "
                                f"into conversation_history for simple_search mode, "
                                f"current request messages: {len(messages)}"
                            )
            # else: channel_values stays as {}
        except Exception as e:
            logger.warning(
                f"[{thread_id}] Failed to restore previous messages in simple_search mode: {e}"
            )
            channel_values = {}
            # Continue with current messages if restoration fails
            pass

    # Deduplicate messages to prevent repeated queries
    # This removes exact duplicate messages while preserving order (keeping the first occurrence)
    seen_message_contents = set()
    deduplicated_messages = []
    for msg in messages:
        if isinstance(msg, dict):
            msg_content = msg.get("content", "")
            msg_role = msg.get("role", "")
            dedup_key = (msg_role, msg_content)
            if dedup_key not in seen_message_contents:
                seen_message_contents.add(dedup_key)
                deduplicated_messages.append(msg)
            else:
                logger.debug(
                    f"[{thread_id}] Skipped duplicate message: {msg_content[:50]}..."
                )
    messages = deduplicated_messages
    effective_thread_id = thread_id

    # simple_search / simple_deepresearch 模式参数重置
    if simple_search or simple_deepresearch:
        auto_accepted_plan = True
        enable_clarification = False
        enable_background_investigation = False
        max_step_num = 2
        max_plan_iterations = 1
    if agent_type == AgentType.REPORT and max_step_num < 4:
        max_step_num = 4
    # 如果messages为空，使用空的历史记录
    if not messages or len(messages) == 0:
        clarification_history = []
        clarified_topic = ""
        latest_message_content = ""
        clarified_research_topic = ""
    else:
        clarification_history = reconstruct_clarification_history(messages)

        clarified_topic, clarification_history = build_clarified_topic_from_history(
            clarification_history
        )
        latest_message_content = messages[-1]["content"] if messages else ""
        clarified_research_topic = clarified_topic or latest_message_content

    # Prepare workflow input
    if regenerate_mode:
        # 重新生成模式：根据官方文档，恢复执行时不应该把原始输入重新传给 astream
        # 而是传 None 作为输入，让 LangGraph 从 checkpoint 恢复状态并继续执行
        workflow_input = None
    else:
        # simple_deepresearch 模式处理（老流程：planner -> researcher -> reporter）
        if simple_deepresearch:
            if simple_search_prompt:
                # simple_search_prompt 有值时，创建包含 RESEARCH 和 ANALYSIS 两个步骤的 plan
                # simple_search_prompt 的内容是预设好的计划，应该直接执行
                research_step = Step(
                    need_search=True,
                    title="简单搜索任务",
                    description=simple_search_prompt,
                    step_type=StepType.RESEARCH,
                )
                analysis_step = Step(
                    need_search=True,
                    title="继续搜索任务",
                    description=f"继续未完成的检索任务：{simple_search_prompt}",
                    step_type=StepType.RESEARCH,
                )
                simple_plan = Plan(
                    locale=locale,
                    has_enough_context=False,
                    thought="简单搜索模式：直接执行研究任务",
                    title="简单搜索计划",
                    steps=[research_step, analysis_step],
                )
                # 构建完整输入，设置 current_plan 和 goto 字段，让 coordinator 直接跳转到 researcher
                workflow_input = {
                    "messages": [{"role": "user", "content": simple_search_prompt}],
                    "plan_iterations": 0,
                    "final_report": "",
                    "current_plan": simple_plan,
                    "observations": [],
                    "auto_accepted_plan": "auto_accepted_plan",
                    "enable_background_investigation": enable_background_investigation,
                    "research_topic": "简单搜索模式：直接执行研究任务",
                    "clarification_history": [],
                    "clarified_research_topic": simple_search_prompt,
                    "enable_clarification": enable_clarification,
                    "max_clarification_rounds": max_clarification_rounds,
                    "locale": locale,
                    "goto": "researcher",  # 设置 goto 字段，让 coordinator 直接跳转到 researcher
                    "is_simple_search": False,  # 标识是 simple_search 模式
                    "simple_search_with_prompt": True,  # 标识是 simple_search_prompt 有值的情况
                    "is_simple_deepresearch": True,
                }
            else:
                workflow_input = {
                    "messages": messages,
                    "plan_iterations": 0,
                    "final_report": "",
                    "current_plan": None,
                    "observations": [],
                    "auto_accepted_plan": auto_accepted_plan,
                    "enable_background_investigation": enable_background_investigation,
                    "research_topic": latest_message_content,
                    "clarification_history": clarification_history,
                    "clarified_research_topic": clarified_research_topic,
                    "enable_clarification": enable_clarification,
                    "max_clarification_rounds": max_clarification_rounds,
                    "locale": locale,
                    "is_simple_search": False,
                    "simple_search_with_prompt": False,
                    "is_simple_deepresearch": True,
                }
        # simple_search 模式处理（新流程：simple_researcher_node）
        elif simple_search:
                # messages 仅包含前端的当前请求消息（不合并 checkpoint 历史），
                # 避免 LangGraph 的 add_messages reducer 在恢复 checkpoint 时
                # 将历史消息作为新消息追加到 state["messages"] 中造成重复。
                workflow_input = {
                    "messages": messages,
                    "conversation_history": conversation_history,
                    "plan_iterations": 0,
                    "final_report": channel_values.get("final_report", "") if channel_values else "",
                    "current_plan": None,
                    "observations": [],
                    "auto_accepted_plan": auto_accepted_plan,
                    "enable_background_investigation": enable_background_investigation,
                    "research_topic": latest_message_content,
                    "clarification_history": clarification_history,
                    "clarified_research_topic": clarified_research_topic,
                    "enable_clarification": enable_clarification,
                    "max_clarification_rounds": max_clarification_rounds,
                    "locale": locale,
                    # 不设置 goto 字段
                    "is_simple_search": True,
                    "simple_search_with_prompt": False,
                }
        else:
            # 正常模式：构建完整输入
            workflow_input = {
                "messages": messages,
                "plan_iterations": 0,
                "final_report": channel_values.get("final_report", ""),
                "current_plan": None,
                # 注释observations将会保留历史的观测结果
                "observations": [],
                "auto_accepted_plan": auto_accepted_plan,
                "enable_background_investigation": enable_background_investigation,
                "research_topic": latest_message_content,
                "clarification_history": clarification_history,
                "clarified_research_topic": clarified_research_topic,
                "enable_clarification": enable_clarification,
                "max_clarification_rounds": max_clarification_rounds,
                "locale": locale,
                "is_simple_search": False,
                "simple_search_with_prompt": False,
            }

            # 中断反馈处理（仅限正常模式）
            if not auto_accepted_plan and interrupt_feedback:
                resume_msg = f"[{interrupt_feedback}]"
                if messages:
                    resume_msg += f" {messages[-1]['content']}"
                workflow_input = Command(resume=resume_msg)

    # Prepare workflow config
    workflow_config = {
        # 保留顶层字段（兼容现有代码）
        "thread_id": effective_thread_id,
        "max_plan_iterations": max_plan_iterations,
        "max_step_num": max_step_num,
        # "max_search_results": max_search_results,
        "mcp_settings": mcp_settings,
        # "report_style": report_style.value,
        "agent_type": agent_type,
        "interrupt_before_tools": interrupt_before_tools,
        "recursion_limit": get_int_env("AGENT_DEEPRESEARCH_RECURSION_LIMIT", 200),
    }

    # 注册生成task_id的tools的任务id
    enable_tools_set = set()
    if workflow_config["mcp_settings"]:
        for _, server_config in workflow_config["mcp_settings"]["servers"].items():
            for i in server_config["enabled_tools"]:
                for j in i["tools"]:
                    enable_tools_set.add(j)
    tools_injection_info = ToolsInjectionInfo(
        enable_tools=list(enable_tools_set) if enable_tools_set else []
    )

    # 重新生成模式：验证checkpoint_id是否有效（复用同一个checkpointer）
    if regenerate_mode and thread_id != "__default__":
        configurable_dict = {
            "thread_id": effective_thread_id,
            "max_plan_iterations": max_plan_iterations,
            "max_step_num": max_step_num,
            "mcp_settings": mcp_settings,
            "interrupt_before_tools": interrupt_before_tools,
            "recursion_limit": get_int_env("AGENT_DEEPRESEARCH_RECURSION_LIMIT", 200),
            "checkpoint_id": checkpoint_id,
        }

        workflow_config["configurable"] = configurable_dict
        try:
            cp_tuple = await checkpointer_pool._checkpointer.aget_tuple(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_id": checkpoint_id,
                    }
                }
            )
            if cp_tuple:
                logger.info(
                    f"[{thread_id}] 找到checkpoint_id={checkpoint_id}，准备从该检查点恢复"
                )
            else:
                logger.warning(
                    f"[{thread_id}] 未找到checkpoint_id={checkpoint_id}的checkpoint"
                )
                # 验证失败时返回错误事件
                yield _make_event(
                    "error",
                    {
                        "thread_id": thread_id,
                        "error": f"未找到checkpoint_id={checkpoint_id}的checkpoint",
                    },
                )
                yield SSE_DONE
                return
        except Exception as e:
            logger.warning(f"[{thread_id}] 验证checkpoint失败: {e}")
            yield _make_event(
                "error",
                {
                    "thread_id": thread_id,
                    "error": f"验证checkpoint失败: {str(e)}",
                },
            )
            yield SSE_DONE
            return
    # 阶段2：未配 Postgres 时（_checkpointer 为 None）保留图自带的 MemorySaver，不覆盖
    if checkpointer_pool._checkpointer is not None:
        graph.checkpointer = checkpointer_pool._checkpointer
    graph.store = in_memory_store

    async for event in _stream_graph_events(
        graph,
        workflow_input,
        workflow_config,
        thread_id,
        tools_injection_info,
    ):
        yield event

    yield SSE_DONE

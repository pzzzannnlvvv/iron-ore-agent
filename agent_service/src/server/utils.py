import hashlib
import pickle
import copy
import asyncio
import json
from typing import Any, cast, Optional

from loguru import logger
from langchain_core.messages import AIMessageChunk, BaseMessage, ToolMessage, AIMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langchain.agents.middleware.tool_call_limit import _build_tool_message_content

from src.utils.json_utils import sanitize_args


CANCEL_KEY = "xmschain:agent:cancel"
SSE_DONE = "event: finish\ndata: [DONE]\n\n"


class ToolsInjectionInfo:
    def __init__(self, enable_tools: list[str] = []):
        self.enable_tools = enable_tools
        self.value_store = {}
        # 按工具名存储最新tool_call_id，用于流式模式回退
        # 流式模式下首帧AIMessageChunk.tool_calls的args={}，
        # 与工具实际调用时的填充args不匹配，需要按名回退查找
        self.name_store = {}

    def _make_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        key_data = pickle.dumps((func_name, args, kwargs))
        return hashlib.md5(key_data).hexdigest()

    def add(self, value: str, func_name: str, args: tuple = (), kwargs: dict = {}):
        func_key = self._make_key(func_name, args, kwargs)

        if self.enable_tools and func_name not in self.enable_tools:
            return

        self.value_store[func_key] = value
        # 始终按工具名存储最新ID（流式模式下args可能逐帧变化的回退）
        if func_name:
            self.name_store[func_name] = value

    def get(self, func_name: str, args: tuple = (), kwargs: dict = {}) -> Optional[str]:
        # 优先级1: 精确的args匹配（block模式，args完全一致）
        func_key = self._make_key(func_name, args, kwargs)
        result = self.value_store.get(func_key)
        if result:
            return result
        # 优先级2: 按工具名回退（streaming模式，首帧args={}后续帧填充）
        return self.name_store.get(func_name)

    def has(self, func_name: str, args: tuple = (), kwargs: dict = {}) -> bool:
        func_key = self._make_key(func_name, args, kwargs)
        return func_key in self.value_store


# Module-level accumulator for streaming tool call args across astream yields
# Key: (thread_id, chunk_index) -> accumulated args dict
_streaming_tool_call_accumulator: dict = {}


def _register_tool_call_ids_from_chunks(
    tool_call_chunks,
    tools_injection_info: ToolsInjectionInfo,
    thread_id: str = "",
):
    """Register tool call IDs from raw tool_call_chunks for injection matching.

    处理流式模式(streaming=True)下的工具调用ID注册。
    流式模式下tool_call chunks被分散到多个astream yield中，
    每个yield只包含args的部分字符串，需跨yield累积后才能解析为完整JSON。

    实现：
    - 使用模块级累积器按(thread_id, index)在多次调用间累积args
    - 每次调用时尝试json.loads()累积后的完整args
    - 若解析成功则注册ID并清理累积器
    - 若解析失败(args仍不完整)则等待后续yield

    注意：必须在_sanitize_args处理之前使用原始chunk数据，
    因为sanitize_args会替换[]字符导致JSON无法解析。
    """
    if not tool_call_chunks:
        return

    # 按index分组累积原始args(仅当前yield内)
    chunks_by_index = {}
    for chunk in tool_call_chunks:
        index = chunk.get("index")
        if index is None:
            continue
        if index not in chunks_by_index:
            chunks_by_index[index] = {
                "name": chunk.get("name", "") or "",
                "args": chunk.get("args", "") or "",
                "id": chunk.get("id", "") or "",
                "index": index,
            }
        else:
            existing = chunks_by_index[index]
            if chunk.get("name"):
                existing["name"] = chunk["name"]
            if chunk.get("id"):
                existing["id"] = chunk["id"]
            if chunk.get("args"):
                existing["args"] += chunk["args"]

    # 合并到跨yield累积器
    for chunk_data in chunks_by_index.values():
        index = chunk_data["index"]
        acc_key = (thread_id, index)

        if acc_key not in _streaming_tool_call_accumulator:
            _streaming_tool_call_accumulator[acc_key] = {
                "id": chunk_data["id"],
                "name": chunk_data["name"],
                "args": chunk_data["args"],
            }
        else:
            acc = _streaming_tool_call_accumulator[acc_key]
            if chunk_data["id"] and not acc["id"]:
                acc["id"] = chunk_data["id"]
            if chunk_data["name"] and not acc["name"]:
                acc["name"] = chunk_data["name"]
            if chunk_data["args"]:
                acc["args"] += chunk_data["args"]

        # 尝试解析累积后的完整args并注册
        acc = _streaming_tool_call_accumulator[acc_key]
        if acc["id"] and acc["name"] and acc["args"]:
            try:
                args_dict = json.loads(acc["args"])
                if isinstance(args_dict, dict):
                    copy_args = copy.deepcopy(args_dict)
                    copy_args.pop("task_id", None)
                    tools_injection_info.add(acc["id"], acc["name"], copy_args)
                    # 注册成功后清理累积器
                    _streaming_tool_call_accumulator.pop(acc_key, None)
            except (json.JSONDecodeError, TypeError):
                # Args仍不完整，等待后续yield
                pass


def _cleanup_streaming_accumulator(thread_id: str):
    """Clean up accumulator entries for a given thread_id."""
    keys_to_delete = [
        key for key in _streaming_tool_call_accumulator if key[0] == thread_id
    ]
    for key in keys_to_delete:
        _streaming_tool_call_accumulator.pop(key, None)


def _validate_tool_call_chunks(tool_call_chunks):
    """Validate and log tool call chunk structure for debugging."""
    if not tool_call_chunks:
        return

    indices_seen = set()
    tool_ids_seen = set()

    for _, chunk in enumerate(tool_call_chunks):
        index = chunk.get("index")
        tool_id = chunk.get("id")

        if index is not None:
            indices_seen.add(index)
        if tool_id:
            tool_ids_seen.add(tool_id)


def _process_tool_call_chunks(tool_call_chunks):
    """
    Process tool call chunks with proper index-based grouping.

    This function handles the concatenation of tool call chunks that belong
    to the same tool call (same index) while properly segregating chunks
    from different tool calls (different indices).

    The issue: In streaming, LangChain's ToolCallChunk concatenates string
    attributes (name, args) when chunks have the same index. We need to:
    1. Group chunks by index
    2. Detect index collisions with different tool names
    3. Accumulate arguments for the same index
    4. Return properly segregated tool calls
    """
    if not tool_call_chunks:
        return []

    _validate_tool_call_chunks(tool_call_chunks)

    chunks = []
    chunk_by_index = {}  # Group chunks by index to handle streaming accumulation

    for chunk in tool_call_chunks:
        index = chunk.get("index")
        chunk_id = chunk.get("id")

        if index is not None:
            # Create or update entry for this index
            if index not in chunk_by_index:
                chunk_by_index[index] = {
                    "name": "",
                    "args": "",
                    "id": chunk_id or "",
                    "index": index,
                    "type": chunk.get("type", ""),
                }

            # Validate and accumulate tool name
            chunk_name = chunk.get("name", "")
            if chunk_name:
                stored_name = chunk_by_index[index]["name"]

                # Check for index collision with different tool names
                if stored_name and stored_name != chunk_name:
                    logger.warning(
                        f"Tool name mismatch detected at index {index}: "
                        f"'{stored_name}' != '{chunk_name}'. "
                        f"This may indicate a streaming artifact or consecutive tool calls "
                        f"with the same index assignment."
                    )
                    # Keep the first name to prevent concatenation
                else:
                    chunk_by_index[index]["name"] = chunk_name

            # Update ID if new one provided
            if chunk_id and not chunk_by_index[index]["id"]:
                chunk_by_index[index]["id"] = chunk_id

            # Accumulate arguments
            if chunk.get("args"):
                chunk_by_index[index]["args"] += chunk.get("args", "")
        else:
            # Handle chunks without explicit index (edge case)
            chunks.append(
                {
                    "name": chunk.get("name", ""),
                    "args": sanitize_args(chunk.get("args", "")),
                    "id": chunk.get("id", ""),
                    "index": 0,
                    "type": chunk.get("type", ""),
                }
            )

    # Convert indexed chunks to list, sorted by index for proper order
    for index in sorted(chunk_by_index.keys()):
        chunk_data = chunk_by_index[index]
        chunk_data["args"] = sanitize_args(chunk_data["args"])
        chunks.append(chunk_data)

    return chunks


def _get_agent_name(agent, message_metadata):
    """Extract agent name from agent tuple."""
    agent_name = "unknown"
    if agent and len(agent) > 0:
        agent_name = agent[0].split(":")[0] if ":" in agent[0] else agent[0]
    else:
        agent_name = message_metadata.get("langgraph_node", "unknown")
    return agent_name


def _create_event_stream_message(
    message_chunk, message_metadata, thread_id, agent_name
):
    """Create base event stream message."""
    content = message_chunk.content
    if not isinstance(content, str):
        # ChatAnthropic 可能返回 content 列表（thinking + text 块），提取 text 部分
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            content = "".join(text_parts)
        else:
            content = json.dumps(content, ensure_ascii=False)

    event_stream_message = {
        "thread_id": thread_id,
        "agent": agent_name,
        "id": message_chunk.id,
        "role": "assistant",
        "checkpoint_id": message_metadata.get("checkpoint_id", ""),
        "checkpoint_ns": message_metadata.get("checkpoint_ns", ""),
        "langgraph_node": message_metadata.get("langgraph_node", ""),
        "langgraph_path": message_metadata.get("langgraph_path", ""),
        "langgraph_step": message_metadata.get("langgraph_step", ""),
        "content": content,
    }

    # Add optional fields
    if message_chunk.additional_kwargs.get("reasoning_content"):
        event_stream_message["reasoning_content"] = message_chunk.additional_kwargs[
            "reasoning_content"
        ]

    if message_chunk.response_metadata.get("finish_reason"):
        event_stream_message["finish_reason"] = message_chunk.response_metadata.get(
            "finish_reason"
        )

    return event_stream_message


def _make_event(event_type: str, data: dict[str, Any]):
    if data.get("content") == "":
        data.pop("content")
    # Ensure JSON serialization with proper encoding
    try:
        json_data = json.dumps(data, ensure_ascii=False)
        return f"event: {event_type}\ndata: {json_data}\n\n"
    except (TypeError, ValueError) as e:
        logger.error(f"Error serializing event data: {e}")
        # Return a safe error event
        error_data = json.dumps({"error": "Serialization failed"}, ensure_ascii=False)
        return f"event: error\ndata: {error_data}\n\n"


def _create_interrupt_event(thread_id, event_data):
    """Create interrupt event."""
    interrupt = event_data["__interrupt__"][0]
    # Use the 'id' attribute (LangGraph 1.0+) instead of deprecated 'ns[0]'
    interrupt_id = getattr(interrupt, "id", None) or thread_id
    return _make_event(
        "interrupt",
        {
            "thread_id": thread_id,
            "id": interrupt_id,
            "role": "assistant",
            "content": interrupt.value,
            "finish_reason": "interrupt",
            "options": [
                {"text": "Edit plan", "value": "edit_plan"},
                {"text": "Start research", "value": "accepted"},
            ],
        },
    )


def _create_tools_injection_event(
    thread_id, event_data: dict, agent, tools_injection_info: ToolsInjectionInfo
):
    """Create tools injection event."""
    agent_name = _get_agent_name(agent, {})
    copy_args = copy.deepcopy(event_data["args"])
    # LLM生成的task_id不可信，我们移除它在做校验
    copy_args.pop("task_id", None)

    # 优先级1: 使用CaptureToolCallIdMiddleware捕获的LLM原始tool_call ID
    # 适用于streaming模式(agent.ainvoke不传播chunks到父astream)
    id = event_data.get("llm_tool_call_id")

    # 优先级2: 使用tools_injection_info的args匹配查找
    # 适用于block模式(完整AIMessage.tool_calls到达_process_message_chunk)
    if id is None:
        id = tools_injection_info.get(event_data["name"], copy_args)

    # 优先级3: 回退到工具包装器生成的task_id(仅用于兼容)
    if id is None:
        id = event_data["args"].get("task_id")
        if id:
            logger.debug(
                f"[{thread_id}] Injection fallback: using task_id '{id}' as tool_call id "
                f"for tool '{event_data['name']}'"
            )
    return _make_event(
        "tool_calls_injection",
        {
            "thread_id": thread_id,
            "agent": agent_name,
            "role": "assistant",
            "checkpoint_ns": agent[0],
            "finish_reason": "tool_calls_injection",
            "tool_calls_injection": {
                "name": event_data["name"],
                "args": event_data["args"],
                "id": id,
                "type": event_data["type"],
            },
        },
    )


async def _stream_graph_events(
    graph_instance: CompiledStateGraph,
    workflow_input: dict,
    workflow_config: RunnableConfig,
    thread_id: str,
    tools_injection_info: ToolsInjectionInfo,
):
    """Stream events from the graph and process them."""
    event_count = 0
    sent_checkpoint_id = None  # 记录已发送过的最新 checkpoint
    try:
        async for agent, _, event_data in graph_instance.astream(
            workflow_input,
            config=workflow_config,
            stream_mode=["messages", "updates", "custom"],
            subgraphs=True,
        ):
            event_count += 1

            if isinstance(event_data, dict):
                if "__interrupt__" in event_data:
                    yield _create_interrupt_event(thread_id, event_data)
                elif "__tools_injection__" in event_data:
                    yield _create_tools_injection_event(
                        thread_id, event_data, agent, tools_injection_info
                    )
                continue

            message_chunk, message_metadata = cast(
                tuple[BaseMessage, dict[str, Any]], event_data
            )
            # 获取最新 checkpoint_tuple
            latest_cp_tuple = await graph_instance.checkpointer.aget_tuple(
                {"configurable": {"thread_id": thread_id}}
            )
            if latest_cp_tuple:
                cp_id = latest_cp_tuple.config["configurable"]["checkpoint_id"]

                # 如果这个 checkpoint_id 和上次不同，就发送一次
                if cp_id != sent_checkpoint_id:
                    sent_checkpoint_id = cp_id
                    message_metadata["checkpoint_id"] = cp_id
            async for event in _process_message_chunk(
                message_chunk, message_metadata, thread_id, agent, tools_injection_info
            ):
                yield event
    except asyncio.CancelledError:
        logger.warning(f"Graph execution cancelled,thread_id:{thread_id}")
        # Re-raise to signal cancellation properly without yielding an error event
        raise
    except Exception:
        logger.exception(f"[{thread_id}] Error during graph execution")
        yield _make_event(
            "error",
            {
                "thread_id": thread_id,
                "error": "Error during graph execution",
            },
        )
    finally:
        # 清理流式tool_call chunks累积器，防止内存泄漏
        _cleanup_streaming_accumulator(thread_id)


async def _process_message_chunk(
    message_chunk,
    message_metadata,
    thread_id,
    agent,
    tools_injection_info: ToolsInjectionInfo,
):
    """Process a single message chunk and yield appropriate events."""

    agent_name = _get_agent_name(agent, message_metadata)

    event_stream_message = _create_event_stream_message(
        message_chunk, message_metadata, thread_id, agent_name
    )

    if isinstance(message_chunk, ToolMessage):
        # Tool Message - Return the result of the tool call
        tool_call_id = message_chunk.tool_call_id
        event_stream_message["tool_call_id"] = tool_call_id
        # 这块langgraph的ToolCallLimitMiddleware设计的不好
        # 并没有预留promot参数，导致我们为了一致性，只能使用_build_tool_message_content函数
        # 此函数从API来讲并不应该被外部调用
        if isinstance(message_chunk.content, str) and message_chunk.content.startswith(
            _build_tool_message_content(None)[:10]
        ):
            logger.warning(message_chunk)
            event_stream_message["content"] = (
                "超出工具调用限制。请勿进行额外的工具调用。"
            )

        yield _make_event("tool_call_result", event_stream_message)
    elif isinstance(message_chunk, AIMessageChunk):
        # AI Message - Raw message tokens
        if message_chunk.tool_calls:
            # Warning 目前ReAct部分节点是streaming=False，触发不到本分支
            # AI Message - Tool Call (complete tool calls)
            event_stream_message["tool_calls"] = message_chunk.tool_calls
            for i in message_chunk.tool_calls:
                copy_args = copy.deepcopy(i["args"])
                copy_args.pop("task_id", None)
                tools_injection_info.add(i["id"], i["name"], copy_args)

            # Process tool_call_chunks with proper index-based grouping
            processed_chunks = _process_tool_call_chunks(message_chunk.tool_call_chunks)
            if processed_chunks:
                event_stream_message["tool_call_chunks"] = processed_chunks

            yield _make_event("tool_calls", event_stream_message)
        elif message_chunk.tool_call_chunks:
            # Warning 目前ReAct部分节点是streaming=False，触发不到本分支
            # AI Message - Tool Call Chunks (streaming)
            processed_chunks = _process_tool_call_chunks(message_chunk.tool_call_chunks)

            # 注册tool_call ID用于tool_calls_injection匹配
            # 流式模式(streaming=True)下tool_calls以chunk形式到达，不会触发上面tool_calls分支
            # 需要在_chunk层面就尝试解析并注册，确保injection事件能匹配到ID
            _register_tool_call_ids_from_chunks(
                message_chunk.tool_call_chunks, tools_injection_info, thread_id
            )

            # Emit separate events for chunks with different indices (tool call boundaries)
            if processed_chunks:
                # Include all processed chunks in the event
                event_stream_message["tool_call_chunks"] = processed_chunks
            yield _make_event("tool_call_chunks", event_stream_message)
        else:
            # AI Message - Raw message tokens
            yield _make_event("message_chunk", event_stream_message)
    elif isinstance(message_chunk, AIMessage):
        if message_chunk.tool_calls:
            event_stream_message["tool_calls"] = message_chunk.tool_calls
            for i in message_chunk.tool_calls:
                copy_args = copy.deepcopy(i["args"])
                copy_args.pop("task_id", None)
                tools_injection_info.add(i["id"], i["name"], copy_args)
            yield _make_event("tool_calls", event_stream_message)
        else:
            # AI Message - Raw message tokens
            if not event_stream_message.get("agent", None) == "planner":
                yield _make_event("message_chunk", event_stream_message)

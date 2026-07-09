from typing import List
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from langgraph.store.memory import InMemoryStore

from src.config.configuration import get_recursion_limit
from src.config.loader import get_bool_env
from src.graph.scenarios.builder import build_graph_with_memory
from src.prompts.template import env as jinja_env
from src.server.utils import ToolsInjectionInfo
from . import checkpointer_pool
from .schemas import ScenariosRequest, ChatMessage
from .utils import SSE_DONE, _stream_graph_events


in_memory_store = InMemoryStore()
graph = build_graph_with_memory()

router = APIRouter(prefix="/agent/api", tags=["scenarios"])


def fetch_scenarios_data(
    scenarios_info: dict = None, scenarios_data: dict = None, is_predict: bool = True
) -> dict:
    """获取情景数据并转换为Markdown格式的表格。

    该函数接收前端传入的情景数据和元信息，完全移除了本地文件读取逻辑。

    Args:
        scenarios_info: 前端传入的情景元信息
        scenarios_data: 前端传入的情景数据表格
        is_predict: 是否为预测模式，当为true时，scenarios_info和scenarios_data为必填

    Returns:
        dict: 情景数据表格和数据属性表
    """

    # 仅在预测模式下检查参数是否存在
    if is_predict:
        if not scenarios_info:
            raise HTTPException(
                status_code=400, detail="未提供情景元信息(scenarios_metedata)"
            )

        if not scenarios_data:
            raise HTTPException(
                status_code=400, detail="未提供情景数据(scenarios_data)"
            )

        # 转换数据格式：从{因子ID: {日期: 值}}转换为{日期: {因子ID: 值}}
        converted_data = {}
        if scenarios_data:
            # 检查数据格式
            first_key = next(iter(scenarios_data.keys()))
            # 如果第一个键对应的值是字典，且字典的值是数值类型，则判断为新格式
            if isinstance(scenarios_data[first_key], dict) and all(
                isinstance(v, (int, float)) for v in scenarios_data[first_key].values()
            ):
                # 执行转换
                for factor_id, date_values in scenarios_data.items():
                    for date, value in date_values.items():
                        if date not in converted_data:
                            converted_data[date] = {}
                        converted_data[date][factor_id] = value

                # 使用转换后的数据
                scenarios_data = converted_data
        logger.info(f"转换后的数据格式为：{scenarios_data}")

        # 对数据做频度转换
        # 从scenarios_info中获取数据频度
        data_frequency = scenarios_info.get("data_frequency", "").lower()
        if scenarios_data and data_frequency:
            # 将数据转换为DataFrame
            df = pd.DataFrame.from_dict(scenarios_data, orient="index")
            # 将索引转换为日期格式
            df.index = pd.to_datetime(df.index)
            # 填充缺失值（部分日期缺少ID，填充0）
            # df = df.fillna(0)
            # 时间索引排序（保证时间序列有序）
            df = df.sort_index()

            # 按频度聚合数据
            if data_frequency == "周度预测":
                logger.info("将数据转换为周度数据")
                # 转换为周度数据，取每周平均值（周日为每周最后一天）
                df = df.resample("W-SUN").mean().interpolate(method="linear").fillna(method="ffill").fillna(0)
            elif data_frequency == "月度预测":
                logger.info("将数据转换为月度数据")
                # 转换为月度数据，取每月平均值
                df = df.resample("ME").mean().interpolate(method="linear").fillna(method="ffill").fillna(0)
            elif data_frequency == "日度预测":
                logger.info("将数据转换为日度数据并填充缺失日期")
                # 转换为日度数据，取每日平均值
                # 使用closed='left'确保包含起始日期，label='left'使用左侧日期作为标签
                # fillna(0)填充缺失日期为0值
                df = df.resample("D").mean().interpolate(method="linear").fillna(method="ffill").fillna(0)
            # 将DataFrame转换回字典格式
            scenarios_data = df.to_dict("index")
            # 格式化日期索引为字符串
            scenarios_data = {
                date.strftime("%Y-%m-%d"): values
                for date, values in scenarios_data.items()
            }
            logger.info(f"频度转换后的数据格式为：{scenarios_data}")

        # 根据因子数量、数据频度，动态计算历史数据的周期
        # 保证 因子*日期 不超过1000个点
        if scenarios_data:
            # 从scenarios_info中获取因子数量
            if "data_info" in scenarios_info:
                factor_count = len(scenarios_info["data_info"])
            else:
                # 如果scenarios_info中没有data_info，则从scenarios_data中计算
                first_date = next(iter(scenarios_data.keys()))
                factor_count = len(scenarios_data[first_date])

            # 计算最多允许的天数：1000 / 因子数量，向下取整
            max_days = 1000 // factor_count if factor_count > 0 else 0

            # 获取所有日期并排序
            all_dates = sorted(scenarios_data.keys())
            date_count = len(all_dates)

            # 如果日期数量超过最大允许天数，则截取最后n天的数据
            if date_count > max_days and max_days > 0:
                # 截取最后max_days天的日期
                selected_dates = all_dates[-max_days:]
                # 构建新的数据字典，只包含选中的日期
                scenarios_data = {date: scenarios_data[date] for date in selected_dates}
                # 记录日志

                logger.info(f"数据量超过限制，已自动截取最后 {max_days} 天的数据")

        # 将前端传入的数据转换为pandas DataFrame
        try:
            # 创建DataFrame后进行转置，使行头为日期，列头为因子ID
            pivoted_df = pd.DataFrame(scenarios_data).T
            logger.debug(f"pivoted_df: {pivoted_df}")
            return {"info": scenarios_info, "data": pivoted_df}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"数据转换失败: {str(e)}")

    # 非预测模式下，返回空数据
    return {"info": scenarios_info or {}, "data": pd.DataFrame()}


def format_scenarios_data(scenarios_data: dict) -> str:
    """获取情景数据并转换为Markdown格式的表格。

    保留了原始data_id在数据表中，并新增数据属性表详细说明各项指标的data_id和data_name对应关系。

    Args:
        scenarios_data: 情景数据

    Returns:
        str: Markdown格式的情景数据表格和数据属性表
    """
    scenarios_info = scenarios_data["info"]
    scenarios_df = scenarios_data["data"]
    scenarios_df.index.name = "因子ID"

    # 数据属性表
    attribute_data = []
    headers = ["data_id", "因子名称", "因子类型"]
    for item in scenarios_info["data_info"]:
        data_id = str(item.get("data_id", ""))
        data_name = item.get("data_name", "")
        data_type = item.get("data_type", "")
        attribute_data.append([data_id, data_name, data_type])
    # 手动构建数据属性表的Markdown表格以减少空格和token消耗
    md_parts = []

    # 添加表头行，使用最小空格
    header_line = "|" + "|".join(headers) + "|"
    md_parts.append(header_line)

    # 添加表头分隔线
    separator_line = "|" + "|".join(["-"] * len(headers)) + "|"
    md_parts.append(separator_line)

    # 添加数据行，所有值都是字符串，直接拼接
    for row in attribute_data:
        # 构建行，使用最小空格
        row_line = "|" + "|".join(map(str, row)) + "|"
        md_parts.append(row_line)

    # 合并为完整的Markdown表格
    attribute_table = "\n".join(md_parts)

    # 历史数据
    history_data = scenarios_df.reset_index()
    # 手动构建Markdown表格以减少空格和token消耗
    md_parts = []

    # 获取表头和数据
    headers = history_data.columns.tolist()
    data_values = history_data.values

    # 添加表头行，使用最小空格
    header_line = "|" + "|".join(headers) + "|"
    md_parts.append(header_line)

    # 添加表头分隔线
    separator_line = "|" + "|".join(["-"] * len(headers)) + "|"
    md_parts.append(separator_line)

    # 添加数据行，格式化浮点数为两位小数，减少空格
    for row in data_values:
        formatted_row = []
        for val in row:
            if isinstance(val, float):
                # 格式化为两位小数，不添加额外空格
                formatted_row.append(f"{val:.2f}")
            else:
                # 其他类型转为字符串
                formatted_row.append(str(val))
        # 构建行，使用最小空格
        row_line = "|" + "|".join(formatted_row) + "|"
        md_parts.append(row_line)

    # 合并为完整的Markdown表格
    history_table = "\n".join(md_parts)

    # 统计性描述
    headers = [
        "data_id",
        # "数据量",
        # "均值",
        # "标准差",
        "最小值",
        # "25分位数",
        # "中位数",
        # "75分位数",
        "最大值",
        # "极差",
    ]
    history_desc_data = []
    for i in scenarios_df.columns:
        desc = scenarios_df[i].describe()
        history_desc_data.append(
            [
                i,
                # int(desc["count"]),
                # desc["mean"],
                # desc["std"],
                desc["min"],
                # desc["25%"],
                # desc["50%"],
                # desc["75%"],
                desc["max"],
                # desc["max"] - desc["min"],
            ]
        )
    # 手动构建统计性描述的Markdown表格以减少空格和token消耗
    md_parts = []

    # 添加表头行，使用最小空格
    header_line = "|" + "|".join(headers) + "|"
    md_parts.append(header_line)

    # 添加表头分隔线
    separator_line = "|" + "|".join(["-"] * len(headers)) + "|"
    md_parts.append(separator_line)

    # 添加数据行，格式化浮点数为两位小数
    for row in history_desc_data:
        formatted_row = []
        for val in row:
            if isinstance(val, float):
                # 格式化为两位小数，不添加额外空格
                formatted_row.append(f"{val:.2f}")
            else:
                # 其他类型转为字符串
                formatted_row.append(str(val))
        # 构建行，使用最小空格
        row_line = "|" + "|".join(formatted_row) + "|"
        md_parts.append(row_line)

    # 合并为完整的Markdown表格
    history_table_desc = "\n".join(md_parts)

    render_info = {
        "model_background": scenarios_info["model_background"],
        "scenarios_info": scenarios_info["scenarios_info"],
        "history_table": history_table,
        "history_table_desc": history_table_desc,
        "attribute_table": attribute_table,
        "predict_date": scenarios_info["predict_date"],
    }

    template = jinja_env.get_template("scenarios/init.md")
    return template.render(**render_info)


@router.post("/scenarios/stream")
async def chat_stream(request: ScenariosRequest):
    # Check if MCP server configuration is enabled
    mcp_enabled = get_bool_env("ENABLE_MCP_SERVER_CONFIGURATION", False)

    # Validate MCP settings if provided
    if request.mcp_settings and not mcp_enabled:
        raise HTTPException(
            status_code=403,
            detail="MCP server configuration is disabled. Set ENABLE_MCP_SERVER_CONFIGURATION=true to enable MCP features.",
        )

    thread_id = request.thread_id
    if not thread_id or thread_id == "__default__":
        thread_id = str(uuid4())
    logger.debug(f"current request thread_id: {thread_id}")

    # 从请求中获取scenarios_info和scenarios_data参数并传递给fetch_scenarios_data
    scenarios_data = fetch_scenarios_data(
        scenarios_info=request.scenarios_metedata,
        scenarios_data=request.scenarios_data,
        is_predict=request.is_predict,
    )

    if request.is_predict:
        default_message = format_scenarios_data(scenarios_data)
        request.messages = [ChatMessage(role="user", content=default_message)]

    return StreamingResponse(
        _astream_workflow_generator(
            request.model_dump()["messages"],
            thread_id,
            request.mcp_settings if mcp_enabled and request.mcp_settings else {},
            request.is_predict,
            [i["data_id"] for i in scenarios_data["info"]["data_info"]],
            scenarios_data["info"]["predict_date"],
        ),
        media_type="text/event-stream",
    )


async def _astream_workflow_generator(
    messages: List[dict],
    thread_id: str,
    mcp_settings: dict,
    is_predict: bool,
    data_ids: list,
    predict_date: list,
):
    workflow_input = {
        "messages": messages,
    }

    # Prepare workflow config
    workflow_config = {
        "thread_id": thread_id,
        "is_predict": is_predict,
        "recursion_limit": get_recursion_limit(default=100),
        "mcp_settings": mcp_settings,
        "data_ids": data_ids,
        "predict_date": predict_date,
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

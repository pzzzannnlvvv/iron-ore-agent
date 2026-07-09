import asyncio
import json
from datetime import datetime
from typing import Optional
from uuid import uuid4

from loguru import logger
from fastapi import APIRouter, HTTPException
from src.config.loader import get_str_env
from fastapi.responses import StreamingResponse
from src.config.loader import get_bool_env
from src.server.deepresearch import _astream_workflow_generator
from .schemas import (
    AgentType,
    SynthesisRequest,
    SynthesisResponse,
    ChartRequest,
    ComplexTableData,
    ChartData,
    ColumnStoreData,
    RowStoreTableData,
)

router = APIRouter(prefix="/agent/api/analysis", tags=["analysis"])


def _parse_sse_event(event_str: str) -> Optional[dict]:
    """解析SSE事件字符串"""
    lines = event_str.strip().split("\n")
    event_type = None
    data = None

    for line in lines:
        if line.startswith("event: "):
            event_type = line[7:].strip()
        elif line.startswith("data: "):
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                return {"type": "done"}
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                logger.warning(f"无法解析SSE数据: {data_str}")
                continue

    if event_type and data is not None:
        return {"type": event_type, "data": data}
    return None


def _format_table_data_to_markdown(table_data: any, predict_num: Optional[int] = None) -> str:
    """将表格数据转换为Markdown格式，支持多种数据格式"""
    md_parts = []

    # 处理第一种格式：新的ChartData格式
    if isinstance(table_data, ChartData) and table_data.extraList is not None:
        logger.info("第1种格式")
        # 检查数据是否为空
        if not table_data.dataList:
            raise ValueError("数据列表为空")
        
        # 表格标题 - 安全获取，避免不存在属性或空值
        if hasattr(table_data, 'tableName') and table_data.tableName:
            md_parts.append(f"## {table_data.tableName}")
        else:
            md_parts.append("## 表格数据")
        
        # 添加频度信息（如果有）
        if hasattr(table_data, 'frequencyName') and table_data.frequencyName:
            md_parts.append(f"- **频度**: {table_data.frequencyName}")
        
        # 添加预测步长信息（如果有）
        if predict_num is not None:
            md_parts.append(f"- **预测步长**: {predict_num}")

        # 从主数据的items中提取单位信息（如果有）
        unit = ""
        if table_data.items:
            for item in table_data.items:
                if hasattr(item, 'unit') and item.unit:
                    unit = item.unit
                    break
        
        # 准备从额外数据系列中提取指标信息（如果有）
        extra_series = table_data.extraList if table_data.extraList else []
        
        md_parts.append("")
        
        # 动态获取所有指标名称，并添加单位
        all_index_names = []
        index_series_map = []  # 保存指标名称到系列的映射
        
        # 遍历所有额外系列，收集所有指标
        if extra_series:
            for series in extra_series:
                if series.items and len(series.items) > 0:
                    # 使用每个系列自己的第一个item作为指标名称
                    item = series.items[0]
                    
                    if unit:
                        # 将单位拼接到指标名称后面
                        indexed_name_with_unit = f"{item.itemName} ({unit})"
                    else:
                        indexed_name_with_unit = item.itemName
                    
                    all_index_names.append(indexed_name_with_unit)
                    index_series_map.append(series)
        
        # 生成表格头
        # 基础列：时间、所有指标（带单位）、差值（带单位）、类型
        diff_column = f"差值 ({unit})" if unit else "差值"
        header_columns = ["时间"] + all_index_names + [diff_column, "类型"]
        md_parts.append("| " + " | ".join(header_columns) + " |")
        
        # 生成表头分隔线
        separator_columns = ["------"] + ["--------" for _ in all_index_names] + ["--------", "----------"]
        md_parts.append("| " + " | ".join(separator_columns) + " |")
        
        # 处理主数据列表 - 按日期倒序排列
        # 定义日期解析和比较函数
        def sort_key(data_point):
            try:
                # 尝试解析日期格式
                return datetime.strptime(data_point.dataMonth, "%Y-%m-%d")
            except ValueError:
                # 如果解析失败，返回一个非常早的日期
                return datetime.min
        
        # 按日期倒序排序数据点
        sorted_data_points = sorted(table_data.dataList, key=sort_key, reverse=True)
        
        for data_point in sorted_data_points:
            date = data_point.dataMonth
            
            # 获取所有指标的值（从对应的系列中获取）
            all_index_values = []
            for i, series in enumerate(index_series_map):
                index_value = "-"
                for data_point_in_series in series.dataList:
                    if data_point_in_series.dataMonth == date:
                        # 根据当前数据类型选择合适的字段值
                        if hasattr(data_point, 'his') and data_point.his is not None:
                            # 历史值：优先使用his字段
                            if hasattr(data_point_in_series, 'his') and data_point_in_series.his is not None:
                                index_value = data_point_in_series.his
                                break
                        elif hasattr(data_point, 'ai') and data_point.ai is not None:
                            # 预测值：优先使用ai字段，如果没有则使用his字段
                            if hasattr(data_point_in_series, 'ai') and data_point_in_series.ai is not None:
                                index_value = data_point_in_series.ai
                                break
                            elif hasattr(data_point_in_series, 'his') and data_point_in_series.his is not None:
                                index_value = data_point_in_series.his
                                break
                all_index_values.append(str(index_value) if index_value != "-" else "-")
            
            # 获取差值（从主数据中获取）
            difference = "-"
            if hasattr(data_point, 'his') and data_point.his is not None:
                difference = data_point.his
            elif hasattr(data_point, 'ai') and data_point.ai is not None:
                difference = data_point.ai
            
            # 确定数据类型
            data_type = "-"
            if hasattr(data_point, 'his') and data_point.his is not None:
                data_type = "历史值"
            elif hasattr(data_point, 'ai') and data_point.ai is not None:
                data_type = "预测值"
            
            # 构建行数据
            row_columns = [date] + all_index_values + [str(difference), data_type]
            md_parts.append("| " + " | ".join(row_columns) + " |")
        
        md_parts.append("")
        return "\n".join(md_parts)
    
    # 处理第二种格式：带表头的表格数据
    if isinstance(table_data, ComplexTableData):
        # 检查数据是否为空
        if not table_data.dataList:
            raise HTTPException(status_code=400, detail="未提供表格数据")
        
        # 确定表格标题
        table_title = "表格信息"  # 默认标题
        if hasattr(table_data, 'tableName') and table_data.tableName:
            table_title = table_data.tableName
        
        # 添加频度信息（如果有）
        if hasattr(table_data, 'frequencyName') and table_data.frequencyName:
            md_parts.append(f"- **频度**: {table_data.frequencyName}")
        
        # 添加预测步长信息（如果有）
        if predict_num is not None:
            md_parts.append(f"- **预测步长**: {predict_num}")

        # 表格标题
        md_parts.append(f"## {table_title}")
        md_parts.append("")
        
        # 准备信息列表
        info_items = []
        # 从数据中提取单位信息（如果有）
        first_entry = table_data.dataList[0]
        if hasattr(first_entry, 'unit') and first_entry.unit:
            info_items.append(f"- **单位**: {first_entry.unit}")
        
        # 获取所有表头信息
        headers = [head.title for head in table_data.head if head.show]
        header_keys = [head.keyId for head in table_data.head if head.show]
        
        # 生成表格头
        header_line = "| " + " | ".join(headers) + " |"
        separator_line = "| " + " | ".join(["---" for _ in headers]) + " |"
        
        md_parts.append(header_line)
        md_parts.append(separator_line)
        
        # 生成数据行
        for data_entry in table_data.dataList:
            row_values = []
            for key in header_keys:
                # 同时支持对象属性访问和字典键访问
                if hasattr(data_entry, '__dict__'):  # 是对象
                    value = getattr(data_entry, key, "-")
                elif isinstance(data_entry, dict):  # 是字典
                    value = data_entry.get(key, "-")
                else:  # 其他类型
                    value = "-"
            
                # 特殊处理标题字段：拼接单位信息
                if (key == "targetName" or key == "title"):
                    # 获取单位信息
                    unit = ""
                    if hasattr(data_entry, '__dict__'):
                        unit = getattr(data_entry, "unit", "")
                    elif isinstance(data_entry, dict):
                        unit = data_entry.get("unit", "")
                    
                    if unit:
                        value = f"{value} ({unit})"
    
                # 检查是否需要格式化百分比
                for head in table_data.head:
                    if head.keyId == key and head.percentage and isinstance(value, (int, float)):
                        value = f"{value}%"
                        break
                row_values.append(str(value))
            row_line = "| " + " | ".join(row_values) + " |"
            md_parts.append(row_line)
        
        
        md_parts.append("")

        return "\n".join(md_parts)
    
    # 处理第三种格式：列存储数据
    if isinstance(table_data, list) and all(isinstance(item, ColumnStoreData) for item in table_data):
        logger.info("第3种格式")
        # 提取数据
        data_list = table_data[0].data.dataList
        if not data_list:
            raise HTTPException(status_code=400, detail="未提供表格数据")
        
        # 添加表格描述信息
        if hasattr(table_data, 'moduleName') and table_data.moduleName: 
            md_parts.append(f"- **表格名称**: {table_data.moduleName}")
        # 添加频度信息（如果有）
        if hasattr(table_data, 'frequencyName') and table_data.frequencyName:
            md_parts.append(f"- **频度**: {table_data.frequencyName}")
        
        # 添加预测步长信息（如果有）
        if predict_num is not None:
            md_parts.append(f"- **预测步长**: {predict_num}")
        
        # 尝试提取单位信息（从第一个非日期列）
        for column in data_list:
            if column.get("id", -1) > 0 and column.get("unit", ""):
                md_parts.append(f"- **单位**: {column['unit']}")
                break
        
        
        md_parts.append("")
        
        # 准备表格数据
        # 按列ID排序数据列表
        sorted_columns = sorted(data_list, key=lambda x: x.get("id", 0))
        
        # 提取表头
        headers = []
        for column in sorted_columns:
            if column.get("id", 0) == 0:
                # 第一列通常是日期列
                headers.append("日期")
            else:
                # 构建列名，包含单位（如果有）
                if column.get("indexName", ""):
                    # 使用指标名称作为基础
                    base_name = column["indexName"]
                else:
                    # 使用value作为基础
                    base_name = column.get("value", f"列{column.get('id', 0)}")
                
                # 添加单位信息（如果有）
                unit = column.get("unit", "").strip()
                if unit:
                    header_name = f"{base_name} ({unit})"
                else:
                    header_name = base_name
                
                headers.append(header_name)
        
        # 生成表格头
        header_line = "| " + " | ".join(headers) + " |"
        separator_line = "| " + " | ".join(["---" for _ in headers]) + " |"
        md_parts.append(header_line)
        md_parts.append(separator_line)
        
        # 转置数据：从列存储转为行存储
        num_rows = len(sorted_columns[0].get("data", [])) if sorted_columns else 0
        for row_idx in range(num_rows):
            row_values = []
            for column in sorted_columns:
                column_data = column.get("data", [])
                if row_idx < len(column_data):
                    row_values.append(str(column_data[row_idx]))
                else:
                    row_values.append("-")
            row_line = "| " + " | ".join(row_values) + " |"
            md_parts.append(row_line)
        
        md_parts.append("")
        return "\n".join(md_parts)
    
    # 处理第四种格式：行存储表格数据
    if isinstance(table_data, list) and all(isinstance(item, RowStoreTableData) for item in table_data):
        logger.info("第4种格式")
        # 提取数据
        data_json = table_data[0].data.dataJson
        if not data_json:
            raise HTTPException(status_code=400, detail="未提供表格数据")
        
        if hasattr(table_data, 'moduleName') and table_data.moduleName: 
            md_parts.append(f"- **表格名称**: {table_data.moduleName}")
        # 添加频度信息（如果有）
        if hasattr(table_data, 'frequencyName') and table_data.frequencyName:
            md_parts.append(f"- **频度**: {table_data.frequencyName}")
        
        # 添加预测步长信息（如果有）
        if predict_num is not None:
            md_parts.append(f"- **预测步长**: {predict_num}")
        
        # 生成表格
        for row_idx, row in enumerate(data_json):
            # 提取当前行的所有文本值
            row_texts = [cell.get("text", "-") for cell in row]
            
            # 生成表格行
            row_line = "| " + " | ".join(row_texts) + " |"
            
            # 如果是第一行，添加表头分隔线
            if row_idx == 0:
                md_parts.append(row_line)
                # 生成分隔线
                separator_line = "| " + " | ".join(["---" for _ in row_texts]) + " |"
                md_parts.append(separator_line)
            else:
                md_parts.append(row_line)
        
        md_parts.append("")
        return "\n".join(md_parts)
    
    # 处理第五种格式：带历史值和预测值的表格数据
    if isinstance(table_data, ChartData) and table_data.extraList is None:
        logger.info("第5种格式")
        # 检查数据是否为空
        if not table_data.dataList:
            raise ValueError("数据列表为空")
        
        item_name = getattr(table_data, 'itemName', '指标')

        # 表格标题 - 使用itemName作为指标名称
        md_parts.append(f"## {item_name}")
        
        # 从items中提取单位信息
        unit = ""
        if table_data.items and len(table_data.items) > 0:
            first_item = table_data.items[0]
            if hasattr(first_item, 'unit') and first_item.unit:
                unit = first_item.unit
        
        # 如果有单位，添加单位信息
        if unit:
            md_parts.append(f"- **单位**: {unit}")
        
        md_parts.append("")
        
        # 生成表格头
        value_header = f"{item_name} ({unit})"
        headers = ["日期", value_header, "类型"]
        md_parts.append("| " + " | ".join(headers) + " |")
        
        # 生成表头分隔线
        separator_columns = ["------", "--------", "----------"]
        md_parts.append("| " + " | ".join(separator_columns) + " |")
        
        # 处理数据列表 - 按日期倒序排列
        def sort_key(data_point):
            try:
                return datetime.strptime(data_point.dataMonth, "%Y-%m-%d")
            except ValueError:
                return datetime.min
        
        sorted_data_points = sorted(table_data.dataList, key=sort_key, reverse=True)
        
        for data_point in sorted_data_points:
            date = data_point.dataMonth
            
            # 处理历史值和预测值，优先使用历史值
            if hasattr(data_point, 'his') and data_point.his is not None:
                # 构建行数据
                row_columns = [date, str(data_point.his), "实际值"]
                md_parts.append("| " + " | ".join(row_columns) + " |")
            elif hasattr(data_point, 'ai') and data_point.ai is not None:
                # 构建行数据
                row_columns = [date, str(data_point.ai), "预测值"]
                md_parts.append("| " + " | ".join(row_columns) + " |")
        
        md_parts.append("")
        return "\n".join(md_parts)
    # 默认情况：返回原始数据的字符串表示
    return f"```json\n{table_data}\n```"


@router.post("/synthesis")
async def synthesis(request: SynthesisRequest):
    """汇总分析接口 - 阻塞模式，只返回reporter节点的content"""
    # 检查MCP服务器配置是否启用
    mcp_enabled = get_bool_env("ENABLE_MCP_SERVER_CONFIGURATION", False)

    # 验证MCP设置
    if request.mcp_settings and not mcp_enabled:
        raise HTTPException(
            status_code=403,
            detail="MCP服务器配置已禁用。设置 ENABLE_MCP_SERVER_CONFIGURATION=true 以启用MCP功能。",
        )

    # 确定thread_id
    thread_id = request.thread_id
    if not thread_id or thread_id == "__default__":
        thread_id = str(uuid4())
    logger.debug(f"当前请求 thread_id: {thread_id}")

    # 准备参数
    messages = request.model_dump()["messages"]

    if request.simple_search_prompt:
        messages.append({"role": "user", "content": request.simple_search_prompt})
    else:
        raise HTTPException(status_code=400, detail="simple_search_prompt不能为空")

    mcp_settings = request.mcp_settings if mcp_enabled and request.mcp_settings else json.loads(get_str_env("MCP_SYNTHESIS_SETTINGS"))

    # 调用流式生成器，收集所有事件
    reporter_content = ""
    buffer = ""

    try:
        async for event_chunk in _astream_workflow_generator(
            messages,
            thread_id,
            max_plan_iterations=1,  # simple_search模式固定为1
            max_step_num=1,  # simple_search模式固定为1
            auto_accepted_plan=True,  # simple_search模式固定为True
            interrupt_feedback=None,
            mcp_settings=mcp_settings,
            enable_background_investigation=False,  # simple_search模式固定为False
            enable_clarification=False,  # simple_search模式固定为False
            max_clarification_rounds=0,
            locale=request.locale or "zh-CN",
            interrupt_before_tools=None,
            checkpoint_id=None,
            agent_type=AgentType.DEFAULT,
            simple_search=False,
            simple_deepresearch=True,
            simple_search_prompt="",
        ):
            # 累积事件块
            buffer += event_chunk

            # 尝试解析完整的SSE事件（以\n\n结尾）
            while "\n\n" in buffer:
                event_str, buffer = buffer.split("\n\n", 1)
                parsed_event = _parse_sse_event(event_str)

                if parsed_event:
                    event_type = parsed_event.get("type")
                    event_data = parsed_event.get("data", {})

                    # 检查是否是reporter节点的message_chunk事件
                    if event_type == "message_chunk":
                        agent = event_data.get("agent", "")
                        content = event_data.get("content", "")

                        if agent == "reporter" and content:
                            # 累积reporter节点的内容
                            reporter_content += content

                    # 检查是否完成
                    elif event_type == "done" or event_type == "finish":
                        break

        # 如果buffer中还有未处理的数据，尝试解析
        if buffer.strip():
            parsed_event = _parse_sse_event(buffer)
            if parsed_event:
                event_type = parsed_event.get("type")
                event_data = parsed_event.get("data", {})

                if event_type == "message_chunk":
                    agent = event_data.get("agent", "")
                    content = event_data.get("content", "")

                    if agent == "reporter" and content:
                        reporter_content += content

    except asyncio.CancelledError:
        logger.warning(f"[{thread_id}] 请求被取消")
        raise HTTPException(status_code=499, detail="请求被取消")
    except Exception as e:
        logger.exception(f"[{thread_id}] 处理请求时发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"处理请求时发生错误: {str(e)}")

    if not reporter_content:
        logger.warning(f"[{thread_id}] 未找到reporter节点的内容")
        raise HTTPException(
            status_code=404, detail="未找到reporter节点的内容，可能工作流未正常完成"
        )

    return SynthesisResponse(content=reporter_content, thread_id=thread_id)


@router.post("/synthesis/stream")
async def synthesis(request: SynthesisRequest):
    """汇总分析接口 - 阻塞模式，只返回reporter节点的content"""
    # 检查MCP服务器配置是否启用
    mcp_enabled = get_bool_env("ENABLE_MCP_SERVER_CONFIGURATION", False)

    # 验证MCP设置
    if request.mcp_settings and not mcp_enabled:
        raise HTTPException(
            status_code=403,
            detail="MCP服务器配置已禁用。设置 ENABLE_MCP_SERVER_CONFIGURATION=true 以启用MCP功能。",
        )

    # 确定thread_id
    thread_id = request.thread_id
    if not thread_id or thread_id == "__default__":
        thread_id = str(uuid4())
    logger.debug(f"当前请求 thread_id: {thread_id}")

    # 准备参数
    messages = request.model_dump()["messages"]

    if request.simple_search_prompt:
        messages.append({"role": "user", "content": request.simple_search_prompt})
    else:
        raise HTTPException(status_code=400, detail="simple_search_prompt不能为空")

    mcp_settings = request.mcp_settings if mcp_enabled and request.mcp_settings else json.loads(get_str_env("MCP_SYNTHESIS_SETTINGS"))

    return StreamingResponse(
        _astream_workflow_generator(
            messages,
            thread_id,
            max_plan_iterations=1,  # simple_search模式固定为1
            max_step_num=1,  # simple_search模式固定为1
            auto_accepted_plan=True,  # simple_search模式固定为True
            interrupt_feedback=None,
            mcp_settings=mcp_settings,
            enable_background_investigation=False,  # simple_search模式固定为False
            enable_clarification=False,  # simple_search模式固定为False
            max_clarification_rounds=0,
            locale=request.locale or "zh-CN",
            interrupt_before_tools=None,
            checkpoint_id=None,
            agent_type=AgentType.DEFAULT,
            simple_search=False,
            simple_deepresearch=True,
            simple_search_prompt="",
        ))

@router.post("/chart")
async def chart_analysis(request: ChartRequest):
    """图表分析接口 - 阻塞模式，只返回reporter节点的content"""
    # 检查MCP服务器配置是否启用
    mcp_enabled = get_bool_env("ENABLE_MCP_SERVER_CONFIGURATION", False)

    # 验证MCP设置
    if request.mcp_settings and not mcp_enabled:
        raise HTTPException(
            status_code=403,
            detail="MCP服务器配置已禁用。设置 ENABLE_MCP_SERVER_CONFIGURATION=true 以启用MCP功能。",
        )

    # 确定thread_id
    thread_id = request.thread_id
    if not thread_id or thread_id == "__default__":
        thread_id = str(uuid4())
    logger.debug(f"当前请求 thread_id: {thread_id}")

    # 准备参数
    messages = request.model_dump()["messages"]
    mcp_settings = request.mcp_settings if mcp_enabled and request.mcp_settings else json.loads(get_str_env("MCP_CHART_SETTINGS"))

    # 将table_data转换为Markdown格式
    table_markdown = _format_table_data_to_markdown(request.table_data,request.predict_num)
    # 组装simple_search_prompt：将table_data的markdown拼接到simple_search_prompt之后
    simple_search_prompt = request.simple_search_prompt or ""
    # 添加表格描述
    if request.table_describe:
        simple_search_prompt = f"{simple_search_prompt}\n\n**表格描述：** {request.table_describe}"

    if table_markdown:
        if simple_search_prompt:
            simple_search_prompt = f"{simple_search_prompt}\n\n{table_markdown}"
        else:
            simple_search_prompt = table_markdown

    # 调用流式生成器，收集所有事件
    reporter_content = ""
    buffer = ""

    try:
        async for event_chunk in _astream_workflow_generator(
            messages,
            thread_id,
            max_plan_iterations=1,  # simple_search模式固定为1
            max_step_num=1,  # simple_search模式固定为1
            auto_accepted_plan=True,  # simple_search模式固定为True
            interrupt_feedback=None,
            mcp_settings=mcp_settings,
            enable_background_investigation=False,  # simple_search模式固定为False
            enable_clarification=False,  # simple_search模式固定为False
            max_clarification_rounds=0,
            locale=request.locale or "zh-CN",
            interrupt_before_tools=None,
            checkpoint_id=None,
            agent_type=AgentType.DEFAULT,
            simple_search=False,
            simple_deepresearch=True,
            simple_search_prompt=simple_search_prompt,
        ):
            # 累积事件块
            buffer += event_chunk

            # 尝试解析完整的SSE事件（以\n\n结尾）
            while "\n\n" in buffer:
                event_str, buffer = buffer.split("\n\n", 1)
                parsed_event = _parse_sse_event(event_str)

                if parsed_event:
                    event_type = parsed_event.get("type")
                    event_data = parsed_event.get("data", {})

                    # 检查是否是reporter节点的message_chunk事件
                    if event_type == "message_chunk":
                        agent = event_data.get("agent", "")
                        content = event_data.get("content", "")

                        if agent == "reporter" and content:
                            # 累积reporter节点的内容
                            reporter_content += content

                    # 检查是否完成
                    elif event_type == "done" or event_type == "finish":
                        break

        # 如果buffer中还有未处理的数据，尝试解析
        if buffer.strip():
            parsed_event = _parse_sse_event(buffer)
            if parsed_event:
                event_type = parsed_event.get("type")
                event_data = parsed_event.get("data", {})

                if event_type == "message_chunk":
                    agent = event_data.get("agent", "")
                    content = event_data.get("content", "")

                    if agent == "reporter" and content:
                        reporter_content += content

    except asyncio.CancelledError:
        logger.warning(f"[{thread_id}] 请求被取消")
        raise HTTPException(status_code=499, detail="请求被取消")
    except Exception as e:
        logger.exception(f"[{thread_id}] 处理请求时发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"处理请求时发生错误: {str(e)}")

    if not reporter_content:
        logger.warning(f"[{thread_id}] 未找到reporter节点的内容")
        raise HTTPException(
            status_code=404, detail="未找到reporter节点的内容，可能工作流未正常完成"
        )

    return SynthesisResponse(content=reporter_content, thread_id=thread_id)


@router.post("/chart/stream")
async def chart_analysis(request: ChartRequest):
    """图表分析接口 - 阻塞模式，只返回reporter节点的content"""
    # 检查MCP服务器配置是否启用
    mcp_enabled = get_bool_env("ENABLE_MCP_SERVER_CONFIGURATION", False)

    # 验证MCP设置
    if request.mcp_settings and not mcp_enabled:
        raise HTTPException(
            status_code=403,
            detail="MCP服务器配置已禁用。设置 ENABLE_MCP_SERVER_CONFIGURATION=true 以启用MCP功能。",
        )

    # 确定thread_id
    thread_id = request.thread_id
    if not thread_id or thread_id == "__default__":
        thread_id = str(uuid4())
    logger.debug(f"当前请求 thread_id: {thread_id}")

    # 准备参数
    messages = request.model_dump()["messages"]
    mcp_settings = request.mcp_settings if mcp_enabled and request.mcp_settings else json.loads(get_str_env("MCP_CHART_SETTINGS"))

    # 将table_data转换为Markdown格式
    table_markdown = _format_table_data_to_markdown(request.table_data,request.predict_num)
    # 组装simple_search_prompt：将table_data的markdown拼接到simple_search_prompt之后
    simple_search_prompt = request.simple_search_prompt or ""
    # 添加表格描述
    if request.table_describe:
        simple_search_prompt = f"{simple_search_prompt}\n\n**表格描述：** {request.table_describe}"

    if table_markdown:
        if simple_search_prompt:
            simple_search_prompt = f"{simple_search_prompt}\n\n{table_markdown}"
        else:
            simple_search_prompt = table_markdown

    return StreamingResponse(
        _astream_workflow_generator(
        messages,
        thread_id,
        max_plan_iterations=1,  # simple_search模式固定为1
        max_step_num=1,  # simple_search模式固定为1
        auto_accepted_plan=True,  # simple_search模式固定为True
        interrupt_feedback=None,
        mcp_settings=mcp_settings,
        enable_background_investigation=False,  # simple_search模式固定为False
        enable_clarification=False,  # simple_search模式固定为False
        max_clarification_rounds=0,
        locale=request.locale or "zh-CN",
        interrupt_before_tools=None,
        checkpoint_id=None,
        agent_type=AgentType.DEFAULT,
        simple_search=False,
        simple_deepresearch=True,
        simple_search_prompt=simple_search_prompt,
    ))
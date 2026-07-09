import json
from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, Field, field_validator, FieldValidationInfo

from src.config.loader import get_str_env


class ReportStyle(Enum):
    ACADEMIC = "academic"


class AgentType(Enum):
    DEFAULT = "default"  # 默认
    SUPPLY = "supply"  # 供应
    DEMAND = "demand"  # 需求
    SD_BALANCE = "sd_balance"  # 供需平衡
    PRICE = "price"  # 价格预测
    REPORT = "report"  # 研报


class ContentItem(BaseModel):
    type: str = Field(..., description="The type of content (text, image, etc.)")
    text: Optional[str] = Field(None, description="The text content if type is 'text'")
    image_url: Optional[str] = Field(
        None, description="The image URL if type is 'image'"
    )


class ChatMessage(BaseModel):
    role: str = Field(
        ..., description="The role of the message sender (user or assistant)"
    )
    content: Union[str, List[ContentItem]] = Field(
        ...,
        description="The content of the message, either a string or a list of content items",
    )


class ChatRequest(BaseModel):
    messages: Optional[List[ChatMessage]] = Field(
        [], description="History of messages between the user and the assistant"
    )
    thread_id: Optional[str] = Field(
        "__default__", description="A specific conversation identifier"
    )
    locale: Optional[str] = Field(
        "zh-CN", description="Language locale for the conversation (e.g., en-US, zh-CN)"
    )
    max_plan_iterations: Optional[int] = Field(
        1, description="The maximum number of plan iterations"
    )
    max_step_num: Optional[int] = Field(
        4, description="The maximum number of steps in a plan"
    )
    # max_search_results: Optional[int] = Field(
    #     3, description="The maximum number of search results"
    # )
    auto_accepted_plan: Optional[bool] = Field(
        False, description="Whether to automatically accept the plan"
    )
    interrupt_feedback: Optional[str] = Field(
        None, description="Interrupt feedback from the user on the plan"
    )
    mcp_settings: Optional[dict] = Field(
        json.loads(get_str_env("MCP_DEFAULT_SETTINGS")),
        description="MCP settings for the chat request",
    )
    enable_background_investigation: Optional[bool] = Field(
        True, description="Whether to get background investigation before plan"
    )
    report_style: Optional[ReportStyle] = Field(
        ReportStyle.ACADEMIC, description="The style of the report"
    )
    enable_clarification: Optional[bool] = Field(
        None,
        description="Whether to enable multi-turn clarification (default: None, uses State default=False)",
    )
    max_clarification_rounds: Optional[int] = Field(
        None,
        description="Maximum number of clarification rounds (default: None, uses State default=3)",
    )
    interrupt_before_tools: List[str] = Field(
        default_factory=list,
        description="List of tool names to interrupt before execution (e.g., ['db_tool', 'api_tool'])",
    )
    checkpoint_id: Optional[str] = Field(
        None,
        description="SSE事件中返回的checkpoint_id字段，用于从该检查点重新生成。如果提供此字段，将进入重新生成模式。",
    )
    agent_type: Optional[AgentType] = Field(
        AgentType.DEFAULT, description="DeepResearch的agent类型"
    )
    simple_search: Optional[bool] = Field(
        False, description="启用简单搜索模式，直接使用 simple_researcher_node 独立完成研究"
    )
    simple_deepresearch: Optional[bool] = Field(
        False, description="启用简单深度研究模式，走原来的 planner -> researcher -> reporter 流程"
    )
    analysis_thread_id: Optional[str] = Field(
        None,
        description="analysis/chart接口返回的thread_id，用于继续对话，传入时会自动开启simple_search模式",
    )


class ChatCancel(BaseModel):
    thread_id: str = Field(description="A specific conversation identifier")


class ScenariosRequest(BaseModel):
    messages: Optional[List[ChatMessage]] = Field(
        [], description="History of messages between the user and the assistant"
    )
    thread_id: Optional[str] = Field(
        "__default__", description="A specific conversation identifier"
    )
    mcp_settings: Optional[dict] = Field(
        json.loads(get_str_env("MCP_DEFAULT_SETTINGS")),
        description="MCP settings for the chat request",
    )
    
    is_predict: bool = Field(
        True,
        description="",
    )
    scenarios_metedata: Optional[dict] = Field(
        None, description="传入的情景元信息"
    )
    scenarios_data: Optional[dict] = Field(
        None, description="传入的情景数据表格"
    )
    # 校验scenarios_metedata和scenarios_data是否为空或非空
    @field_validator('scenarios_metedata')
    def check_scenarios_metedata_required(cls, v, info: FieldValidationInfo):
        if info.data.get('is_predict') and v is None:
            raise ValueError("当 is_predict 为 true 时，scenarios_metedata 字段为必填项")
        return v

    @field_validator('scenarios_data')
    def check_scenarios_data_required(cls, v, info: FieldValidationInfo):
        if info.data.get('is_predict') and v is None:
            raise ValueError("当 is_predict 为 true 时，scenarios_data 字段为必填项")
        return v
class SynthesisRequest(BaseModel):
    """汇总分析请求模型"""
    messages: Optional[List[ChatMessage]] = Field(
        [], description="历史消息列表"
    )
    thread_id: Optional[str] = Field(
        "__default__", description="会话标识符"
    )
    locale: Optional[str] = Field(
        "zh-CN", description="语言区域"
    )
    simple_search: Optional[bool] = Field(
        True, description="是否启用简单搜索模式"
    )
    simple_search_prompt: Optional[str] = Field(
        None, description="简单搜索模式的提示词"
    )
    mcp_settings: Optional[dict] = Field(
        None, description="MCP设置"
    )

class SynthesisResponse(BaseModel):
    """汇总分析响应模型"""
    content: str = Field(..., description="reporter节点的内容")
    thread_id: str = Field(..., description="会话标识符")


class DataPoint(BaseModel):
    """数据点模型"""
    dataDate: str = Field(..., description="数据日期")
    value: str = Field(..., description="数据值")


# 指标项模型（用于新的第一种格式）
class ChartItem(BaseModel):
    """图表指标项模型"""
    itemName: str = Field(..., description="指标名称")
    id: str = Field(..., description="指标ID")
    lineType: int = Field(..., description="线条类型")
    unit: str = Field(..., description="单位")

# 历史数据项模型（用于新的第一种格式）
class ChartDataPoint(BaseModel):
    """图表数据点模型"""
    dataMonth: str = Field(..., description="数据月份/日期")
    # 动态字段，根据items中的id定义
    
    class Config:
        extra = "allow"

# 额外数据系列模型（用于新的第一种格式）
class ChartExtraSeries(BaseModel):
    """图表额外数据系列模型"""
    items: List[ChartItem] = Field(..., description="指标项列表")
    dataList: List[ChartDataPoint] = Field(..., description="数据点列表")
    extraList: Optional[List[dict]] = Field(None, description="额外列表")
    rise: Optional[dict] = Field(None, description="上涨相关信息")
    down: Optional[dict] = Field(None, description="下跌相关信息")

# 新的第一种格式：图表数据模型
class ChartData(BaseModel):
    """图表数据模型"""
    items: List[ChartItem] = Field(..., description="指标项列表")
    dataList: List[ChartDataPoint] = Field(..., description="数据点列表")
    extraList: Optional[List[ChartExtraSeries]] = Field(None, description="额外数据系列列表")
    rise: Optional[dict] = Field(None, description="上涨相关信息")
    down: Optional[dict] = Field(None, description="下跌相关信息")
    itemName: Optional[str] = Field(None, description="指标名称")

# 第三种格式：列存储数据模型
class ColumnStoreData(BaseModel):
    """第三种数据格式：列存储数据模型"""
    viewId: int = Field(..., description="视图ID")
    moduleType: int = Field(..., description="模块类型")
    moduleName: str = Field(..., description="模块名称")
    
    class Data(BaseModel):
        """核心数据模型"""
        dataList: List[dict] = Field(..., description="数据列表")
        titleList: Optional[List[dict]] = Field(None, description="标题列表")
    
    data: Data = Field(..., description="核心数据")

# 列数据项模型（用于第三种格式）
class ColumnDataItem(BaseModel):
    """列数据项模型"""
    unit: str = Field(..., description="单位")
    data: List[Union[str, float, int]] = Field(..., description="该列的所有数据值")
    id: int = Field(..., description="列ID")
    value: str = Field(..., description="列标识值")
    indexName: Optional[str] = Field(None, description="指标名称")
    indexId: Optional[str] = Field(None, description="指标ID")

# 表头信息模型（用于第三种格式）
class ColumnTitleItem(BaseModel):
    """表头信息模型"""
    id: int = Field(..., description="列ID")
    value: str = Field(..., description="列标题")
    indexId: Optional[str] = Field(None, description="指标ID")

# 第四种格式：行存储表格数据模型
class RowStoreTableData(BaseModel):
    """第四种数据格式：行存储表格数据模型"""
    viewId: int = Field(..., description="视图ID")
    moduleType: int = Field(..., description="模块类型")
    moduleName: str = Field(..., description="模块名称")
    
    class Data(BaseModel):
        """核心数据模型"""
        dataJson: List[List[dict]] = Field(..., description="数据列表")
    
    data: Data = Field(..., description="核心数据")

# 表格单元格模型（用于第四种格式）
class TableCell(BaseModel):
    """表格单元格模型"""
    colspan: int = Field(..., description="合并列数")
    id: str = Field(..., description="单元格ID")
    rowspan: int = Field(..., description="合并行数")
    text: str = Field(..., description="单元格文本内容")
    time: Optional[str] = Field(None, description="时间信息（如果有）")

# 旧的第一种格式（为了兼容性暂时保留）
class HistoricalDataItem(BaseModel):
    """历史数据项模型"""
    dataMonth: str = Field(..., description="数据月份/日期")
    his: Optional[float] = Field(None, description="历史数据值")
    ai: Optional[Union[float, str]] = Field(None, description="AI预测数据值")


# 表格头模型（第二种格式）
class TableHead(BaseModel):
    """表格头模型"""
    title: str = Field(..., description="列标题")
    keyId: str = Field(..., description="列的键ID")
    show: bool = Field(..., description="是否显示该列")
    percentage: bool = Field(..., description="是否为百分比值")


# 复杂表格数据模型（第二种格式）
class ComplexTableData(BaseModel):
    """复杂表格数据模型"""
    head: List[TableHead] = Field(..., description="表格头")
    dataList: List[object] = Field(..., description="数据列表")
    tableName: Optional[str] = Field(None, description="表格名称")
    frequencyName: Optional[str] = Field(None, description="频度名称")
    tableId: Optional[int] = Field(None, description="表格ID")


class TableDataItem(BaseModel):
    """表格数据项模型"""
    indexName: str = Field(..., description="预测标的名称")
    forecastFrequency: str = Field(..., description="预测频度")
    forecastHorizon: int = Field(..., description="预测步长")
    yearOverYearChange: float = Field(..., description="同比变化")
    monthOverMonthChange: float = Field(..., description="环比变化")
    forecastSeries: List[DataPoint] = Field(..., description="预测序列数据")
    historicalSeries: List[DataPoint] = Field(..., description="历史序列数据")


class ChartRequest(BaseModel):
    """图表分析请求模型"""
    messages: Optional[List[ChatMessage]] = Field(
        [], description="历史消息列表"
    )
    thread_id: Optional[str] = Field(
        "__default__", description="会话标识符"
    )
    locale: Optional[str] = Field(
        "zh-CN", description="语言区域"
    )
    predict_num: Optional[int] = Field(
        None, description="预测数据的步长"
    )
    simple_search: Optional[bool] = Field(
        True, description="是否启用简单搜索模式"
    )
    table_describe: Optional[str] = Field(
        None, description="表格描述"
    )
    simple_search_prompt: Optional[str] = Field(
        None, description="简单搜索模式的提示词"
    )
    table_data: Union[List[HistoricalDataItem], ComplexTableData, ChartData, List[ColumnStoreData], List[RowStoreTableData]] = Field(
        ..., description="表格数据，支持多种格式"
    )
    mcp_settings: Optional[dict] = Field(
        None, description="MCP设置"
    )


class SuggestedRequest(BaseModel):
    """问题推荐请求模型"""
    prompt: str = Field(..., description="用户输入的提示词")


class SuggestedResponse(BaseModel):
    """问题推荐响应模型"""
    questions: List[str] = Field(..., description="推荐的问题列表，最多三个")

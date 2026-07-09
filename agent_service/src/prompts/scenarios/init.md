帮我基于研究数据，进行数据分析。

# 注意事项

* 选择合适的工具进行知识检索以辅助你进行数据探索
* 寻找各因子与情景概览直接潜在的影响关系

# 研究数据

## 1. 建模背景

{{ model_background }}

## 2. 情景概览

**情景信息**：{{ scenarios_info.name }}

**情景描述**：
{% for item in scenarios_info.desc %}
- {{ loop.index }}. {{ item }}
{% endfor %}

## 3. 数据

### 3.1 数据属性

{{ attribute_table }}

### 3.2 历史数据

{{ history_table }}

### 3.3 历史数据统计描述

{{ history_table_desc }}

### 3.4 说明

- 以上数据为历史指标数据
- 列名中的方括号内数字为预测模型的因子ID
- 忽略上下文中markdown格式的**图、超链接**内容，不要引用它们，也不要输出它们，`![chart_tool](UUId)`仅为示例，不要输出。

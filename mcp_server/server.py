"""
复现2 自建 MCP server
桥接 agent（researcher/analyst/background 节点）与复现2 的建模产出。
端口 17000，streamable_http，路径 /mcp。

启动：uv run python server.py
"""
import json
from pathlib import Path

import pandas as pd
from mcp.server.fastmcp import FastMCP

# 复现2 根目录 = mcp_server 的上一级
ROOT = Path(__file__).resolve().parent.parent
OUT_07 = ROOT / "07_model_training" / "outputs"
OUT_05 = ROOT / "05_feature_engineering" / "outputs"
OUT_MULTI = ROOT / "multi_target" / "outputs"

mcp = FastMCP("iron-ore-reproduction-mcp", host="127.0.0.1", port=17000)


def _read_text(p: Path) -> str:
    """安全读取文件文本，不存在则返回提示。"""
    if not p.exists():
        return f"（未找到文件 {p.name}）"
    try:
        return p.read_text(encoding="utf-8")
    except Exception as e:
        return f"（读取 {p.name} 失败: {e}）"


def _read_csv_tail(p: Path, n: int = 10) -> str:
    """读取 CSV 最后 n 行的字符串表示。"""
    if not p.exists():
        return f"（未找到文件 {p.name}）"
    try:
        df = pd.read_csv(p)
        tail = df.tail(n).to_string(index=False)
        return f"列: {list(df.columns)}\n最近 {n} 行:\n{tail}"
    except Exception as e:
        return f"（读取 {p.name} 失败: {e}）"


@mcp.tool()
def fetch_data(query: str) -> str:
    """查询铁矿石库存预测的核心数据：标的、模型评估指标、测试集预测结果、特征体系。
    当需要预测数值、MDA/MAPE 等评估指标、库存数据、特征列表时调用本工具。

    Args:
        query: 自然语言查询，例如"铁矿石库存预测结果""模型准确率""最近预测"
    """
    parts = []
    parts.append("【标的】ID00186052 铁矿石：进口：库存：45个港口（周度）—— 全项目参考基准主标的")

    parts.append("【模型评估 final_evaluation_weekly.json】\n" + _read_text(OUT_07 / "final_evaluation_weekly.json"))
    parts.append("【最优超参 best_params_weekly.json】\n" + _read_text(OUT_07 / "best_params_weekly.json"))
    parts.append("【测试集预测 test_predictions_weekly.csv】\n" + _read_csv_tail(OUT_07 / "test_predictions_weekly.csv", 10))

    feat_path = OUT_05 / "final_feature_list_weekly.csv"
    if feat_path.exists():
        try:
            feats = pd.read_csv(feat_path)
            n = len(feats)
            sample = list(feats.iloc[:, 0])[:20] if n > 0 else []
            parts.append(f"【特征体系 final_feature_list_weekly.csv】共 {n} 个特征，前20个示例: {sample}")
        except Exception as e:
            parts.append(f"【特征体系】读取失败: {e}")
    else:
        parts.append("【特征体系】未找到 final_feature_list_weekly.csv")

    return "\n\n".join(parts)


@mcp.tool()
def fetch_background(query: str) -> str:
    """铁矿石库存预测项目的背景资料：数据时间范围、标的说明、建模流程。
    当需要项目背景、数据说明、建模方法概述时调用本工具。

    Args:
        query: 自然语言查询，例如"项目背景""数据范围""建模流程"
    """
    return """【项目背景】铁矿石库存预测项目（项目复现2）

- 标的：ID00186052 铁矿石：进口：库存：45个港口（周度），作为全项目参考基准主标的
- 数据时间范围：2015-12-25 至 2026-05-08，周频
- 训练集：2015-12-25 ~ 2024-12-31（470 行）
- Gap 隔离带：14 周
- 测试集：2025-04-11 ~ 2026-05-08（56 行）
- 建模流程：数据准备 → 三种预处理 → 数据清洗校验 → 平稳性检验(ADF) → 特征工程(11步) → 数据分割 → 模型训练(LightGBM + Optuna 200次超参优化)
- 训练目标：一阶差分 diff（预测下期库存变化量），原因：ADF 确认原始序列非平稳，差分后平稳，MDA 直接优化方向
- 评估指标：MDA（方向准确率）为主，MAPE、MAE 辅助
- 多目标框架：26 个铁矿石库存相关标的（地区维度/品种维度/港口维度）
- 主标的测试集 MDA = 84.91%
"""


@mcp.tool()
def fetch_news(query: str) -> str:
    """新闻舆情 / 市场动态 / 政策。本项目为纯量化建模，不含新闻数据。
    如需预测数据请改用 fetch_data，模型评估请用 model_choice。

    Args:
        query: 自然语言查询
    """
    return ("本项目（项目复现2）为铁矿石库存量化预测，不含新闻舆情/市场动态数据。"
            "预测结果与模型评估请调用 fetch_data；模型选择与评估详情请调用 model_choice；"
            "项目背景请调用 fetch_background。")


@mcp.tool()
def fetch_graph(query: str) -> str:
    """知识图谱查询。本项目无图谱数据，请改用 fetch_data。

    Args:
        query: 自然语言查询
    """
    return "本项目无知识图谱数据。请使用 fetch_data 获取预测与特征数据。"


@mcp.tool()
def model_choice(query: str) -> str:
    """模型选择与评估详情：最优超参、训练/测试指标、优化摘要、过拟合判断。
    当需要模型性能、超参组合、是否过拟合、Optuna 优化结果时调用本工具。

    Args:
        query: 自然语言查询，例如"模型评估""超参""过拟合"
    """
    parts = []
    parts.append("【final_evaluation_weekly.json】\n" + _read_text(OUT_07 / "final_evaluation_weekly.json"))
    parts.append("【best_params_weekly.json】\n" + _read_text(OUT_07 / "best_params_weekly.json"))
    parts.append("【optimization_summary_weekly.json】\n" + _read_text(OUT_07 / "optimization_summary_weekly.json"))
    parts.append(
        "【指标说明】\n"
        "- MDA（方向准确率）：预测方向与实际方向一致的比例，越高越好。test_mda=0.8491 即 84.91%。\n"
        "- MAPE（平均绝对百分比误差）：预测值偏离实际值的百分比，越低越好。\n"
        "- MAE（平均绝对误差）：预测值与实际值的平均绝对差，单位万吨。\n"
        "- 过拟合判断：train_mda(99.56%) 与 test_mda(84.91%) 差距 14.65%，存在一定过拟合。"
    )
    return "\n\n".join(parts)


@mcp.tool()
def model_predict(query: str) -> str:
    """模型预测结果。预测数据详见 test_predictions_weekly.csv（也可通过 fetch_data 获取）。

    Args:
        query: 自然语言查询，例如"预测结果""最近预测"
    """
    return "【预测结果 test_predictions_weekly.csv】\n" + _read_csv_tail(OUT_07 / "test_predictions_weekly.csv", 12)


@mcp.tool()
def draw_chart(query: str) -> str:
    """绘制图表。当前 MCP server 版本暂不支持绘图，请在报告中用 markdown 表格描述数据。

    Args:
        query: 自然语言描述要绘制的图表
    """
    return ("当前 MCP server 暂不支持绘图能力。请在报告中使用 markdown 表格描述预测数据，"
            "或引用 weekly_model_results.html 中的可视化。")


if __name__ == "__main__":
    print("MCP server 启动：127.0.0.1:17000/mcp (streamable-http)")
    mcp.run(transport="streamable-http")

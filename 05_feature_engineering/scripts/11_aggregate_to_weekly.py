"""
Step 11: 将日频特征聚合为周频

聚合策略：取每周最后一个交易日（优先周五）

输出:
- outputs/09_features_final_weekly.csv
- outputs/final_feature_list_weekly.csv
- outputs/weekly_aggregation_summary.html
"""

import csv
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = PROJECT_ROOT / "05_feature_engineering/outputs/09_features_final.csv"
FEATURE_LIST_PATH = PROJECT_ROOT / "05_feature_engineering/outputs/final_feature_list.csv"
OUTPUT_DIR = PROJECT_ROOT / "05_feature_engineering/outputs"


def read_dual_header_csv(path):
    """读取双表头 CSV，返回 header 行和 data 行列表"""
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header1 = next(reader)
        header2 = next(reader)
        rows = list(reader)

    return header1, header2, rows


def get_week_key(dt):
    """获取日期所属的周的key: (year, week_number)"""
    # 使用ISO周定义
    isocal = dt.isocalendar()
    return (isocal.year, isocal.week)


def is_friday(dt):
    """判断是否为周五"""
    return dt.weekday() == 4


def find_weekly_end_dates(dates):
    """
    为每个周找到最后一个交易日

    策略：
    - 每周内找最后一个日期
    - 如果该周有周五，则优先选周五
    """
    # 先按日期排序
    sorted_dates = sorted(dates)

    # 构建周->日期列表的映射
    week_to_dates = {}
    for d_str in sorted_dates:
        dt = datetime.strptime(d_str, "%Y-%m-%d")
        week_key = get_week_key(dt)
        if week_key not in week_to_dates:
            week_to_dates[week_key] = []
        week_to_dates[week_key].append((dt, d_str))

    # 为每个周选最后一个日期
    selected_dates = []
    for week_key in sorted(week_to_dates.keys()):
        dates_in_week = week_to_dates[week_key]

        # 优先选周五
        fridays = [(dt, d_str) for dt, d_str in dates_in_week if is_friday(dt)]
        if fridays:
            selected_dates.append(fridays[-1][1])
        else:
            # 没有周五，选本周最后一个日期
            selected_dates.append(dates_in_week[-1][1])

    return selected_dates


def regenerate_time_features(date_str):
    """为给定日期重新生成时间特征"""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    isocal = dt.isocalendar()

    time_features = {}

    # 基础时间特征
    time_features["year"] = dt.year
    time_features["month"] = dt.month
    time_features["day"] = dt.day
    time_features["quarter"] = (dt.month - 1) // 3 + 1
    time_features["dayofweek"] = dt.weekday()
    time_features["dayofyear"] = int(dt.strftime("%j"))
    time_features["weekofyear"] = isocal.week

    # 月度哑变量
    for m in range(1, 12):
        time_features[f"month_{m}"] = 1 if dt.month == m else 0

    # 季度哑变量
    for q in range(1, 4):
        time_features[f"quarter_{q}"] = 1 if time_features["quarter"] == q else 0

    # 正弦余弦编码
    time_features["month_sin"] = np.sin(2 * np.pi * dt.month / 12)
    time_features["month_cos"] = np.cos(2 * np.pi * dt.month / 12)
    time_features["quarter_sin"] = np.sin(2 * np.pi * time_features["quarter"] / 4)
    time_features["quarter_cos"] = np.cos(2 * np.pi * time_features["quarter"] / 4)

    # 春节标记（简化版）
    # 这里用简化策略：原数据中的春节标记已经计算好，直接用最后一天的值即可
    # 所以这部分在下面聚合时直接取最后一天的原始值

    return time_features


def aggregate_to_weekly():
    print("=" * 70)
    print("Step 11: 将日频特征聚合为周频")
    print("=" * 70)

    # 读取数据
    print("\n[1/4] 读取日频特征数据...")
    header_ids, header_names, rows = read_dual_header_csv(INPUT_PATH)

    date_idx = header_ids.index("data_date")
    print(f"  原始数据: {len(rows)} 行, {len(header_ids)} 列")

    # 读取特征列表
    print("\n[2/4] 读取特征类型列表...")
    feature_types = {}
    with open(FEATURE_LIST_PATH, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            feature_types[row["列ID"]] = row["类型"]
    print(f"  已加载 {len(feature_types)} 个特征的类型")

    # 找到每个周的最后一个交易日
    print("\n[3/4] 识别每周最后一个交易日...")
    all_dates = [row[date_idx] for row in rows]
    selected_dates = find_weekly_end_dates(all_dates)
    print(f"  原始: {len(all_dates)} 个日期 → 周频: {len(selected_dates)} 个日期")
    print(f"  聚合比例: {len(selected_dates)/len(all_dates):.1%}")

    # 构建日期 -> 行的映射
    date_to_row = {row[date_idx]: row for row in rows}

    # 聚合
    print("\n[4/4] 执行聚合...")
    weekly_rows = []

    for selected_date in selected_dates:
        original_row = date_to_row[selected_date]
        dt = datetime.strptime(selected_date, "%Y-%m-%d")

        new_row = []
        for col_idx, col_id in enumerate(header_ids):
            if col_id == "data_date":
                # 日期列直接使用选中的日期
                new_row.append(selected_date)
            elif col_id in feature_types and feature_types[col_id] == "time":
                # 时间特征：重新生成
                regenerated = regenerate_time_features(selected_date)
                if col_id in regenerated:
                    new_row.append(str(regenerated[col_id]))
                else:
                    # 其他时间特征（春节标记等）直接取原始值
                    new_row.append(original_row[col_idx])
            else:
                # 其他特征：直接取原始值
                new_row.append(original_row[col_idx])

        weekly_rows.append(new_row)

    print(f"  聚合完成: {len(weekly_rows)} 行")

    # 保存
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 保存聚合后的特征
    output_path = OUTPUT_DIR / "09_features_final_weekly.csv"
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header_ids)
        writer.writerow(header_names)
        writer.writerows(weekly_rows)
    print(f"\n已保存: {output_path} ({os.path.getsize(output_path)/1024/1024:.1f} MB)")

    # 保存特征列表
    feature_list_path = OUTPUT_DIR / "final_feature_list_weekly.csv"
    with open(feature_list_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["列ID", "列名", "类型"])
        for col_id, col_name in zip(header_ids, header_names):
            if col_id == "data_date":
                continue
            ftype = feature_types.get(col_id, "factor")
            writer.writerow([col_id, col_name, ftype])
    print(f"已保存: {feature_list_path}")

    # 生成汇总报告
    generate_summary_html(selected_dates, len(rows), len(weekly_rows))

    print("\n" + "=" * 70)
    print("周频聚合完成！")
    print("=" * 70)


def generate_summary_html(selected_dates, original_rows, weekly_rows):
    """生成聚合汇总HTML报告"""
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>周频聚合汇总报告</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 40px; background: #f5f7fa; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }}
        h1 {{ color: #1a202c; margin-bottom: 30px; }}
        .stat-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 30px 0; }}
        .stat-card {{ background: #f7fafc; padding: 24px; border-radius: 8px; text-align: center; border: 1px solid #e2e8f0; }}
        .stat-value {{ font-size: 36px; font-weight: 700; color: #3182ce; margin-bottom: 8px; }}
        .stat-label {{ color: #718096; font-size: 14px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px 16px; border-bottom: 1px solid #e2e8f0; text-align: left; }}
        th {{ background: #2d3748; color: white; font-weight: 600; }}
        .info {{ background: #ebf8ff; border-left: 4px solid #3182ce; padding: 16px; margin: 20px 0; border-radius: 0 8px 8px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>周频聚合汇总报告</h1>

        <div class="info">
            <strong>聚合策略:</strong> 取每周最后一个交易日（优先周五）
        </div>

        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-value">{original_rows}</div>
                <div class="stat-label">原始日频数据行数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{weekly_rows}</div>
                <div class="stat-label">周频数据行数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{weekly_rows/original_rows:.1%}</div>
                <div class="stat-label">压缩比例</div>
            </div>
        </div>

        <h3>日期范围</h3>
        <table>
            <thead>
                <tr>
                    <th>数据集</th>
                    <th>起始日期</th>
                    <th>结束日期</th>
                    <th>数据点数</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>日频</td>
                    <td>-</td>
                    <td>-</td>
                    <td>{original_rows}</td>
                </tr>
                <tr>
                    <td>周频</td>
                    <td>{selected_dates[0]}</td>
                    <td>{selected_dates[-1]}</td>
                    <td>{weekly_rows}</td>
                </tr>
            </tbody>
        </table>

        <h3>聚合说明</h3>
        <ul>
            <li><strong>数值列:</strong> 取每周最后一个交易日的值</li>
            <li><strong>时间特征:</strong> 为周频日期重新生成</li>
            <li><strong>目标列:</strong> 取每周最后一天的值</li>
        </ul>

        <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #e2e8f0; color: #a0aec0; text-align: center;">
            生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        </div>
    </div>
</body>
</html>
"""

    summary_path = OUTPUT_DIR / "weekly_aggregation_summary.html"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"已保存: {summary_path}")


if __name__ == "__main__":
    aggregate_to_weekly()


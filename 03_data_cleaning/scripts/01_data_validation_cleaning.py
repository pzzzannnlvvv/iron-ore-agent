import pandas as pd
import numpy as np
import os
import csv

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
input_path = os.path.join(base_dir, '01_data_preparation', 'outputs', 'merged_factors_daily_aligned.csv')
output_dir = os.path.join(base_dir, '04_feature_engineering', 'outputs')
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("特征前轻量校验与清洗")
print("=" * 80)

# 读取数据
print("\n读取数据...")
df = pd.read_csv(input_path, header=[0, 1], skiprows=0, encoding='utf-8-sig')
date_col = df.iloc[:, 0]
values_df = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

col_ids = list(df.columns.get_level_values(0))[1:]
col_names = list(df.columns.get_level_values(1))[1:]

print(f"数据形状: {df.shape}")
print(f"日期范围: {date_col.iloc[0]} 至 {date_col.iloc[-1]}")

# ==========================================
# 校验与清洗
# ==========================================
print("\n开始校验与清洗...")

cleaned_values = values_df.copy()
validation_report = []

for col_idx in range(values_df.shape[1]):
    col_id = col_ids[col_idx]
    col_name = col_names[col_idx]
    col_data = values_df.iloc[:, col_idx]

    # 统计原始数据
    original_valid = col_data.notna().sum()
    original_mean = col_data.mean()
    original_min = col_data.min()
    original_max = col_data.max()

    # 标记问题
    issues = []
    fixes = []

    # 1. 业务逻辑校验：库存数据非负
    # 判断是否是库存相关列
    is_inventory = any(keyword in col_name for keyword in ['库存', '库存:', '库存：'])

    if is_inventory:
        neg_count = (col_data < 0).sum()
        if neg_count > 0:
            issues.append(f"存在负值: {neg_count}个")
            # 将负值设为NaN
            cleaned_values.iloc[:, col_idx] = cleaned_values.iloc[:, col_idx].mask(lambda x: x < 0)
            fixes.append(f"负值设为缺失: {neg_count}个")

    # 2. 极端异常值检测：IQR方法
    # 只对有足够数据的列做
    valid_data = col_data.dropna()
    if len(valid_data) > 100:
        q1 = valid_data.quantile(0.25)
        q3 = valid_data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr

        # 检测异常值
        outliers_low = (col_data < lower_bound).sum()
        outliers_high = (col_data > upper_bound).sum()

        if outliers_low > 0:
            issues.append(f"下界异常: {outliers_low}个 (< {lower_bound:.2f})")
        if outliers_high > 0:
            issues.append(f"上界异常: {outliers_high}个 (> {upper_bound:.2f})")

        # 修正极端异常值（用上下界截断）
        if outliers_low + outliers_high > 0:
            cleaned_values.iloc[:, col_idx] = cleaned_values.iloc[:, col_idx].clip(lower=lower_bound, upper=upper_bound)
            fixes.append(f"截断极端值: {outliers_low + outliers_high}个")

    # 3. 检测明显的跳变：单日变化超过5倍标准差
    if len(valid_data) > 100:
        col_diff = col_data.diff()
        diff_std = col_diff.std()
        if pd.notna(diff_std) and diff_std > 0:
            big_jumps = (col_diff.abs() > 5 * diff_std).sum()
            if big_jumps > 0:
                issues.append(f"异常跳变: {big_jumps}次 (>5σ)")
                # 对跳变点设为NaN（保守处理）
                cleaned_values.iloc[:, col_idx] = cleaned_values.iloc[:, col_idx].mask(col_diff.abs() > 5 * diff_std)
                fixes.append(f"跳变点设为缺失: {big_jumps}个")

    # 统计清洗后
    cleaned_data = cleaned_values.iloc[:, col_idx]
    cleaned_valid = cleaned_data.notna().sum()

    validation_report.append({
        'col_id': col_id,
        'col_name': col_name,
        'is_inventory': is_inventory,
        'original_valid': original_valid,
        'original_mean': original_mean,
        'original_min': original_min,
        'original_max': original_max,
        'cleaned_valid': cleaned_valid,
        'issues': '; '.join(issues) if issues else '无',
        'fixes': '; '.join(fixes) if fixes else '无'
    })

    if col_idx % 50 == 0:
        print(f"  已处理 {col_idx}/{values_df.shape[1]} 列")

# ==========================================
# 保存清洗后的数据
# ==========================================
print("\n保存清洗后的数据...")

output_rows = []
output_rows.append(['data_date'] + col_ids)
output_rows.append(['data_date'] + col_names)

for i in range(len(date_col)):
    row = [date_col.iloc[i]]
    for j in range(cleaned_values.shape[1]):
        val = cleaned_values.iloc[i, j]
        if pd.isna(val):
            row.append('')
        else:
            row.append(f"{val:.6f}")
    output_rows.append(row)

cleaned_path = os.path.join(output_dir, 'cleaned_data_daily.csv')
with open(cleaned_path, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(output_rows)

# ==========================================
# 保存校验报告
# ==========================================
report_path = os.path.join(output_dir, 'validation_report.csv')
with open(report_path, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        '列ID', '列名', '是否库存列', '原始有效样本', '清洗后有效样本',
        '原始均值', '原始最小值', '原始最大值', '发现问题', '修复操作'
    ])
    for r in validation_report:
        writer.writerow([
            r['col_id'],
            r['col_name'],
            '是' if r['is_inventory'] else '否',
            r['original_valid'],
            r['cleaned_valid'],
            r['original_mean'],
            r['original_min'],
            r['original_max'],
            r['issues'],
            r['fixes']
        ])

# ==========================================
# 统计摘要
# ==========================================
total_issues = sum(1 for r in validation_report if r['issues'] != '无')
inventory_cols = sum(1 for r in validation_report if r['is_inventory'])

print("\n" + "=" * 80)
print("清洗完成摘要")
print("=" * 80)
print(f"总列数: {len(validation_report)}")
print(f"库存相关列: {inventory_cols}")
print(f"发现问题的列: {total_issues}")
print(f"\n数据已保存: {os.path.basename(cleaned_path)}")
print(f"报告已保存: {os.path.basename(report_path)}")

# 生成HTML报告
print("\n生成HTML报告...")

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>数据校验与清洗报告</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 40px;
            background-color: #f5f7fa;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        }}
        h1 {{
            color: #1a365d;
            border-bottom: 3px solid #3182ce;
            padding-bottom: 15px;
        }}
        .summary-box {{
            background: linear-gradient(135deg, #ebf8ff 0%, #e6fffa 100%);
            padding: 25px;
            border-radius: 10px;
            margin: 20px 0;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin: 20px 0;
        }}
        .summary-card {{
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .summary-card h4 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            color: #718096;
        }}
        .summary-card .number {{
            font-size: 36px;
            font-weight: 700;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 12px;
        }}
        th, td {{
            border: 1px solid #e2e8f0;
            padding: 8px 10px;
            text-align: left;
        }}
        th {{
            background-color: #2d3748;
            color: white;
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background-color: #f7fafc;
        }}
        tr:hover {{
            background-color: #ebf8ff;
        }}
        .tag {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 11px;
            font-weight: 600;
        }}
        .tag-issue {{
            background-color: #fed7d7;
            color: #742a2a;
        }}
        .tag-ok {{
            background-color: #c6f6d5;
            color: #22543d;
        }}
        .tag-inventory {{
            background-color: #e6fffa;
            color: #234e52;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🛡️ 数据校验与清洗报告</h1>

        <div class="summary-box">
            <h3>概览</h3>
            <p><strong>数据时间范围:</strong> {date_col.iloc[0]} 至 {date_col.iloc[-1]}</p>
            <p><strong>总列数:</strong> {len(validation_report)}</p>
        </div>

        <div class="summary-grid">
            <div class="summary-card" style="background: #ebf8ff;">
                <h4>总列数</h4>
                <div class="number" style="color: #3182ce;">{len(validation_report)}</div>
            </div>
            <div class="summary-card" style="background: #e6fffa;">
                <h4>库存相关列</h4>
                <div class="number" style="color: #319795;">{inventory_cols}</div>
            </div>
            <div class="summary-card" style="background: #fed7d7;">
                <h4>发现问题的列</h4>
                <div class="number" style="color: #c53030;">{total_issues}</div>
            </div>
            <div class="summary-card" style="background: #c6f6d5;">
                <h4>正常列</h4>
                <div class="number" style="color: #276749;">{len(validation_report) - total_issues}</div>
            </div>
        </div>

        <h3>📋 校验内容</h3>
        <ul>
            <li>✅ <strong>业务逻辑校验:</strong> 库存数据检查非负</li>
            <li>✅ <strong>极端异常值:</strong> IQR方法检测（±3×IQR）</li>
            <li>✅ <strong>异常跳变:</strong> 单日变化超过5倍标准差</li>
        </ul>

        <h3>📊 详细报告</h3>
        <table>
            <thead>
                <tr>
                    <th>列ID</th>
                    <th>列名</th>
                    <th>类型</th>
                    <th>原始有效</th>
                    <th>清洗后有效</th>
                    <th>原始统计</th>
                    <th>发现问题</th>
                    <th>修复操作</th>
                </tr>
            </thead>
            <tbody>
"""

for r in validation_report:
    issue_tag = '<span class="tag tag-issue">有问题</span>' if r['issues'] != '无' else '<span class="tag tag-ok">正常</span>'
    inv_tag = '<span class="tag tag-inventory">库存</span>' if r['is_inventory'] else ''
    html_content += f"""
                <tr>
                    <td>{r['col_id']}</td>
                    <td>{r['col_name'][:50]}</td>
                    <td>{inv_tag}</td>
                    <td>{r['original_valid']:,}</td>
                    <td>{r['cleaned_valid']:,}</td>
                    <td>均值:{r['original_mean']:.2f}<br>范围:[{r['original_min']:.2f}, {r['original_max']:.2f}]</td>
                    <td>{r['issues']} {issue_tag if r['issues'] != '无' else ''}</td>
                    <td>{r['fixes']}</td>
                </tr>
    """

html_content += f"""
            </tbody>
        </table>

        <footer style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #e2e8f0; color: #718096;">
            <p>生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </footer>
    </div>
</body>
</html>
"""

html_report_path = os.path.join(output_dir, 'validation_report.html')
with open(html_report_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"HTML报告已保存: {os.path.basename(html_report_path)}")
print("\n" + "=" * 80)
print("完成！")
print("=" * 80)

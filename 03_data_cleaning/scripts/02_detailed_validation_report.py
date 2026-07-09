import pandas as pd
import numpy as np
import os
import csv

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
input_path = os.path.join(base_dir, '01_data_preparation', 'outputs', 'merged_factors_daily_aligned.csv')
output_dir = os.path.join(base_dir, '04_feature_engineering', 'outputs')
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("生成详细校验报告（含处理方法和数目）")
print("=" * 80)

# 读取数据
print("\n读取数据...")
df = pd.read_csv(input_path, header=[0, 1], skiprows=0, encoding='utf-8-sig')
date_col = df.iloc[:, 0]
values_df = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

col_ids = list(df.columns.get_level_values(0))[1:]
col_names = list(df.columns.get_level_values(1))[1:]

# ==========================================
# 详细校验与清洗（记录每一步数量）
# ==========================================
print("\n开始详细校验...")

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

    # 记录各类问题数量
    neg_count = 0
    outlier_low_count = 0
    outlier_high_count = 0
    jump_count = 0

    # 1. 业务逻辑校验：库存数据非负
    is_inventory = any(keyword in col_name for keyword in ['库存', '库存:', '库存：'])

    if is_inventory:
        neg_mask = col_data < 0
        neg_count = neg_mask.sum()
        if neg_count > 0:
            cleaned_values.iloc[:, col_idx] = cleaned_values.iloc[:, col_idx].mask(neg_mask)

    # 2. 极端异常值检测：IQR方法
    valid_data = col_data.dropna()
    lower_bound = np.nan
    upper_bound = np.nan

    if len(valid_data) > 100:
        q1 = valid_data.quantile(0.25)
        q3 = valid_data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr

        outlier_low_mask = col_data < lower_bound
        outlier_high_mask = col_data > upper_bound
        outlier_low_count = outlier_low_mask.sum()
        outlier_high_count = outlier_high_mask.sum()

        if outlier_low_count + outlier_high_count > 0:
            cleaned_values.iloc[:, col_idx] = cleaned_values.iloc[:, col_idx].clip(lower=lower_bound, upper=upper_bound)

    # 3. 检测明显的跳变
    if len(valid_data) > 100:
        col_diff = col_data.diff()
        diff_std = col_diff.std()
        if pd.notna(diff_std) and diff_std > 0:
            jump_mask = col_diff.abs() > 5 * diff_std
            jump_count = jump_mask.sum()
            if jump_count > 0:
                cleaned_values.iloc[:, col_idx] = cleaned_values.iloc[:, col_idx].mask(jump_mask)

    # 统计清洗后
    cleaned_data = cleaned_values.iloc[:, col_idx]
    cleaned_valid = cleaned_data.notna().sum()

    validation_report.append({
        'col_id': col_id,
        'col_name': col_name,
        'is_inventory': is_inventory,
        'original_valid': original_valid,
        'cleaned_valid': cleaned_valid,
        'original_mean': original_mean,
        'original_min': original_min,
        'original_max': original_max,
        # 详细数目
        'neg_count': neg_count,
        'outlier_low_count': outlier_low_count,
        'outlier_high_count': outlier_high_count,
        'jump_count': jump_count,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    })

    if col_idx % 50 == 0:
        print(f"  已处理 {col_idx}/{values_df.shape[1]} 列")

# ==========================================
# 生成详细HTML报告
# ==========================================
print("\n生成详细HTML报告...")

# 统计汇总
total_neg = sum(r['neg_count'] for r in validation_report)
total_outlier_low = sum(r['outlier_low_count'] for r in validation_report)
total_outlier_high = sum(r['outlier_high_count'] for r in validation_report)
total_jump = sum(r['jump_count'] for r in validation_report)
total_fixed = total_neg + total_outlier_low + total_outlier_high + total_jump

cols_with_neg = sum(1 for r in validation_report if r['neg_count'] > 0)
cols_with_outlier = sum(1 for r in validation_report if r['outlier_low_count'] + r['outlier_high_count'] > 0)
cols_with_jump = sum(1 for r in validation_report if r['jump_count'] > 0)

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>数据校验与清洗详细报告</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 40px;
            background-color: #f5f7fa;
        }}
        .container {{
            max-width: 1600px;
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
        h2 {{
            color: #2d3748;
            margin-top: 35px;
            border-left: 4px solid #4299e1;
            padding-left: 12px;
        }}
        h3 {{
            color: #4a5568;
        }}
        .summary-box {{
            background: linear-gradient(135deg, #ebf8ff 0%, #e6fffa 100%);
            padding: 25px;
            border-radius: 10px;
            margin: 20px 0;
        }}
        .method-box {{
            background: #fffaf0;
            border-left: 4px solid #ed8936;
            padding: 15px 20px;
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
            font-size: 32px;
            font-weight: 700;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 11px;
        }}
        th, td {{
            border: 1px solid #e2e8f0;
            padding: 6px 8px;
            text-align: left;
        }}
        th {{
            background-color: #2d3748;
            color: white;
            font-weight: 600;
            position: sticky;
            top: 0;
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
            font-size: 10px;
            font-weight: 600;
            margin-right: 4px;
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
        .num-highlight {{
            color: #c53030;
            font-weight: 700;
        }}
        .table-container {{
            max-height: 600px;
            overflow-y: auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🛡️ 数据校验与清洗详细报告</h1>

        <div class="summary-box">
            <h3>概览</h3>
            <p><strong>数据时间范围:</strong> {date_col.iloc[0]} 至 {date_col.iloc[-1]}</p>
            <p><strong>总列数:</strong> {len(validation_report)}</p>
        </div>

        <div class="method-box">
            <h3>🔧 处理方法说明</h3>
            <table>
                <thead>
                    <tr>
                        <th>校验类型</th>
                        <th>判断标准</th>
                        <th>处理方法</th>
                        <th>业务逻辑</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>库存负值</strong></td>
                        <td>value &lt; 0</td>
                        <td>设为缺失值（NaN）</td>
                        <td>库存不可能为负，明显数据错误</td>
                    </tr>
                    <tr>
                        <td><strong>下界异常值</strong></td>
                        <td>value &lt; Q1 - 3×IQR</td>
                        <td>截断到下界</td>
                        <td>保留趋势，避免极端值影响</td>
                    </tr>
                    <tr>
                        <td><strong>上界异常值</strong></td>
                        <td>value &gt; Q3 + 3×IQR</td>
                        <td>截断到上界</td>
                        <td>保留趋势，避免极端值影响</td>
                    </tr>
                    <tr>
                        <td><strong>异常跳变</strong></td>
                        <td>|Δt| &gt; 5×σ(Δ)</td>
                        <td>设为缺失值（NaN）</td>
                        <td>单日跳变过大，可能是数据错误</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <h2>📊 总体处理统计</h2>

        <div class="summary-grid">
            <div class="summary-card" style="background: #fed7d7;">
                <h4>库存负值</h4>
                <div class="number" style="color: #c53030;">{total_neg:,}</div>
                <div style="font-size: 12px; color: #718096;">涉及 {cols_with_neg} 列</div>
            </div>
            <div class="summary-card" style="background: #feebc8;">
                <h4>下界异常值</h4>
                <div class="number" style="color: #c05621;">{total_outlier_low:,}</div>
                <div style="font-size: 12px; color: #718096;">涉及 {cols_with_outlier} 列</div>
            </div>
            <div class="summary-card" style="background: #feebc8;">
                <h4>上界异常值</h4>
                <div class="number" style="color: #c05621;">{total_outlier_high:,}</div>
                <div style="font-size: 12px; color: #718096;">涉及 {cols_with_outlier} 列</div>
            </div>
            <div class="summary-card" style="background: #fed7d7;">
                <h4>异常跳变</h4>
                <div class="number" style="color: #c53030;">{total_jump:,}</div>
                <div style="font-size: 12px; color: #718096;">涉及 {cols_with_jump} 列</div>
            </div>
        </div>

        <div class="summary-box">
            <p><strong>总计修正数据点:</strong> <span class="num-highlight">{total_fixed:,}</span> 个</p>
        </div>

        <h2>📋 每列详细报告</h2>

        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>列ID</th>
                        <th>列名</th>
                        <th>类型</th>
                        <th>原始有效</th>
                        <th>清洗后有效</th>
                        <th>原始统计</th>
                        <th>IQR边界</th>
                        <th>库存负值</th>
                        <th>下界异常</th>
                        <th>上界异常</th>
                        <th>异常跳变</th>
                        <th>总计修正</th>
                    </tr>
                </thead>
                <tbody>
"""

for r in validation_report:
    total_fixed_col = r['neg_count'] + r['outlier_low_count'] + r['outlier_high_count'] + r['jump_count']
    has_issue = total_fixed_col > 0

    inv_tag = '<span class="tag tag-inventory">库存</span>' if r['is_inventory'] else ''
    issue_tag = '<span class="tag tag-issue">有问题</span>' if has_issue else '<span class="tag tag-ok">正常</span>'

    neg_display = f'<span class="num-highlight">{r["neg_count"]:,}</span>' if r['neg_count'] > 0 else '-'
    out_low_display = f'<span class="num-highlight">{r["outlier_low_count"]:,}</span>' if r['outlier_low_count'] > 0 else '-'
    out_high_display = f'<span class="num-highlight">{r["outlier_high_count"]:,}</span>' if r['outlier_high_count'] > 0 else '-'
    jump_display = f'<span class="num-highlight">{r["jump_count"]:,}</span>' if r['jump_count'] > 0 else '-'
    total_display = f'<span class="num-highlight">{total_fixed_col:,}</span>' if total_fixed_col > 0 else '-'

    bounds_display = f'[{r["lower_bound"]:.2f}, {r["upper_bound"]:.2f}]' if pd.notna(r['lower_bound']) else '-'

    html_content += f"""
                    <tr>
                        <td>{r['col_id']}</td>
                        <td>{r['col_name'][:45]}</td>
                        <td>{inv_tag} {issue_tag}</td>
                        <td>{r['original_valid']:,}</td>
                        <td>{r['cleaned_valid']:,}</td>
                        <td>均值:{r['original_mean']:.2f}<br>范围:[{r['original_min']:.2f}, {r['original_max']:.2f}]</td>
                        <td>{bounds_display}</td>
                        <td style="text-align: center;">{neg_display}</td>
                        <td style="text-align: center;">{out_low_display}</td>
                        <td style="text-align: center;">{out_high_display}</td>
                        <td style="text-align: center;">{jump_display}</td>
                        <td style="text-align: center;"><strong>{total_display}</strong></td>
                    </tr>
    """

html_content += f"""
                </tbody>
            </table>
        </div>

        <footer style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #e2e8f0; color: #718096;">
            <p>生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </footer>
    </div>
</body>
</html>
"""

# 保存HTML报告
html_report_path = os.path.join(output_dir, 'validation_report_detailed.html')
with open(html_report_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

# 保存详细CSV
csv_path = os.path.join(output_dir, 'validation_report_detailed.csv')
with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        '列ID', '列名', '是否库存列', '原始有效样本', '清洗后有效样本',
        '原始均值', '原始最小值', '原始最大值',
        '库存负值数目', '下界异常值数目', '上界异常值数目', '异常跳变数目',
        'IQR下界', 'IQR上界'
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
            r['neg_count'],
            r['outlier_low_count'],
            r['outlier_high_count'],
            r['jump_count'],
            r['lower_bound'],
            r['upper_bound']
        ])

print(f"\n详细报告已保存:")
print(f"  HTML: {os.path.basename(html_report_path)}")
print(f"  CSV: {os.path.basename(csv_path)}")
print("\n" + "=" * 80)
print("完成！")
print("=" * 80)

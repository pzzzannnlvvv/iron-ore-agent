import pandas as pd
import numpy as np
import os
import csv

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
preprocessing_dir = os.path.join(base_dir, '02_preprocessing', 'outputs')
output_dir = os.path.join(base_dir, '03_stationarity_test', 'outputs')
figures_dir = os.path.join(base_dir, '03_stationarity_test', 'figures')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

print("=" * 80)
print("ADF检验与可视化")
print("=" * 80)

# 文件列表
files = [
    ('原始值', os.path.join(preprocessing_dir, '01_preprocessed_raw.csv')),
    ('一阶差分', os.path.join(preprocessing_dir, '02_preprocessed_diff.csv')),
    ('百分比变化', os.path.join(preprocessing_dir, '03_preprocessed_pct.csv')),
]

# ==========================================
# 简易ADF检验（自己实现，不依赖statsmodels）
# 使用OLS估计：Δy_t = α + βt + γy_{t-1} + δ1Δy_{t-1} + ... + ε_t
# H0: γ=0 (存在单位根，非平稳)
# ==========================================
def simple_adf_test(series, maxlag=5):
    """简易ADF检验实现"""
    series = series.dropna()
    n = len(series)

    if n < 50:
        return None, None, None

    # 差分
    dy = series.diff().dropna()
    y_lag = series.shift(1).dropna()

    # 对齐长度
    min_len = min(len(dy), len(y_lag))
    dy = dy.iloc[:min_len]
    y_lag = y_lag.iloc[:min_len]

    # 构建回归矩阵 X: [1, trend, y_lag1, dy_lag1, ...]
    X = np.ones((len(dy), 3 + maxlag))
    X[:, 1] = np.arange(len(dy))  # 趋势项
    X[:, 2] = y_lag.values

    for lag in range(1, maxlag + 1):
        dy_lag = dy.shift(lag).fillna(0)
        X[:, 2 + lag] = dy_lag.values

    y = dy.values

    # OLS估计 (X'X)^{-1}X'y
    try:
        XTX = X.T @ X
        XTy = X.T @ y
        beta = np.linalg.inv(XTX) @ XTy

        # 计算t统计量 for γ (y_lag系数)
        residuals = y - X @ beta
        sigma2 = (residuals @ residuals) / (len(y) - len(beta))
        cov = sigma2 * np.linalg.inv(XTX)
        se = np.sqrt(np.diag(cov))
        t_stat = beta[2] / se[2]

        # 简易临界值判断（简化版，仅供参考）
        # 真实ADF临界值需要查表，这里用简化规则
        if t_stat < -3.45:
            result = "平稳 (99%)"
        elif t_stat < -2.87:
            result = "平稳 (95%)"
        elif t_stat < -2.57:
            result = "平稳 (90%)"
        else:
            result = "非平稳"

        return t_stat, result, beta[2]
    except:
        return None, "计算失败", None

# ==========================================
# 生成简易SVG图表（不依赖matplotlib）
# ==========================================
def generate_svg_plot(series, title, filename):
    """生成简易SVG时间序列图"""
    series = series.dropna()
    if len(series) < 10:
        return

    n = len(series)
    vals = series.values

    # 归一化
    y_min, y_max = vals.min(), vals.max()
    y_range = y_max - y_min if y_max > y_min else 1

    width, height = 900, 300
    padding = 50

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
        <rect width="{width}" height="{height}" fill="#f7fafc"/>
        <text x="{width/2}" y="25" text-anchor="middle" font-family="Arial" font-size="16" fill="#2d3748">{title}</text>
    """

    # 网格线
    for i in range(5):
        y = padding + (height - 2*padding) * i / 4
        val = y_max - y_range * i / 4
        svg += f'<line x1="{padding}" y1="{y}" x2="{width-padding}" y2="{y}" stroke="#e2e8f0" stroke-width="1"/>'
        svg += f'<text x="{padding-5}" y="{y+4}" text-anchor="end" font-family="Arial" font-size="10" fill="#718096">{val:.2f}</text>'

    # 绘制线条
    path = "M "
    for i in range(n):
        x = padding + (width - 2*padding) * i / (n-1)
        y = padding + (height - 2*padding) * (y_max - vals[i]) / y_range
        path += f"{x},{y} "
        if i == 0:
            path += "L "

    svg += f'<path d="{path}" fill="none" stroke="#3182ce" stroke-width="1.5"/>'

    # 边界
    svg += f'<rect x="{padding}" y="{padding}" width="{width-2*padding}" height="{height-2*padding}" fill="none" stroke="#cbd5e0" stroke-width="1"/>'
    svg += '</svg>'

    with open(filename, 'w') as f:
        f.write(svg)

# ==========================================
# 处理每个文件
# ==========================================
all_results = []

for name, filepath in files:
    print(f"\n处理 {name} ...")

    df = pd.read_csv(filepath, header=[0, 1], skiprows=0, encoding='utf-8-sig')
    date_col = df.iloc[:, 0]
    values_df = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

    col_ids = list(df.columns.get_level_values(0))[1:]
    col_names = list(df.columns.get_level_values(1))[1:]

    # 对前20列生成图表和检验
    for col_idx in range(min(20, values_df.shape[1])):
        col_data = values_df.iloc[:, col_idx].dropna()

        if len(col_data) < 100:
            continue

        print(f"  列 {col_idx+1}: {col_names[col_idx][:30]}...")

        # ADF检验
        t_stat, adf_result, gamma = simple_adf_test(col_data, maxlag=3)

        # 生成SVG图表
        safe_name = "".join(c if c.isalnum() else "_" for c in col_names[col_idx][:30])
        svg_filename = os.path.join(figures_dir, f"{name.replace(' ','_')}_col{col_idx+1}_{safe_name}.svg")
        try:
            generate_svg_plot(col_data, f"{name} - {col_names[col_idx][:40]}", svg_filename)
        except:
            pass

        all_results.append({
            'version': name,
            'col_id': col_ids[col_idx],
            'col_name': col_names[col_idx],
            'n_obs': len(col_data),
            'mean': col_data.mean(),
            'std': col_data.std(),
            'adf_t_stat': t_stat,
            'adf_result': adf_result,
            'gamma': gamma,
            'svg_file': os.path.basename(svg_filename)
        })

# ==========================================
# 生成增强版HTML报告（带图表）
# ==========================================
print("\n生成HTML报告...")

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>铁矿石库存预测 - ADF平稳性检验报告</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            margin: 40px;
            background-color: #f5f7fa;
        }}
        .container {{
            max-width: 1200px;
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
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 13px;
        }}
        th, td {{
            border: 1px solid #e2e8f0;
            padding: 10px 12px;
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
            padding: 4px 12px;
            border-radius: 14px;
            font-size: 12px;
            font-weight: 600;
        }}
        .tag-stationary {{
            background-color: #c6f6d5;
            color: #22543d;
        }}
        .tag-nonstationary {{
            background-color: #fed7d7;
            color: #742a2a;
        }}
        .tag-neutral {{
            background-color: #e2e8f0;
            color: #2d3748;
        }}
        .version-section {{
            margin-top: 40px;
            padding: 25px;
            background: #f7fafc;
            border-radius: 10px;
        }}
        .chart-card {{
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
        }}
        .chart-container {{
            overflow-x: auto;
        }}
        .chart-container img {{
            max-width: 100%;
        }}
        .info-note {{
            background: #fffaf0;
            border-left: 4px solid #ed8936;
            padding: 15px 20px;
            margin: 20px 0;
        }}
        .result-summary {{
            display: flex;
            gap: 20px;
            margin: 20px 0;
            flex-wrap: wrap;
        }}
        .result-card {{
            flex: 1;
            min-width: 200px;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .result-card h4 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            color: #718096;
        }}
        .result-card .number {{
            font-size: 32px;
            font-weight: 700;
        }}
        .nav-tabs {{
            display: flex;
            gap: 5px;
            margin: 20px 0;
        }}
        .nav-tab {{
            padding: 10px 20px;
            background: #e2e8f0;
            border-radius: 8px 8px 0 0;
            cursor: pointer;
            font-weight: 600;
        }}
        .nav-tab.active {{
            background: #3182ce;
            color: white;
        }}
        .tab-content {{
            display: none;
        }}
        .tab-content.active {{
            display: block;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 铁矿石库存预测 - ADF平稳性检验报告</h1>

        <div class="summary-box">
            <h3>概览</h3>
            <p><strong>分析时间范围:</strong> 2015-12-25 至 2026-05-08</p>
            <p><strong>检验方法:</strong> 简易ADF单位根检验（自实现）</p>
            <p><strong>预处理版本:</strong> 原始值、一阶差分、百分比变化</p>
        </div>

        <div class="info-note">
            <strong>说明:</strong> 本报告使用简易ADF检验实现，仅供参考。
            严谨的学术研究建议安装statsmodels库使用官方ADF检验。
            <br><strong>ADF原假设:</strong> 存在单位根（序列非平稳）
        </div>
"""

# 统计汇总
for version_name in ['原始值', '一阶差分', '百分比变化']:
    version_results = [r for r in all_results if r['version'] == version_name]
    stationary_count = sum(1 for r in version_results if '平稳' in str(r['adf_result']))
    total_count = len(version_results)

    html_content += f"""
        <div class="version-section">
            <h2>📈 {version_name}</h2>

            <div class="result-summary">
                <div class="result-card" style="background: #ebf8ff;">
                    <h4>总列数</h4>
                    <div class="number" style="color: #3182ce;">{total_count}</div>
                </div>
                <div class="result-card" style="background: #c6f6d5;">
                    <h4>平稳列数</h4>
                    <div class="number" style="color: #22543d;">{stationary_count}</div>
                </div>
                <div class="result-card" style="background: #fed7d7;">
                    <h4>非平稳列数</h4>
                    <div class="number" style="color: #742a2a;">{total_count - stationary_count}</div>
                </div>
            </div>

            <table>
                <thead>
                    <tr>
                        <th>列ID</th>
                        <th>列名</th>
                        <th>样本数</th>
                        <th>均值</th>
                        <th>ADF t统计量</th>
                        <th>检验结果</th>
                    </tr>
                </thead>
                <tbody>
    """

    for r in version_results:
        tag_class = 'tag-stationary' if '平稳' in str(r['adf_result']) else 'tag-nonstationary'
        html_content += f"""
                    <tr>
                        <td>{r['col_id']}</td>
                        <td>{r['col_name'][:50]}</td>
                        <td>{r['n_obs']:,}</td>
                        <td>{r['mean']:.4f}</td>
                        <td>{r['adf_t_stat']:.4f}</td>
                        <td><span class="tag {tag_class}">{r['adf_result']}</span></td>
                    </tr>
        """

    html_content += """
                </tbody>
            </table>

            <h3>📊 序列图</h3>
    """

    for r in version_results:
        tag_class = 'tag-stationary' if '平稳' in str(r['adf_result']) else 'tag-nonstationary'
        html_content += f"""
            <div class="chart-card">
                <h4>{r['col_id']} - {r['col_name'][:50]}
                    <span class="tag {tag_class}" style="float:right;">{r['adf_result']}</span>
                </h4>
                <div class="chart-container">
                    <img src="../figures/{r['svg_file']}" alt="序列图">
                </div>
                <p style="font-size: 12px; color: #718096; margin: 10px 0 0 0;">
                    ADF t统计量: {r['adf_t_stat']:.4f} | 样本数: {r['n_obs']:,} | 均值: {r['mean']:.4f}
                </p>
            </div>
        """

    html_content += """
        </div>
    """

html_content += """
        <h2>💡 建议</h2>
        <div class="summary-box">
            <ul>
                <li><strong>原始值:</strong> 通常非平稳，适合树模型、神经网络（不需要严格平稳）</li>
                <li><strong>一阶差分/百分比变化:</strong> 通常更平稳，适合ARIMA等传统时间序列模型</li>
                <li><strong>实践策略:</strong> 保留三个版本，在模型训练阶段对比效果</li>
                <li><strong>时间范围:</strong> 建议使用2018-2024年数据（更完整）</li>
            </ul>
        </div>

        <footer style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #e2e8f0; color: #718096;">
            <p>生成时间: """ + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
        </footer>
    </div>
</body>
</html>
"""

# 保存HTML
html_path = os.path.join(output_dir, 'adf_stationarity_report.html')
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

# 保存CSV详细结果
csv_path = os.path.join(output_dir, 'adf_test_results.csv')
with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['预处理版本', '列ID', '列名', '样本数', '均值', '标准差', 'ADF_t统计量', '检验结果', 'gamma系数'])
    for r in all_results:
        writer.writerow([
            r['version'],
            r['col_id'],
            r['col_name'],
            r['n_obs'],
            r['mean'],
            r['std'],
            r['adf_t_stat'],
            r['adf_result'],
            r['gamma']
        ])

print(f"\nADF报告已保存:")
print(f"  HTML: {os.path.basename(html_path)}")
print(f"  CSV: {os.path.basename(csv_path)}")
print(f"  图表已保存到 figures/ 文件夹")
print("\n" + "=" * 80)
print("完成！")
print("=" * 80)

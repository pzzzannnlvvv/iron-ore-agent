import pandas as pd
import numpy as np
import os

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
preprocessing_dir = os.path.join(base_dir, '02_preprocessing', 'outputs')
output_dir = os.path.join(base_dir, '03_stationarity_test', 'outputs')
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("生成平稳性检验HTML报告")
print("=" * 80)

# 文件列表
files = [
    ('原始值', os.path.join(preprocessing_dir, '01_preprocessed_raw.csv')),
    ('一阶差分', os.path.join(preprocessing_dir, '02_preprocessed_diff.csv')),
    ('百分比变化', os.path.join(preprocessing_dir, '03_preprocessed_pct.csv')),
]

# ==========================================
# 计算统计指标
# ==========================================
all_stats = []

for name, filepath in files:
    print(f"\n处理 {name} ...")

    df = pd.read_csv(filepath, header=[0, 1], skiprows=0, encoding='utf-8-sig')
    date_col = df.iloc[:, 0]
    values_df = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

    col_ids = list(df.columns.get_level_values(0))[1:]
    col_names = list(df.columns.get_level_values(1))[1:]

    for col_idx in range(min(50, values_df.shape[1])):  # 最多分析前50列
        col_data = values_df.iloc[:, col_idx].dropna()

        if len(col_data) < 100:
            continue

        mean_val = col_data.mean()
        std_val = col_data.std()
        min_val = col_data.min()
        max_val = col_data.max()

        if len(col_data) > 10:
            autocorr1 = col_data.autocorr(lag=1)
            autocorr7 = col_data.autocorr(lag=7) if len(col_data) > 14 else np.nan
        else:
            autocorr1 = np.nan
            autocorr7 = np.nan

        # 简易平稳判断：基于自相关系数快速衰减
        is_stationary_simple = "可能平稳" if (pd.notna(autocorr1) and abs(autocorr1) < 0.9) else "可能非平稳"

        all_stats.append({
            'version': name,
            'col_id': col_ids[col_idx],
            'col_name': col_names[col_idx],
            'n_obs': len(col_data),
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'autocorr1': autocorr1,
            'autocorr7': autocorr7,
            'stationary_hint': is_stationary_simple
        })

# ==========================================
# 生成HTML
# ==========================================
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>铁矿石库存预测 - 平稳性检验报告</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
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
        h2 {{
            color: #2d3748;
            margin-top: 35px;
            border-left: 4px solid #4299e1;
            padding-left: 12px;
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
        .tag-stationary {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }}
        .tag-yes {{
            background-color: #c6f6d5;
            color: #22543d;
        }}
        .tag-no {{
            background-color: #fed7d7;
            color: #742a2a;
        }}
        .version-section {{
            margin-top: 40px;
            padding: 20px;
            background: #f7fafc;
            border-radius: 8px;
        }}
        .info-note {{
            background: #fffaf0;
            border-left: 4px solid #ed8936;
            padding: 15px 20px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 铁矿石库存预测 - 平稳性检验报告</h1>

        <div class="summary-box">
            <h3>概览</h3>
            <p><strong>分析时间范围:</strong> 2015-12-25 至 2026-05-08</p>
            <p><strong>预处理版本:</strong> 原始值、一阶差分、百分比变化</p>
            <p><strong>分析列数:</strong> 前50列（包含标的数据和主要因子）</p>
        </div>

        <div class="info-note">
            <strong>说明:</strong> 本报告使用简易平稳性判断（基于自相关系数）。
            如需严谨的ADF/KPSS检验，请安装statsmodels库后运行完整检验。
        </div>

        <h2>📈 三种预处理版本对比</h2>

        <table>
            <thead>
                <tr>
                    <th>版本</th>
                    <th>特点</th>
                    <th>适用模型</th>
                    <th>业务含义</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>原始值</strong></td>
                    <td>保留数据原始水平，通常有趋势</td>
                    <td>XGBoost, LightGBM, 神经网络等</td>
                    <td>直接预测库存水平</td>
                </tr>
                <tr>
                    <td><strong>一阶差分</strong></td>
                    <td>ΔY = Yt - Yt-1，通常较平稳</td>
                    <td>ARIMA, SARIMA等时间序列模型</td>
                    <td>预测库存变化量</td>
                </tr>
                <tr>
                    <td><strong>百分比变化</strong></td>
                    <td>(Yt - Yt-1)/|Yt-1|，通常较平稳</td>
                    <td>各类模型均可</td>
                    <td>预测库存增长率</td>
                </tr>
            </tbody>
        </table>
"""

for version_name in ['原始值', '一阶差分', '百分比变化']:
    version_stats = [s for s in all_stats if s['version'] == version_name]

    html_content += f"""
        <div class="version-section">
            <h2>🔍 {version_name} - 详细统计</h2>
            <table>
                <thead>
                    <tr>
                        <th>列ID</th>
                        <th>列名</th>
                        <th>样本数</th>
                        <th>均值</th>
                        <th>标准差</th>
                        <th>最小值</th>
                        <th>最大值</th>
                        <th>自相关(lag1)</th>
                        <th>自相关(lag7)</th>
                        <th>平稳性判断</th>
                    </tr>
                </thead>
                <tbody>
    """

    for stat in version_stats:
        tag_class = 'tag-yes' if stat['stationary_hint'] == '可能平稳' else 'tag-no'
        html_content += f"""
                    <tr>
                        <td>{stat['col_id']}</td>
                        <td>{stat['col_name'][:40]}</td>
                        <td>{stat['n_obs']:,}</td>
                        <td>{stat['mean']:.4f}</td>
                        <td>{stat['std']:.4f}</td>
                        <td>{stat['min']:.4f}</td>
                        <td>{stat['max']:.4f}</td>
                        <td>{stat['autocorr1']:.4f}</td>
                        <td>{stat['autocorr7']:.4f}</td>
                        <td><span class="tag-stationary {tag_class}">{stat['stationary_hint']}</span></td>
                    </tr>
        """

    html_content += """
                </tbody>
            </table>
        </div>
    """

html_content += """
        <h2>💡 建议</h2>
        <div class="summary-box">
            <ul>
                <li><strong>原始值:</strong> 适用于树模型、神经网络，不需要严格平稳</li>
                <li><strong>一阶差分/百分比变化:</strong> 适用于传统时间序列模型</li>
                <li><strong>实践建议:</strong> 保留三个版本，在模型训练阶段对比效果</li>
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
html_path = os.path.join(output_dir, 'stationarity_report.html')
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"\nHTML报告已保存: {os.path.basename(html_path)}")
print("\n" + "=" * 80)
print("完成！")
print("=" * 80)

#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import csv

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
preprocessing_dir = os.path.join(base_dir, '02_preprocessing', 'outputs')
output_dir = os.path.join(base_dir, '04_stationarity_test', 'outputs')
figures_dir = os.path.join(base_dir, '04_stationarity_test', 'figures')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

print("=" * 80)
print("补充检验：卡拉加斯粉 + 超特粉")
print("=" * 80)

# 目标标的列表
target_columns = [
    ('ID01517102', '超特粉：港口库存：华东地区：青岛港：明细（周度）'),
    ('ID01517011', '超特粉：港口库存：华北地区：曹妃甸港：明细（周度）'),
    ('ID01516873', '超特粉：港口库存：长江沿江地区：江阴港：明细（周度）'),
    ('ID01517226', '卡拉加斯粉：港口库存：华东地区：青岛港：明细（周度）'),
    ('ID01517224', '卡拉加斯粉：港口库存：华北地区：曹妃甸港：明细（周度）'),
    ('ID01517220', '卡拉加斯粉：港口库存：长江沿江地区：江阴港：明细（周度）'),
]

# 文件列表
files = [
    ('原始值', os.path.join(preprocessing_dir, '01_preprocessed_raw.csv')),
    ('一阶差分', os.path.join(preprocessing_dir, '02_preprocessed_diff.csv')),
    ('百分比变化', os.path.join(preprocessing_dir, '03_preprocessed_pct.csv')),
]

# ==========================================
# 简易ADF检验（复用原有实现）
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

    # OLS估计
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

        # 简易临界值判断
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
# 生成SVG图表
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
# 读取现有结果
# ==========================================
existing_results = []
existing_csv_path = os.path.join(output_dir, 'adf_test_results.csv')
if os.path.exists(existing_csv_path):
    with open(existing_csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_results.append(row)

print(f"\n现有结果数: {len(existing_results)}")

# ==========================================
# 检查哪些已经检验过的列ID
tested_col_ids = set(r['列ID'] for r in existing_results)
print(f"已检验的列ID数: {len(tested_col_ids)}")

# ==========================================
# 处理每个文件，补充检验
# ==========================================
new_results = []

for name, filepath in files:
    print(f"\n处理 {name} ...")

    df = pd.read_csv(filepath, header=[0, 1], skiprows=0, encoding='utf-8-sig')

    # 获取列ID和列名
    col_ids = list(df.columns.get_level_values(0))[1:]
    col_names = list(df.columns.get_level_values(1))[1:]
    values_df = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

    # 查找目标列
    for target_id, target_name in target_columns:
        # 查找该ID的位置
        col_idx = None
        for i, col_id in enumerate(col_ids):
            if col_id == target_id:
                col_idx = i
                break

        if col_idx is None:
            print(f"  未找到: {target_id} - {target_name}")
            continue

        # 检查是否已经检验过
        already_tested = False
        for r in existing_results:
            if r['列ID'] == target_id and r['预处理版本'] == name:
                already_tested = True
                break

        if already_tested:
            print(f"  已检验过，跳过: {target_id} - {target_name[:30]}")
            continue

        col_data = values_df.iloc[:, col_idx].dropna()

        if len(col_data) < 100:
            print(f"  数据不足，跳过: {target_id} - {target_name[:30]} (n={len(col_data)})")
            continue

        print(f"  正在检验: {target_id} - {target_name[:40]} (n={len(col_data)})")

        # ADF检验
        t_stat, adf_result, gamma = simple_adf_test(col_data, maxlag=3)

        # 生成SVG图表
        safe_name = "".join(c if c.isalnum() else "_" for c in target_name[:30])
        svg_filename = os.path.join(figures_dir, f"{name.replace(' ','_')}_{target_id}_{safe_name}.svg")
        try:
            generate_svg_plot(col_data, f"{name} - {target_name[:40]}", svg_filename)
        except Exception as e:
            print(f"    SVG生成失败: {e}")
            svg_filename = ""

        new_results.append({
            'version': name,
            'col_id': target_id,
            'col_name': target_name,
            'n_obs': len(col_data),
            'mean': col_data.mean(),
            'std': col_data.std(),
            'adf_t_stat': t_stat,
            'adf_result': adf_result,
            'gamma': gamma,
            'svg_file': os.path.basename(svg_filename) if svg_filename else ""
        })

print(f"\n新增检验结果数: {len(new_results)}")

# ==========================================
# 合并并保存CSV
# ==========================================
if new_results:
    combined_results = existing_results.copy()

    for r in new_results:
        combined_results.append({
            '预处理版本': r['version'],
            '列ID': r['col_id'],
            '列名': r['col_name'],
            '样本数': str(r['n_obs']),
            '均值': str(r['mean']),
            '标准差': str(r['std']),
            'ADF_t统计量': str(r['adf_t_stat']),
            '检验结果': str(r['adf_result']),
            'gamma系数': str(r['gamma'])
        })

    # 保存CSV
    csv_path = os.path.join(output_dir, 'adf_test_results.csv')
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['预处理版本', '列ID', '列名', '样本数', '均值', '标准差', 'ADF_t统计量', '检验结果', 'gamma系数'])
        writer.writeheader()
        for r in combined_results:
            writer.writerow(r)
    print(f"\n已更新: {csv_path}")

    # ==========================================
    # 更新HTML报告
    # ==========================================
    # 先读取现有的HTML，在HTML太，，，，
    #，
    # ==========================================
    # 为新增结果生成补充 HTML 部分
    # ==========================================
    html_addendum = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <title>补充检验：卡拉加斯粉+超特粉</title>
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
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 补充检验：卡拉加斯粉 + 超特粉</h1>

            <div style="background: #fffaf0; border-left: 4px solid #ed8936; padding: 15px 20px; margin: 20px 0;">
                <strong>说明:</strong> 本报告是对原有ADF平稳性检验的补充，专门针对卡拉加斯粉和超特粉的6个库存标的。
            </div>
            <p>生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    """

    for version_name in ['原始值', '一阶差分', '百分比变化']:
        version_new_results = [r for r in new_results if r['version'] == version_name]

        if not version_new_results:
            continue

        stationary_count = sum(1 for r in version_new_results if '平稳' in str(r['adf_result']))
        total_count = len(version_new_results)

        html_addendum += f"""
            <div class="version-section">
                <h2>📈 {version_name}</h2>

                <div style="display: flex; gap: 20px; margin: 20px 0; flex-wrap: wrap;">
                    <div style="flex: 1; min-width: 200px; padding: 20px; border-radius: 10px; text-align: center; background: #c6f6d5;">
                        <h4 style="margin: 0 0 10px 0; font-size: 14px; color: #718096;">平稳列数</h4>
                        <div style="font-size: 32px; font-weight: 700; color: #22543d;">{stationary_count}</div>
                    </div>
                    <div style="flex: 1; min-width: 200px; padding: 20px; border-radius: 10px; text-align: center; background: #fed7d7;">
                        <h4 style="margin: 0 0 10px 0; font-size: 14px; color: #718096;">非平稳列数</h4>
                        <div style="font-size: 32px; font-weight: 700; color: #742a2a;">{total_count - stationary_count}</div>
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

        for r in version_new_results:
            tag_class = 'tag-stationary' if '平稳' in str(r['adf_result']) else 'tag-nonstationary'
            html_addendum += f"""
                        <tr>
                            <td>{r['col_id']}</td>
                            <td>{r['col_name'][:50]}</td>
                            <td>{r['n_obs']:,}</td>
                            <td>{r['mean']:.4f}</td>
                            <td>{r['adf_t_stat']:.4f}</td>
                            <td><span class="tag {tag_class}">{r['adf_result']}</span></td>
                        </tr>
            """

        html_addendum += """
                    </tbody>
                </table>

                <h3>📊 序列图</h3>
        """

        for r in version_new_results:
            if not r['svg_file']:
                continue
            tag_class = 'tag-stationary' if '平稳' in str(r['adf_result']) else 'tag-nonstationary'
            html_addendum += f"""
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

        html_addendum += """
            </div>
        """

    html_addendum += """
        </div>
    </body>
    </html>
    """

    supplement_html_path = os.path.join(output_dir, 'adf_stationarity_report_supplement.html')
    with open(supplement_html_path, 'w', encoding='utf-8') as f:
        f.write(html_addendum)
    print(f"已生成补充HTML: {supplement_html_path}")

    print(f"\n完成！")
    print(f"  - 新增检验数: {len(new_results)}")
    print(f"  - 累计检验: {len(combined_results)}")
    print("=" * 80)

else:
    print("\n没有新结果，所有标的已检验过！")
    print("=" * 80)

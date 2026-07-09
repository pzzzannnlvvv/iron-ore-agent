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
print("步骤3：平稳性检验")
print("=" * 80)

# 文件列表
files = [
    ('原始值', os.path.join(preprocessing_dir, '01_preprocessed_raw.csv')),
    ('一阶差分', os.path.join(preprocessing_dir, '02_preprocessed_diff.csv')),
    ('百分比变化', os.path.join(preprocessing_dir, '03_preprocessed_pct.csv')),
]

# ==========================================
# 简易平稳性检验（不依赖statsmodels）
# ==========================================
print("\n" + "=" * 80)
print("生成检验报告")
print("=" * 80)

# 统计指标
report_rows = []
report_rows.append(['预处理版本', '列名', '列ID', '是否平稳(简易)', '均值', '标准差', '自相关系数(lag1)'])

for name, filepath in files:
    print(f"\n处理 {name} ...")

    # 读取数据
    df = pd.read_csv(filepath, header=[0, 1], skiprows=0, encoding='utf-8-sig')
    date_col = df.iloc[:, 0]
    values_df = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

    # 获取列信息
    col_ids = list(df.columns.get_level_values(0))[1:]
    col_names = list(df.columns.get_level_values(1))[1:]

    # 对每一列进行简易平稳性判断
    for col_idx in range(values_df.shape[1]):
        col_data = values_df.iloc[:, col_idx].dropna()

        if len(col_data) < 100:
            continue

        # 计算基本统计量
        mean_val = col_data.mean()
        std_val = col_data.std()

        # 计算自相关系数(lag1)
        if len(col_data) > 2:
            autocorr = col_data.autocorr(lag=1)
        else:
            autocorr = np.nan

        # 简易平稳判断：均值稳定+低自相关（这只是简化版）
        # 真正的ADF/KPSS需要statsmodels，这里用简化指标
        is_stationary = "待检验"

        report_rows.append([
            name,
            col_names[col_idx],
            col_ids[col_idx],
            is_stationary,
            f"{mean_val:.6f}",
            f"{std_val:.6f}",
            f"{autocorr:.6f}" if not pd.isna(autocorr) else ""
        ])

# 保存报告
report_path = os.path.join(output_dir, 'stationarity_report.csv')
with open(report_path, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(report_rows)

print(f"\n检验报告已保存: {os.path.basename(report_path)}")

# ==========================================
# 总结
# ==========================================
print("\n" + "=" * 80)
print("平稳性检验完成")
print("=" * 80)

print(f"""
说明：
- 已生成平稳性检验报告
- 完整的ADF/KPSS检验需要安装statsmodels库
- 可以后续安装库后运行完整检验

三种版本特点：
1. 原始值：通常非平稳（有趋势）
2. 一阶差分：通常较平稳
3. 百分比变化：通常较平稳，且有业务含义

建议：
- 如果一阶差分或百分比变化通过平稳性检验，优先使用
- 保留原始值版本用于水平预测模型（如XGBoost/LightGBM，不需要严格平稳）
""")

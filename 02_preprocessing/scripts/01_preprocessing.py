import pandas as pd
import numpy as np
import os
import csv

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
input_path = os.path.join(base_dir, '01_data_preparation', 'outputs', 'merged_factors_daily_aligned.csv')
output_dir = os.path.join(base_dir, '02_preprocessing', 'outputs')
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("步骤2：三种数据预处理")
print("=" * 80)

# 读取数据
print("\n读取对齐后的数据...")
df = pd.read_csv(input_path, header=[0, 1], skiprows=0, encoding='utf-8-sig')

# 分离日期列和数值列
date_col = df.iloc[:, 0]
values_df = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

print(f"数据形状: {df.shape}")
print(f"日期范围: {date_col.iloc[0]} 至 {date_col.iloc[-1]}")

# 获取列标题
header_ids = ['data_date'] + list(df.columns.get_level_values(0))[1:]
header_names = ['data_date'] + list(df.columns.get_level_values(1))[1:]

# ==========================================
# 预处理1：原始值
# ==========================================
print("\n" + "=" * 80)
print("生成1：原始值数据")
print("=" * 80)

output_rows = []
output_rows.append(header_ids)
output_rows.append(header_names)

for i in range(len(date_col)):
    row = [date_col.iloc[i]]
    for j in range(values_df.shape[1]):
        val = values_df.iloc[i, j]
        if pd.isna(val):
            row.append('')
        else:
            row.append(f"{val:.6f}")
    output_rows.append(row)

raw_path = os.path.join(output_dir, '01_preprocessed_raw.csv')
with open(raw_path, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(output_rows)

print(f"已保存: {os.path.basename(raw_path)}")

# ==========================================
# 预处理2：一阶差分
# ==========================================
print("\n" + "=" * 80)
print("生成2：一阶差分数据")
print("=" * 80)

diff_df = values_df.diff()

output_rows = []
output_rows.append(header_ids)
output_rows.append(header_names)

for i in range(len(date_col)):
    row = [date_col.iloc[i]]
    for j in range(diff_df.shape[1]):
        val = diff_df.iloc[i, j]
        if pd.isna(val):
            row.append('')
        else:
            row.append(f"{val:.6f}")
    output_rows.append(row)

diff_path = os.path.join(output_dir, '02_preprocessed_diff.csv')
with open(diff_path, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(output_rows)

print(f"已保存: {os.path.basename(diff_path)}")

# ==========================================
# 预处理3：百分比变化
# ==========================================
print("\n" + "=" * 80)
print("生成3：百分比变化数据")
print("=" * 80)

# 计算百分比变化（注意：分母接近0时处理）
pct_df = pd.DataFrame(index=values_df.index, columns=values_df.columns)

for col in range(values_df.shape[1]):
    col_data = values_df.iloc[:, col]
    col_shifted = col_data.shift(1)

    # 计算百分比变化：(当前 - 上期) / |上期|
    # 对于绝对值小于0.0001的上期值，设为NaN避免极端值
    pct_change = np.where(
        np.abs(col_shifted) >= 0.0001,
        (col_data - col_shifted) / np.abs(col_shifted),
        np.nan
    )
    pct_df.iloc[:, col] = pct_change

output_rows = []
output_rows.append(header_ids)
output_rows.append(header_names)

for i in range(len(date_col)):
    row = [date_col.iloc[i]]
    for j in range(pct_df.shape[1]):
        val = pct_df.iloc[i, j]
        if pd.isna(val):
            row.append('')
        else:
            row.append(f"{val:.8f}")
    output_rows.append(row)

pct_path = os.path.join(output_dir, '03_preprocessed_pct.csv')
with open(pct_path, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(output_rows)

print(f"已保存: {os.path.basename(pct_path)}")

# ==========================================
# 生成预处理统计报告
# ==========================================
print("\n" + "=" * 80)
print("预处理统计报告")
print("=" * 80)

print(f"""
三种预处理版本已生成:
1. 原始值: {os.path.basename(raw_path)}
2. 一阶差分: {os.path.basename(diff_path)}
3. 百分比变化: {os.path.basename(pct_path)}

各版本统计:
- 原始值: 保留数据原始水平，用于水平预测模型
- 一阶差分: 计算ΔY = Yt - Yt-1，用于平稳性建模
- 百分比变化: 计算(Yt - Yt-1)/|Yt-1|，用于增长率建模

下一阶段建议:
- 对三种版本分别进行平稳性检验
- 根据检验结果选择适合的版本进入特征工程
""")

print("\n" + "=" * 80)
print("预处理完成！")
print("=" * 80)

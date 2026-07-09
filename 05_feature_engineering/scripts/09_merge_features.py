import pandas as pd
import numpy as np
import os
import csv

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
output_dir = os.path.join(base_dir, '05_feature_engineering', 'outputs')
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("Step 9: 特征合并与初步清洗")
print("=" * 80)

feature_files = [
    '01_time_features.csv',
    '02_target_lags.csv',
    '03_factor_lags.csv',
    '04_trend_volatility.csv',
    '05_business_features.csv',
    '06_technical_indicators.csv',
    '07_yoy_mom_features.csv',
]

# ==========================================
# 读取所有特征文件
# ==========================================
all_dfs = []
all_ids = []
all_names = []
date_col = None

for fname in feature_files:
    fpath = os.path.join(output_dir, fname)
    if not os.path.exists(fpath):
        print(f"跳过(不存在): {fname}")
        continue

    print(f"\n读取 {fname}...")
    fsize_mb = os.path.getsize(fpath) / (1024 * 1024)
    print(f"  文件大小: {fsize_mb:.1f} MB")

    df = pd.read_csv(fpath, header=[0, 1], encoding='utf-8-sig', low_memory=False)

    if date_col is None:
        date_col = df.iloc[:, 0]
    else:
        # 验证日期一致性
        if not (df.iloc[:, 0] == date_col).all():
            print("  WARNING: 日期列不一致!")

    col_ids = list(df.columns.get_level_values(0))[1:]
    col_names = list(df.columns.get_level_values(1))[1:]
    values = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

    all_ids.extend(col_ids)
    all_names.extend(col_names)
    all_dfs.append(values)

    print(f"  列数: {len(col_ids)}, 累积列数: {sum(d.shape[1] for d in all_dfs)}")

n = len(date_col)
total_cols_before = sum(d.shape[1] for d in all_dfs)
print(f"\n合并前列总数: {total_cols_before}")

# ==========================================
# 合并 — 用 MultiIndex 列保证标签与数据始终同步
# ==========================================
print("\n合并特征矩阵...")
merged = pd.concat(all_dfs, axis=1)
merged.columns = pd.MultiIndex.from_arrays([all_ids, all_names])
print(f"合并后形状: {merged.shape}")

# ==========================================
# 清洗
# ==========================================
print("\n清洗...")

# 1. 转为float32节省内存
print("转换为float32...")
merged = merged.astype(np.float32)

# 2. 删除全部为NaN的列
print("删除全NaN列...")
all_nan_mask = merged.isna().all(axis=0)
n_before = merged.shape[1]
merged = merged.loc[:, ~all_nan_mask]
print(f"  删除了 {n_before - merged.shape[1]} 列全NaN")

# 3. 删除NaN比例 > 50% 的列
print("删除NaN比例>50%的列...")
nan_ratio = merged.isna().mean(axis=0)
high_nan_mask = nan_ratio > 0.5
n_before = merged.shape[1]
merged = merged.loc[:, ~high_nan_mask]
print(f"  删除了 {n_before - merged.shape[1]} 列 (NaN > 50%)")

# 4. 删除常数列 (std = 0)
print("删除常数列...")
std = merged.std(axis=0)
const_mask = std < 1e-10
n_before = merged.shape[1]
merged = merged.loc[:, ~const_mask]
print(f"  删除了 {n_before - merged.shape[1]} 列常数")

# 5. 前向填充 + 后向填充
print("前向填充NaN...")
merged = merged.ffill()
print("后向填充剩余NaN...")
merged = merged.bfill()

# 验证无NaN
remaining_nan = merged.isna().sum().sum()
print(f"剩余NaN数: {remaining_nan}")

# 从 MultiIndex 提取最终的列 ID 和列名
final_ids = list(merged.columns.get_level_values(0))
final_names = list(merged.columns.get_level_values(1))

# ==========================================
# 统计
# ==========================================
print("\n" + "=" * 80)
print("清洗统计")
print("=" * 80)
print(f"合并前: {total_cols_before} 列")
print(f"合并后: {merged.shape[1]} 列")
print(f"删除: {total_cols_before - merged.shape[1]} 列")
print(f"数据行数: {n}")

# ==========================================
# 保存
# ==========================================
print("\n保存合并后的特征矩阵...")
output_path = os.path.join(output_dir, '08_features_merged.csv')

with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['data_date'] + final_ids)
    writer.writerow(['data_date'] + final_names)
    for i in range(n):
        if i % 500 == 0:
            print(f"  写入行 {i}/{n}...")
        date_str = date_col.iloc[i]
        row = [date_str]
        for j in range(merged.shape[1]):
            val = merged.iloc[i, j]
            if pd.isna(val):
                row.append('')
            else:
                row.append(f"{val:.6f}")
        writer.writerow(row)

fsize_mb = os.path.getsize(output_path) / (1024 * 1024)
print(f"\n已保存: {os.path.basename(output_path)} ({fsize_mb:.1f} MB)")

# ==========================================
# 按来源统计剩余列数
# ==========================================
print("\n特征来源分布:")
cumsum = 0
for fname in feature_files:
    fpath = os.path.join(output_dir, fname)
    if os.path.exists(fpath):
        df = pd.read_csv(fpath, header=[0, 1], encoding='utf-8-sig', nrows=0)
        orig_cols = len(df.columns) - 1
        # 估算剩余: 按删除比例缩放到各文件
        scale = merged.shape[1] / total_cols_before
        remaining = int(orig_cols * scale)
        print(f"  {fname:<35s}: {orig_cols:>5d} → ~{remaining:>5d} 列")

print("\n" + "=" * 80)
print("Step 9 完成！")
print("=" * 80)

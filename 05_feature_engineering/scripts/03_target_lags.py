import pandas as pd
import numpy as np
import os
import csv

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
input_path = os.path.join(base_dir, '03_data_cleaning', 'outputs', 'cleaned_data_daily.csv')
mapping_path = os.path.join(base_dir, '05_feature_engineering', 'outputs', 'feature_group_mapping.csv')
output_dir = os.path.join(base_dir, '05_feature_engineering', 'outputs')
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("Step 3: 标的自身滞后特征")
print("=" * 80)

# 读取数据
print("\n读取数据...")
df = pd.read_csv(input_path, header=[0, 1], encoding='utf-8-sig')
date_col = df.iloc[:, 0]
col_ids = list(df.columns.get_level_values(0))[1:]
col_names = list(df.columns.get_level_values(1))[1:]

# 读取分组映射，找标的列
print("读取分组映射...")
mapping_df = pd.read_csv(mapping_path, encoding='utf-8-sig')
target_rows = mapping_df[mapping_df['分组'] == '标的(Y变量)']

# 优先选主标的：45个港口总库存
primary_target_id = 'ID00186052'
primary_target_name = '铁矿石：进口：库存：45个港口（周度）'

# 查找该列在values中的位置
target_idx = None
for idx, cid in enumerate(col_ids):
    if cid == primary_target_id:
        target_idx = idx
        break

if target_idx is None:
    print(f"未找到主标的列 {primary_target_id}，使用分组中第一个核心因子")
    core_targets = target_rows[target_rows['是否核心因子'] == '是']
    if len(core_targets) > 0:
        primary_target_id = core_targets.iloc[0]['列ID']
        primary_target_name = core_targets.iloc[0]['列名']
        for idx, cid in enumerate(col_ids):
            if cid == primary_target_id:
                target_idx = idx
                break

print(f"标的列: {primary_target_id}")
print(f"标的名称: {primary_target_name[:60]}")

values_df = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
target_data = values_df.iloc[:, target_idx].values.astype(np.float64)
n = len(target_data)

# ==========================================
# 生成滞后特征
# ==========================================
print("\n生成滞后特征...")

features = {}
feature_ids = []
feature_names = []
prefix = 'target'

# 短期滞后
short_lags = [1, 2, 3, 5, 7]
for lag in short_lags:
    fid = f'{prefix}_lag_{lag}'
    arr = np.full(n, np.nan, dtype=np.float32)
    arr[lag:] = target_data[:-lag]
    features[fid] = arr
    feature_ids.append(fid)
    feature_names.append(f'标的滞后{lag}日')

# 中期滞后
mid_lags = [14, 21, 30]
for lag in mid_lags:
    fid = f'{prefix}_lag_{lag}'
    arr = np.full(n, np.nan, dtype=np.float32)
    arr[lag:] = target_data[:-lag]
    features[fid] = arr
    feature_ids.append(fid)
    feature_names.append(f'标的滞后{lag}日')

# 长期滞后
long_lags = [60, 90, 180, 365]
for lag in long_lags:
    fid = f'{prefix}_lag_{lag}'
    arr = np.full(n, np.nan, dtype=np.float32)
    arr[lag:] = target_data[:-lag]
    features[fid] = arr
    feature_ids.append(fid)
    feature_names.append(f'标的滞后{lag}日')

# 差分特征
for lag in [1, 7]:
    fid = f'{prefix}_diff_{lag}'
    diff_arr = np.full(n, np.nan, dtype=np.float32)
    diff_arr[lag:] = target_data[lag:] - target_data[:-lag]
    features[fid] = diff_arr
    feature_ids.append(fid)
    feature_names.append(f'标的差分(lag={lag})')

# 变化率
for lag in [1, 7]:
    fid = f'{prefix}_pct_{lag}'
    pct_arr = np.full(n, np.nan, dtype=np.float32)
    valid = target_data[:-lag] != 0
    ratio = np.full(n - lag, np.nan, dtype=np.float32)
    safe_denom = np.where(np.abs(target_data[:-lag]) > 1e-10, target_data[:-lag], np.nan)
    ratio = (target_data[lag:] - target_data[:-lag]) / safe_denom
    pct_arr[lag:] = ratio
    features[fid] = pct_arr
    feature_ids.append(fid)
    feature_names.append(f'标的变化率(lag={lag})')

print(f"生成特征数: {len(feature_ids)}")

# ==========================================
# 统计数据覆盖率
# ==========================================
print("\n特征覆盖率:")
for fid in feature_ids:
    valid_count = np.sum(~np.isnan(features[fid]))
    coverage = valid_count / n * 100
    print(f"  {fid:<25s}: {valid_count:>5d}/{n} ({coverage:.1f}%)")

# ==========================================
# 保存
# ==========================================
print("\n保存...")

output_path = os.path.join(output_dir, '02_target_lags.csv')
with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['data_date'] + feature_ids)
    writer.writerow(['data_date'] + feature_names)
    for i in range(n):
        date_str = date_col.iloc[i]
        row = [date_str]
        for fid in feature_ids:
            val = features[fid][i]
            if np.isnan(val):
                row.append('')
            else:
                row.append(f"{val:.6f}")
        writer.writerow(row)

fsize = os.path.getsize(output_path) / 1024
print(f"已保存: {os.path.basename(output_path)} ({fsize:.1f} KB)")

print("\n" + "=" * 80)
print("Step 3 完成！")
print("=" * 80)

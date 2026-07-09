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
print("Step 8: 同比环比特征")
print("=" * 80)

# 读取数据
print("\n读取数据...")
df = pd.read_csv(input_path, header=[0, 1], encoding='utf-8-sig')
date_col = df.iloc[:, 0]
col_ids = list(df.columns.get_level_values(0))[1:]
col_names = list(df.columns.get_level_values(1))[1:]
values_df = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
n = len(date_col)

# 读取分组
mapping_df = pd.read_csv(mapping_path, encoding='utf-8-sig')
core_ids = set(mapping_df[mapping_df['是否核心因子'] == '是']['列ID'].values)

# 构建索引
col_idx_map = {}
for idx, cid in enumerate(col_ids):
    col_idx_map[cid] = idx

# 标的列 + 核心因子
target_id = 'ID00186052'
process_ids = [target_id] + sorted(core_ids - {target_id})
print(f"目标列数: {len(process_ids)}")


def compute_yoy_mom(data):
    """计算同比环比特征"""
    results = []

    # YoY: value(t) / value(t-365) - 1
    yoy = np.full(n, np.nan, dtype=np.float32)
    denom = np.abs(data[:-365])
    safe = np.where(denom > 1e-10, denom, np.nan)
    yoy[365:] = (data[365:] - data[:-365]) / safe
    results.append(('yoy_365', yoy))

    # MoM: value(t) / value(t-30) - 1
    mom_30 = np.full(n, np.nan, dtype=np.float32)
    denom_30 = np.abs(data[:-30])
    safe_30 = np.where(denom_30 > 1e-10, denom_30, np.nan)
    mom_30[30:] = (data[30:] - data[:-30]) / safe_30
    results.append(('mom_30', mom_30))

    # MoM: value(t) / value(t-90) - 1
    mom_90 = np.full(n, np.nan, dtype=np.float32)
    denom_90 = np.abs(data[:-90])
    safe_90 = np.where(denom_90 > 1e-10, denom_90, np.nan)
    mom_90[90:] = (data[90:] - data[:-90]) / safe_90
    results.append(('mom_90', mom_90))

    # WoW: value(t) / value(t-7) - 1
    wow_7 = np.full(n, np.nan, dtype=np.float32)
    denom_7 = np.abs(data[:-7])
    safe_7 = np.where(denom_7 > 1e-10, denom_7, np.nan)
    wow_7[7:] = (data[7:] - data[:-7]) / safe_7
    results.append(('wow_7', wow_7))

    # WoW: value(t) / value(t-14) - 1
    wow_14 = np.full(n, np.nan, dtype=np.float32)
    denom_14 = np.abs(data[:-14])
    safe_14 = np.where(denom_14 > 1e-10, denom_14, np.nan)
    wow_14[14:] = (data[14:] - data[:-14]) / safe_14
    results.append(('wow_14', wow_14))

    return results


# ==========================================
# 生成同比环比特征
# ==========================================
print("\n生成同比环比特征...")

feature_ids = []
feature_names = []
feature_arrays = []

for i, cid in enumerate(process_ids):
    if cid not in col_idx_map:
        continue
    idx = col_idx_map[cid]
    data = values_df.iloc[:, idx].values.astype(np.float64)
    cname = col_names[idx][:35]

    ratios = compute_yoy_mom(data)
    for suffix, arr in ratios:
        fid = f'{cid}_{suffix}'
        fname = f'{cname}_{suffix}'
        feature_ids.append(fid)
        feature_names.append(fname)
        feature_arrays.append(arr)

    if (i + 1) % 50 == 0:
        print(f"  已处理 {i+1}/{len(process_ids)} 列, 当前特征数: {len(feature_ids)}")

print(f"总特征数: {len(feature_ids)}")

# ==========================================
# 保存
# ==========================================
print("\n保存...")
output_path = os.path.join(output_dir, '07_yoy_mom_features.csv')
with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['data_date'] + feature_ids)
    writer.writerow(['data_date'] + feature_names)
    for i in range(n):
        if i % 500 == 0:
            print(f"  写入行 {i}/{n}...")
        date_str = date_col.iloc[i]
        row = [date_str]
        for arr in feature_arrays:
            val = arr[i]
            if np.isnan(val):
                row.append('')
            else:
                row.append(f"{val:.6f}")
        writer.writerow(row)

fsize_mb = os.path.getsize(output_path) / (1024 * 1024)
print(f"已保存: {os.path.basename(output_path)} ({fsize_mb:.1f} MB)")

print("\n" + "=" * 80)
print("Step 8 完成！")
print("=" * 80)

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
print("Step 4: 因子基础滞后特征")
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

# 分级
core_ids = set()
important_ids = set()
basic_ids = set()

important_groups = ['供应-发运', '供应-发货量', '供应-到港', '供应-产量', '供应-矿山',
                    '需求-疏港', '需求-开工率', '需求-生铁粗钢', '需求-消费',
                    '价格-矿石价格', '价格-价差利润', '价格-关联商品', '价格-海运',
                    '宏观-PMI', '宏观-投资', '宏观-货币', '宏观-GDP', '宏观-地产', '宏观-基建',
                    '供需-海漂压港', '供需-配比']

for _, row in mapping_df.iterrows():
    cid = row['列ID']
    is_core = row['是否核心因子'] == '是'
    group = row['分组']

    if is_core:
        core_ids.add(cid)
    elif group in important_groups:
        important_ids.add(cid)
    else:
        basic_ids.add(cid)

print(f"核心因子: {len(core_ids)} 列 (每列10特征 = {len(core_ids)*10})")
print(f"重要因子: {len(important_ids)} 列 (每列3特征 = {len(important_ids)*3})")
print(f"基础因子: {len(basic_ids)} 列 (每列1特征 = {len(basic_ids)*1})")
total_features = len(core_ids) * 10 + len(important_ids) * 3 + len(basic_ids) * 1
print(f"预估总特征数: {total_features}")

# ==========================================
# 构建列索引映射
# ==========================================
col_idx_map = {}
for idx, cid in enumerate(col_ids):
    col_idx_map[cid] = idx

# ==========================================
# 生成滞后特征
# ==========================================
feature_ids = []
feature_names = []
feature_arrays = []

core_lags = [1, 2, 3, 7, 14, 30]
core_extra = [('diff_1', 1, 'diff'), ('diff_7', 7, 'diff'), ('pct_1', 1, 'pct'), ('pct_7', 7, 'pct')]


def make_lag(data, lag):
    arr = np.full(n, np.nan, dtype=np.float32)
    arr[lag:] = data[:-lag]
    return arr


def make_diff(data, lag):
    arr = np.full(n, np.nan, dtype=np.float32)
    arr[lag:] = data[lag:] - data[:-lag]
    return arr


def make_pct(data, lag):
    arr = np.full(n, np.nan, dtype=np.float32)
    denom = np.abs(data[:-lag])
    safe = np.where(denom > 1e-10, denom, np.nan)
    ratio = (data[lag:] - data[:-lag]) / safe
    arr[lag:] = ratio
    return arr


def process_column(cid, cname, tier):
    """处理单列，生成滞后特征"""
    idx = col_idx_map[cid]
    data = values_df.iloc[:, idx].values.astype(np.float64)

    cname_short = cname[:40]

    if tier == 'core':
        for lag in core_lags:
            fid = f'{cid}_lag_{lag}'
            fname = f'{cname_short}_lag{lag}'
            feature_ids.append(fid)
            feature_names.append(fname)
            feature_arrays.append(make_lag(data, lag))

        for suffix, lag, ftype in core_extra:
            fid = f'{cid}_{suffix}'
            fname = f'{cname_short}_{suffix}'
            feature_ids.append(fid)
            feature_names.append(fname)
            if ftype == 'diff':
                feature_arrays.append(make_diff(data, lag))
            else:
                feature_arrays.append(make_pct(data, lag))

    elif tier == 'important':
        for lag in [1, 7, 30]:
            fid = f'{cid}_lag_{lag}'
            fname = f'{cname_short}_lag{lag}'
            feature_ids.append(fid)
            feature_names.append(fname)
            feature_arrays.append(make_lag(data, lag))

    else:  # basic
        fid = f'{cid}_lag_1'
        fname = f'{cname_short}_lag1'
        feature_ids.append(fid)
        feature_names.append(fname)
        feature_arrays.append(make_lag(data, 1))


# 处理所有列
print("\n生成滞后特征...")

all_cols = [(cid, 'core') for cid in core_ids] + \
           [(cid, 'important') for cid in important_ids] + \
           [(cid, 'basic') for cid in basic_ids]

for i, (cid, tier) in enumerate(all_cols):
    cname = col_names[col_idx_map[cid]]
    process_column(cid, cname, tier)
    if (i + 1) % 100 == 0:
        print(f"  已处理 {i+1}/{len(all_cols)} 列, 当前特征数: {len(feature_ids)}")

print(f"\n实际生成特征数: {len(feature_ids)}")

# ==========================================
# 保存
# ==========================================
print("\n保存...")

output_path = os.path.join(output_dir, '03_factor_lags.csv')
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

# 统计
nan_ratios = []
for arr in feature_arrays:
    nan_ratios.append(np.isnan(arr).sum() / n)

print(f"NaN比例: min={min(nan_ratios)*100:.1f}%, mean={np.mean(nan_ratios)*100:.1f}%, max={max(nan_ratios)*100:.1f}%")

print("\n" + "=" * 80)
print("Step 4 完成！")
print("=" * 80)

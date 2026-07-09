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
print("Step 5: 趋势与波动特征")
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

# 核心因子 + 标的列
target_id = 'ID00186052'
target_idx = col_idx_map[target_id]

# 收集要处理的列
process_cols = [(target_id, '标的(Y)')]
for cid in sorted(core_ids):
    if cid != target_id:
        process_cols.append((cid, '核心因子'))

print(f"目标列数: {len(process_cols)}")

# Rolling windows
ma_windows = [3, 5, 7, 14, 30, 60, 90]
std_windows = [7, 14, 30, 60]
mom_windows = [5, 10, 20]


def compute_rolling_stats(series, col_id, col_name_short):
    """对单个序列计算所有趋势波动特征"""
    results = []
    s = pd.Series(series)

    # 移动平均
    for w in ma_windows:
        ma = s.rolling(window=w, min_periods=max(1, w // 2)).mean().values.astype(np.float32)
        results.append((f'{col_id}_MA{w}', f'{col_name_short[:30]}_MA{w}', ma))

    # 移动标准差
    for w in std_windows:
        std = s.rolling(window=w, min_periods=max(1, w // 2)).std().values.astype(np.float32)
        results.append((f'{col_id}_STD{w}', f'{col_name_short[:30]}_STD{w}', std))

    # 移动分位数 (P10, P90)
    for w in [30, 60, 90]:
        p10 = s.rolling(window=w, min_periods=max(1, w // 2)).quantile(0.1).values.astype(np.float32)
        p90 = s.rolling(window=w, min_periods=max(1, w // 2)).quantile(0.9).values.astype(np.float32)
        spread = p90 - p10
        results.append((f'{col_id}_P10_{w}d', f'{col_name_short[:30]}_P10_{w}d', p10))
        results.append((f'{col_id}_P90_{w}d', f'{col_name_short[:30]}_P90_{w}d', p90))
        results.append((f'{col_id}_Spread_{w}d', f'{col_name_short[:30]}_Spread_{w}d', spread))

    # 趋势指标: Value/MA7, Value/MA30
    ma7 = s.rolling(window=7, min_periods=3).mean().values
    ma30 = s.rolling(window=30, min_periods=10).mean().values
    ratio7 = np.where(np.abs(ma7) > 1e-10, series / ma7, np.nan).astype(np.float32)
    ratio30 = np.where(np.abs(ma30) > 1e-10, series / ma30, np.nan).astype(np.float32)
    ma7_ma30 = np.where(np.abs(ma30) > 1e-10, ma7 / ma30, np.nan).astype(np.float32)

    results.append((f'{col_id}_Ratio_MA7', f'{col_name_short[:30]}_Ratio_MA7', ratio7))
    results.append((f'{col_id}_Ratio_MA30', f'{col_name_short[:30]}_Ratio_MA30', ratio30))
    results.append((f'{col_id}_MA7_MA30', f'{col_name_short[:30]}_MA7_MA30', ma7_ma30))

    # 动量指标
    for w in mom_windows:
        mom = np.full(n, np.nan, dtype=np.float32)
        denom = np.abs(series[:-w])
        safe = np.where(denom > 1e-10, denom, np.nan)
        mom[w:] = (series[w:] - series[:-w]) / safe
        results.append((f'{col_id}_MOM{w}', f'{col_name_short[:30]}_MOM{w}', mom.astype(np.float32)))

    return results


# ==========================================
# 生成所有趋势波动特征
# ==========================================
print("\n生成趋势波动特征...")

feature_ids = []
feature_names = []
feature_arrays = []

for i, (cid, tier) in enumerate(process_cols):
    idx = col_idx_map[cid]
    data = values_df.iloc[:, idx].values.astype(np.float64)
    cname = col_names[idx][:40]

    col_features = compute_rolling_stats(data, cid, cname)
    for fid, fname, arr in col_features:
        feature_ids.append(fid)
        feature_names.append(fname)
        feature_arrays.append(arr)

    if (i + 1) % 50 == 0:
        print(f"  已处理 {i+1}/{len(process_cols)} 列, 当前特征数: {len(feature_ids)}")

per_col = len(compute_rolling_stats(values_df.iloc[:, target_idx].values.astype(np.float64), 'X', 'X'))
print(f"\n每列生成: {per_col} 个特征")
print(f"总特征数: {len(feature_ids)}")

# ==========================================
# 保存
# ==========================================
print("\n保存...")
output_path = os.path.join(output_dir, '04_trend_volatility.csv')
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
print("Step 5 完成！")
print("=" * 80)

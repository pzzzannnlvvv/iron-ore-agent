"""
Step 10: 特征重要性分析与最终选择
===================================
从 7112 列精简到 3000-3500 列。

筛选策略（三层漏斗）：
1. 保护层：时间特征 + 目标特征 + 库存相关因子 → 全部保留
2. 组内去重：按基础因子ID分组，组内 |corr| > 0.98 的只保留方差最高者
3. 全局去重：跨组贪心去重（按方差降序，与已保留列比相关）

输出：
    outputs/09_features_final.csv
    outputs/final_feature_list.csv
"""

import csv
import os
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_PATH = os.path.join(BASE_DIR, "05_feature_engineering", "outputs", "08_features_merged.csv")
MAPPING_PATH = os.path.join(BASE_DIR, "05_feature_engineering", "outputs", "feature_group_mapping.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "05_feature_engineering", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_COLS = 3200
CORR_THRESHOLD = 0.98
MIN_VARIANCE = 1e-8

TIME_KEYWORDS = ["year", "month", "day", "quarter", "spring_", "heating"]

# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 70)
print("Step 10: 特征重要性分析与最终选择")
print("=" * 70)

print("\n[1/5] 加载合并特征...")
t0 = time.time()

merged = pd.read_csv(INPUT_PATH, header=[0, 1], encoding="utf-8-sig", low_memory=False)
col_ids = list(merged.columns.get_level_values(0))[1:]
col_names = list(merged.columns.get_level_values(1))[1:]
date_col = merged.iloc[:, 0]

print(f"  总列数: {len(col_ids)}, 行数: {len(merged)}")

print("  转换为数值矩阵...")
values_df = merged.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").astype(np.float32)
values_df = values_df.ffill().bfill()
V = values_df.values  # (3788, 7112)
n_rows, n_cols = V.shape
print(f"  矩阵: {n_rows} 行 x {n_cols} 列")

# ============================================================
# 2. 标记每列类型 & 受保护列
# ============================================================
print("\n[2/5] 识别受保护特征...")

PROTECTED = set()
col_type = [""] * n_cols  # 'time', 'target', 'inventory', 'factor'

for i, cid in enumerate(col_ids):
    # 目标列
    if cid.startswith("target_"):
        PROTECTED.add(i)
        col_type[i] = "target"
        continue
    # 时间列
    if any(cid.startswith(kw) for kw in TIME_KEYWORDS):
        PROTECTED.add(i)
        col_type[i] = "time"
        continue
    # 库存相关（库存_base ID 的衍生列）
    INVENTORY_BASES = {"ID00186052", "ID00186053", "ID00186054", "ID00186055",
                       "ID00186056", "ID00186057", "ID00186058", "ID00186059", "ID00186060"}
    base = cid.split("_")[0]
    if base in INVENTORY_BASES:
        PROTECTED.add(i)
        col_type[i] = "inventory"
        continue
    col_type[i] = "factor"

print(f"  受保护列: {len(PROTECTED)} (目标+时间+库存)")

# ============================================================
# 3. 提取基础因子ID（用于分组）
# ============================================================
def extract_base_id(cid):
    """提取基础因子ID。例: ID00186052_lag7 → ID00186052"""
    parts = cid.split("_")
    if parts[0].startswith("ID"):
        return parts[0]
    # 对于特殊命名的列（如 IO_SHIP_VOL_GLOB_016_year_pct_7），
    # 取前缀部分作为base
    # 移除常见的后缀模式
    suffixes = ["_lag", "_MA", "_STD", "_P", "_MOM", "_yoy", "_mom",
                "_wow", "_pct", "_diff", "_ratio", "_RSI", "_MACD",
                "_ATR", "_BB", "_BBW", "_SMA", "_EMA"]
    base = cid
    for suf in suffixes:
        idx = base.find(suf)
        if idx > 0:
            base = base[:idx]
            break
    return base

print("\n[3/5] 构建因子分组...")
factor_groups = defaultdict(list)  # base_id → [column_indices]
for i in range(n_cols):
    if col_type[i] == "factor":
        base = extract_base_id(col_ids[i])
        factor_groups[base].append(i)

n_groups = len(factor_groups)
group_sizes = [len(g) for g in factor_groups.values()]
print(f"  因子分组数: {n_groups}")
print(f"  每组列数: min={min(group_sizes)}, max={max(group_sizes)}, "
      f"avg={np.mean(group_sizes):.1f}, median={np.median(group_sizes):.0f}")

# ============================================================
# 4. 低方差过滤 + 组内去重
# ============================================================
print(f"\n[4/5] 筛选 (阈值: corr>{CORR_THRESHOLD}, var>{MIN_VARIANCE})...")

var_all = np.var(V, axis=0)
selected = set(PROTECTED)

def dedup_group(indices, corr_threshold=CORR_THRESHOLD):
    """组内去重：贪心保留高方差列"""
    if len(indices) <= 1:
        return set(indices)

    # 按方差降序
    vars_ = var_all[indices]
    order = np.argsort(-vars_)
    sorted_idx = [indices[i] for i in order]

    kept = set()
    for i in sorted_idx:
        redundant = False
        for j in kept:
            c = np.corrcoef(V[:, i], V[:, j])[0, 1]
            if abs(c) > corr_threshold:
                redundant = True
                break
        if not redundant:
            kept.add(i)
    return kept


# 处理每个因子组
n_processed = 0
for base, indices in factor_groups.items():
    # 过滤低方差
    valid = [i for i in indices if var_all[i] >= MIN_VARIANCE]
    if not valid:
        continue
    kept = dedup_group(valid)
    selected.update(kept)

    n_processed += 1
    if n_processed % 500 == 0:
        elapsed = time.time() - t0
        print(f"  处理 {n_processed}/{n_groups} 组, 已选 {len(selected)} 列, 耗时 {elapsed:.0f}s")

print(f"\n  组内去重后: {len(selected)} 列")

# ============================================================
# 5. LightGBM 重要性截断（如需）
# ============================================================
selected_list = sorted(selected)
if len(selected_list) <= TARGET_COLS + 500:
    print(f"\n[5/5] 跳过 LightGBM 截断（{len(selected_list)} 列已达目标 {TARGET_COLS}±500）")
else:
    print(f"\n[5/5] LightGBM 重要性截断 ({len(selected_list)} → ~{TARGET_COLS})...")

    # 只用候选因子训练
    candidate = [i for i in selected_list if col_type[i] == "factor"]
    X_sub = V[:, candidate]

    # 目标: target_lag_1 的下一日值
    target_col_idx = None
    for i, cid in enumerate(col_ids):
        if cid == "target_lag_1":
            target_col_idx = i
            break
    y_raw = V[:, target_col_idx]
    y = np.full(n_rows, np.nan, dtype=np.float32)
    y[:-1] = y_raw[1:]
    mask = ~np.isnan(y)

    model_params = {
        "n_estimators": 200, "num_leaves": 31, "learning_rate": 0.1,
        "random_state": 42, "deterministic": True, "verbosity": -1, "n_jobs": -1,
    }

    import lightgbm as lgb
    model = lgb.LGBMRegressor(**model_params)
    model.fit(X_sub[mask], y[mask])

    importances = model.feature_importances_
    imp_order = np.argsort(-importances)

    # 保留 top N 个因子
    n_keep_factor = TARGET_COLS - len(PROTECTED)
    kept_factor = set(candidate[i] for i in imp_order[:n_keep_factor])

    selected_list = sorted(PROTECTED | kept_factor)
    print(f"  最终选择: {len(selected_list)} 列")

# ============================================================
# 保存
# ============================================================
print("\n保存结果...")
final_values = V[:, selected_list]
final_ids = [col_ids[i] for i in selected_list]
final_names = [col_names[i] for i in selected_list]

# 统计
n_target = sum(1 for i in selected_list if col_type[i] == "target")
n_time = sum(1 for i in selected_list if col_type[i] == "time")
n_inventory = sum(1 for i in selected_list if col_type[i] == "inventory")
n_factor = sum(1 for i in selected_list if col_type[i] == "factor")
print(f"  目标特征: {n_target}, 时间特征: {n_time}, "
      f"库存特征: {n_inventory}, 因子: {n_factor}")

# 保存 CSV
final_path = os.path.join(OUTPUT_DIR, "09_features_final.csv")
with open(final_path, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["data_date"] + final_ids)
    writer.writerow(["data_date"] + final_names)
    for row_idx in range(n_rows):
        if row_idx % 500 == 0:
            print(f"  写入行 {row_idx}/{n_rows}...")
        row = [date_col.iloc[row_idx]]
        for j in range(len(selected_list)):
            row.append(f"{final_values[row_idx, j]:.6f}")
        writer.writerow(row)

fsize_mb = os.path.getsize(final_path) / (1024 * 1024)
print(f"已保存: 09_features_final.csv ({fsize_mb:.1f} MB)")

# 保存特征列表
list_path = os.path.join(OUTPUT_DIR, "final_feature_list.csv")
with open(list_path, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["列ID", "列名", "类型"])
    for i in selected_list:
        writer.writerow([col_ids[i], col_names[i], col_type[i]])

elapsed = time.time() - t0
print(f"\n特征选择完成: {n_cols} → {len(selected_list)} 列, 耗时 {elapsed:.0f}s")
print(f"  筛选率: {len(selected_list)/n_cols*100:.1f}%")

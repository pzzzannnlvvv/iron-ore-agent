import pandas as pd
import numpy as np
import os
import csv

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
input_path = os.path.join(base_dir, '03_data_cleaning', 'outputs', 'cleaned_data_daily.csv')
output_dir = os.path.join(base_dir, '05_feature_engineering', 'outputs')
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("Step 6: 业务逻辑衍生特征")
print("=" * 80)

print("\n读取数据...")
df = pd.read_csv(input_path, header=[0, 1], encoding='utf-8-sig')
date_col = df.iloc[:, 0]
col_ids = list(df.columns.get_level_values(0))[1:]
col_names = list(df.columns.get_level_values(1))[1:]
values_df = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
n = len(date_col)


def find_col(keywords, require_all=False):
    """查找列索引。require_all=True时所有关键词都要有，否则任一即可"""
    results = []
    for idx, name in enumerate(col_names):
        if require_all:
            if all(kw in name for kw in keywords):
                results.append((idx, col_ids[idx], name))
        else:
            if any(kw in name for kw in keywords):
                results.append((idx, col_ids[idx], name))
    return results


def safe_ratio(a, b):
    arr = np.full(n, np.nan, dtype=np.float32)
    denom = np.abs(b)
    mask = denom > 1e-10
    arr[mask] = a[mask] / b[mask]
    return arr


def get_data(idx_list):
    """从索引列表获取第一个有效列的数据"""
    if idx_list:
        return values_df.iloc[:, idx_list[0][0]].values.astype(np.float64)
    return None


feature_ids = []
feature_names = []
feature_arrays = []


def add_feature(fid, fname, arr):
    feature_ids.append(fid)
    feature_names.append(fname)
    feature_arrays.append(arr.astype(np.float32))


print("\n计算业务衍生特征...")

# ===== 1. 四大矿山发运特征 =====
rio_idx = find_col(['力拓', '发运量'], require_all=True)
bhp_idx = find_col(['必和必拓', '发运量'], require_all=True)
fmg_idx = find_col(['福蒂斯丘', '发运量'], require_all=True)
vale_idx = find_col(['淡水河谷', '发运量'], require_all=True)
# 如果日度没找到，尝试不限定日度
if not rio_idx:
    rio_idx = find_col(['力拓', '发运量'], require_all=True)
if not bhp_idx:
    bhp_idx = find_col(['必和必拓', '发运量'], require_all=True)

all_ship_data = []
for label, idx_list in [('Rio', rio_idx), ('BHP', bhp_idx), ('FMG', fmg_idx), ('Vale', vale_idx)]:
    data = get_data(idx_list)
    if data is not None:
        all_ship_data.append((label, data))
        print(f"  {label}: {idx_list[0][2][:50]}")

# 四大矿山总量
if all_ship_data:
    total = np.sum([d for _, d in all_ship_data], axis=0)
    add_feature('biz_top4_ship_total', '四大矿山发运总量', total)

    # 各矿山占比
    for label, data in all_ship_data:
        add_feature(f'biz_{label}_ratio', f'{label}占四大矿山比', safe_ratio(data, total))

# ===== 2. 供需平衡特征 =====
# 库存/疏港量(库存周转天数)
inv_45 = find_col(['45个港口', '库存'])
discharge_45 = find_col(['疏港量', '45个港口'])

inv_data = get_data(inv_45)
discharge_data = get_data(discharge_45)

if inv_data is not None and discharge_data is not None:
    add_feature('biz_inv_discharge_ratio', '库存_疏港量比率', safe_ratio(inv_data, discharge_data))

# ===== 3. 产能利用率变化 =====
util_idx = find_col(['产能利用率'])
util_data = get_data(util_idx)
if util_data is not None:
    print(f"  产能利用率: {util_idx[0][2][:50]}")
    util_diff = np.full(n, np.nan, dtype=np.float32)
    util_diff[1:] = util_data[1:] - util_data[:-1]
    add_feature('biz_util_change', '产能利用率日度变化', util_diff)

    util_ma7 = pd.Series(util_data).rolling(7, min_periods=3).mean().values.astype(np.float32)
    util_ma30 = pd.Series(util_data).rolling(30, min_periods=10).mean().values.astype(np.float32)
    add_feature('biz_util_MA7', '产能利用率MA7', util_ma7)
    add_feature('biz_util_MA30', '产能利用率MA30', util_ma30)
    # 趋势方向
    util_ratio = np.where(np.abs(util_ma30) > 1e-10, util_ma7 / util_ma30, np.nan).astype(np.float32)
    add_feature('biz_util_trend', '产能利用率趋势', util_ratio)

# ===== 4. 价格侧特征 =====
# 铁矿石价格: 优先新交所掉期，其次品牌价格
ore_price_idx = find_col(['结算价'])
if not ore_price_idx:
    ore_price_idx = find_col(['品牌价格'])

ore_price_data = get_data(ore_price_idx)
if ore_price_data is not None:
    print(f"  铁矿石价格: {ore_price_idx[0][2][:50]}")
    # 日度收益率
    ore_ret = np.full(n, np.nan, dtype=np.float32)
    denom = np.abs(ore_price_data[:-1])
    safe = np.where(denom > 1e-10, denom, np.nan)
    ore_ret[1:] = (ore_price_data[1:] - ore_price_data[:-1]) / safe
    add_feature('biz_ore_daily_return', '铁矿石日度收益率', ore_ret)

    # 波动率 MA30
    ret_series = pd.Series(ore_ret)
    ore_vol = ret_series.rolling(30, min_periods=10).std().values.astype(np.float32)
    add_feature('biz_ore_vol_30d', '铁矿石30日波动率', ore_vol)

# 普钢价格指数
steel_idx = find_col(['普钢', '价格指数'])
steel_data = get_data(steel_idx)
if steel_data is not None and ore_price_data is not None:
    print(f"  普钢指数: {steel_idx[0][2][:50]}")
    spread = (steel_data - ore_price_data).astype(np.float32)
    add_feature('biz_steel_ore_spread', '普钢_铁矿石价差', spread)

    # 钢材/矿石比价
    add_feature('biz_steel_ore_ratio', '普钢_铁矿石比价', safe_ratio(steel_data, ore_price_data))

# 高炉现金利润
profit_idx = find_col(['现金利润'])
profit_data = get_data(profit_idx)
if profit_data is not None:
    print(f"  现金利润: {profit_idx[0][2][:50]}")
    # 利润变化
    profit_diff = np.full(n, np.nan, dtype=np.float32)
    profit_diff[1:] = profit_data[1:] - profit_data[:-1]
    add_feature('biz_profit_change', '高炉利润日度变化', profit_diff)

    # 利润MA
    profit_ma7 = pd.Series(profit_data).rolling(7, min_periods=3).mean().values.astype(np.float32)
    add_feature('biz_profit_MA7', '高炉利润MA7', profit_ma7)

# ===== 5. 废钢相关特征 =====
scrap_spread = find_col(['废钢', '价差'])
scrap_data = get_data(scrap_spread)
if scrap_data is not None:
    print(f"  废钢价差: {scrap_spread[0][2][:50]}")
    add_feature('biz_scrap_spread', '废钢铁水价差', scrap_data.astype(np.float32))

# 废钢到货
scrap_arrive = find_col(['废钢到货'])
scrap_arrive_data = get_data(scrap_arrive)
if scrap_arrive_data is not None:
    add_feature('biz_scrap_arrive', '废钢到货量', scrap_arrive_data.astype(np.float32))

# ===== 6. 港口到港总量 =====
arrival_north = find_col(['到港量', '北方港口'])
if not arrival_north:
    arrival_north = find_col(['到港量合计'])
arrival_total = get_data(arrival_north)
if arrival_total is not None:
    print(f"  到港量: {arrival_north[0][2][:50]}")
    # 到港量变化
    arrival_diff = np.full(n, np.nan, dtype=np.float32)
    arrival_diff[7:] = arrival_total[7:] - arrival_total[:-7]
    add_feature('biz_arrival_change_7d', '到港量7日变化', arrival_diff)

# ===== 7. 发货量(周度)特征 =====
ship_au = find_col(['发货量合计', '14个港口'])
ship_au_data = get_data(ship_au)
if ship_au_data is not None:
    print(f"  澳洲发货: {ship_au[0][2][:50]}")
    ship_au_diff = np.full(n, np.nan, dtype=np.float32)
    ship_au_diff[7:] = ship_au_data[7:] - ship_au_data[:-7]
    add_feature('biz_au_ship_change_7d', '澳洲发货7日变化', ship_au_diff)

# ===== 统计 =====
print(f"\n总业务特征数: {len(feature_ids)}")
for i, (fid, fname) in enumerate(zip(feature_ids, feature_names)):
    valid = np.sum(~np.isnan(feature_arrays[i]))
    print(f"  {fid:<35s} {fname:<30s} 有效: {valid}/{n} ({valid/n*100:.1f}%)")

# ===== 保存 =====
print("\n保存...")
output_path = os.path.join(output_dir, '05_business_features.csv')
with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['data_date'] + feature_ids)
    writer.writerow(['data_date'] + feature_names)
    for i in range(n):
        date_str = date_col.iloc[i]
        row = [date_str]
        for arr in feature_arrays:
            val = arr[i]
            if np.isnan(val):
                row.append('')
            else:
                row.append(f"{val:.6f}")
        writer.writerow(row)

fsize = os.path.getsize(output_path) / 1024
print(f"已保存: {os.path.basename(output_path)} ({fsize:.1f} KB)")

print("\n" + "=" * 80)
print("Step 6 完成！")
print("=" * 80)

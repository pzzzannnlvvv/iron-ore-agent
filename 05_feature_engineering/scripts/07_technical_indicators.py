import pandas as pd
import numpy as np
import os
import csv

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
input_path = os.path.join(base_dir, '03_data_cleaning', 'outputs', 'cleaned_data_daily.csv')
output_dir = os.path.join(base_dir, '05_feature_engineering', 'outputs')
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("Step 7: 技术指标特征")
print("=" * 80)

print("\n读取数据...")
df = pd.read_csv(input_path, header=[0, 1], encoding='utf-8-sig')
date_col = df.iloc[:, 0]
col_ids = list(df.columns.get_level_values(0))[1:]
col_names = list(df.columns.get_level_values(1))[1:]
values_df = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
n = len(date_col)

# 选择真正的价格列: 价格、指数、结算价，排除海漂量/可用天数/压港量/成交量
price_cols = []
exclude_kw = ['海漂量', '可用天数', '压港量', '成交量', '废钢到货']
price_kw = ['品牌价格', '价格指数', '结算价', '现货落地利润', '现金利润']

for idx, name in enumerate(col_names):
    if any(kw in name for kw in exclude_kw):
        continue
    if any(kw in name for kw in price_kw):
        price_cols.append((idx, col_ids[idx], name))

# 限制数量
price_cols = price_cols[:8]
print(f"选中的价格列: {len(price_cols)}")
for idx, cid, cname in price_cols:
    print(f"  {cid}: {cname[:60]}")


def compute_rsi(prices, period):
    rsi = np.full(n, np.nan, dtype=np.float32)
    if n < period + 1:
        return rsi
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, n):
        if i == period:
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
        else:
            avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period

        if avg_loss < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


def compute_bb_position(prices, window=20, num_std=2):
    s = pd.Series(prices)
    ma = s.rolling(window=window, min_periods=window // 2).mean().values
    std = s.rolling(window=window, min_periods=window // 2).std().values
    upper = ma + num_std * std
    lower = ma - num_std * std
    band_width = upper - lower
    position = np.full(n, np.nan, dtype=np.float32)
    mask = band_width > 1e-10
    position[mask] = (prices[mask] - lower[mask]) / band_width[mask]
    return position


def compute_macd(prices):
    s = pd.Series(prices)
    ema12 = s.ewm(span=12, adjust=False).mean().values
    ema26 = s.ewm(span=26, adjust=False).mean().values
    macd_line = ema12 - ema26
    signal = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values
    histogram = macd_line - signal
    return macd_line.astype(np.float32), signal.astype(np.float32), histogram.astype(np.float32)


feature_ids = []
feature_names = []
feature_arrays = []


def add_feature(fid, fname, arr):
    feature_ids.append(fid)
    feature_names.append(fname)
    feature_arrays.append(arr)


print("\n计算技术指标...")

for idx, cid, cname in price_cols:
    prices = values_df.iloc[:, idx].values.astype(np.float64)
    cname_short = cname[:28]

    # RSI
    for period in [7, 14]:
        rsi = compute_rsi(prices, period)
        add_feature(f'{cid}_RSI{period}', f'{cname_short}_RSI{period}', rsi)

    # 均线关系: Price/MA
    s = pd.Series(prices)
    for w in [5, 10, 20, 40, 60]:
        ma = s.rolling(window=w, min_periods=max(1, w // 2)).mean().values
        ratio = np.where(np.abs(ma) > 1e-10, prices / ma, np.nan).astype(np.float32)
        add_feature(f'{cid}_Price_MA{w}', f'{cname_short}_P/MA{w}', ratio)

    # 均线发散 MA5/MA20
    ma5 = s.rolling(window=5, min_periods=3).mean().values
    ma20 = s.rolling(window=20, min_periods=10).mean().values
    divergence = np.where(np.abs(ma20) > 1e-10, ma5 / ma20, np.nan).astype(np.float32)
    add_feature(f'{cid}_MA5_MA20', f'{cname_short}_MA5_MA20', divergence)

    # 金叉死叉标记
    crossover = np.zeros(n, dtype=np.float32)
    for i in range(5, n):
        if pd.notna(ma5[i]) and pd.notna(ma20[i]) and pd.notna(ma5[i - 1]) and pd.notna(ma20[i - 1]):
            if ma5[i - 1] <= ma20[i - 1] and ma5[i] > ma20[i]:
                crossover[i] = 1.0
            elif ma5[i - 1] >= ma20[i - 1] and ma5[i] < ma20[i]:
                crossover[i] = -1.0
    add_feature(f'{cid}_crossover', f'{cname_short}_金叉死叉', crossover)

    # 布林带位置
    bb_pos = compute_bb_position(prices)
    add_feature(f'{cid}_BB_pos', f'{cname_short}_BB位置', bb_pos)

    # MACD
    macd_line, signal, histogram = compute_macd(prices)
    add_feature(f'{cid}_MACD', f'{cname_short}_MACD', macd_line)
    add_feature(f'{cid}_MACD_sig', f'{cname_short}_MACD信号', signal)
    add_feature(f'{cid}_MACD_hist', f'{cname_short}_MACD柱', histogram)

    # 波动率
    returns = np.full(n, np.nan)
    denom = np.abs(prices[:-1])
    safe = np.where(denom > 1e-10, denom, np.nan)
    returns[1:] = (prices[1:] - prices[:-1]) / safe
    vol_14 = pd.Series(returns).rolling(14, min_periods=7).std().values.astype(np.float32)
    add_feature(f'{cid}_Vol14', f'{cname_short}_波动率14日', vol_14)

per_col = 14  # 2 RSI + 5 MA ratios + 1 divergence + 1 crossover + 1 BB + 3 MACD + 1 vol
print(f"\n每列特征数: {per_col}")
print(f"总特征数: {len(feature_ids)}")

# ===== 保存 =====
print("\n保存...")
output_path = os.path.join(output_dir, '06_technical_indicators.csv')
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
print("Step 7 完成！")
print("=" * 80)

import pandas as pd
import numpy as np
import os
import csv

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
input_path = os.path.join(base_dir, '03_data_cleaning', 'outputs', 'cleaned_data_daily.csv')
output_dir = os.path.join(base_dir, '05_feature_engineering', 'outputs')
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("Step 2: 基础时间特征")
print("=" * 80)

# 读取日期列
print("\n读取数据...")
df = pd.read_csv(input_path, header=[0, 1], encoding='utf-8-sig')
date_col = pd.to_datetime(df.iloc[:, 0])
n = len(date_col)
print(f"日期范围: {date_col.iloc[0].strftime('%Y-%m-%d')} 至 {date_col.iloc[-1].strftime('%Y-%m-%d')}")
print(f"行数: {n}")

# ==========================================
# 春节日期表 (2016-2026)
# ==========================================
spring_festival = {
    2016: '2016-02-08', 2017: '2017-01-28', 2018: '2018-02-16',
    2019: '2019-02-05', 2020: '2020-01-25', 2021: '2021-02-12',
    2022: '2022-02-01', 2023: '2023-01-22', 2024: '2024-02-10',
    2025: '2025-01-29', 2026: '2026-02-17'
}

# ==========================================
# 生成时间特征
# ==========================================
print("\n生成时间特征...")

features = {}
feature_names = []

# 基本时间特征
features['year'] = date_col.dt.year.values.astype(np.float32)
features['month'] = date_col.dt.month.values.astype(np.float32)
features['day'] = date_col.dt.day.values.astype(np.float32)
features['quarter'] = date_col.dt.quarter.values.astype(np.float32)
features['dayofweek'] = date_col.dt.dayofweek.values.astype(np.float32)
features['dayofyear'] = date_col.dt.dayofyear.values.astype(np.float32)
features['weekofyear'] = date_col.dt.isocalendar().week.values.astype(np.float32)
feature_names += ['year', 'month', 'day', 'quarter', 'dayofweek', 'dayofyear', 'weekofyear']

# 月度哑变量 (11列，避免共线性)
for m in range(1, 12):
    name = f'month_{m}'
    features[name] = (date_col.dt.month == m).astype(np.float32).values
    feature_names.append(name)

# 季度哑变量 (3列)
for q in range(1, 4):
    name = f'quarter_{q}'
    features[name] = (date_col.dt.quarter == q).astype(np.float32).values
    feature_names.append(name)

# 正弦/余弦编码
features['month_sin'] = np.sin(2 * np.pi * date_col.dt.month.values / 12).astype(np.float32)
features['month_cos'] = np.cos(2 * np.pi * date_col.dt.month.values / 12).astype(np.float32)
features['quarter_sin'] = np.sin(2 * np.pi * date_col.dt.quarter.values / 4).astype(np.float32)
features['quarter_cos'] = np.cos(2 * np.pi * date_col.dt.quarter.values / 4).astype(np.float32)
feature_names += ['month_sin', 'month_cos', 'quarter_sin', 'quarter_cos']

# 春节标记
sf_before = np.zeros(n, dtype=np.float32)
sf_during = np.zeros(n, dtype=np.float32)
sf_after = np.zeros(n, dtype=np.float32)

for year, sf_date_str in spring_festival.items():
    sf_date = pd.Timestamp(sf_date_str)
    sf_before[(date_col >= sf_date - pd.Timedelta(days=15)) & (date_col < sf_date)] = 1.0
    sf_during[(date_col >= sf_date) & (date_col < sf_date + pd.Timedelta(days=7))] = 1.0
    sf_after[(date_col >= sf_date + pd.Timedelta(days=7)) & (date_col <= sf_date + pd.Timedelta(days=22))] = 1.0

features['spring_before_15d'] = sf_before
features['spring_during'] = sf_during
features['spring_after_15d'] = sf_after
feature_names += ['spring_before_15d', 'spring_during', 'spring_after_15d']

# 采暖季标记 (11月15日 ~ 3月15日)
heating = np.zeros(n, dtype=np.float32)
for i, dt in enumerate(date_col):
    m = dt.month
    d = dt.day
    if m > 11 or m < 3:
        heating[i] = 1.0
    elif m == 11 and d >= 15:
        heating[i] = 1.0
    elif m == 3 and d <= 15:
        heating[i] = 1.0
features['heating_season'] = heating
feature_names.append('heating_season')

# 月初/月末标记
features['month_start'] = (date_col.dt.day <= 5).astype(np.float32).values
features['month_end'] = (date_col.dt.day >= 25).astype(np.float32).values
feature_names += ['month_start', 'month_end']

print(f"生成特征数: {len(feature_names)}")

# ==========================================
# 保存时间特征
# ==========================================
print("\n保存时间特征...")

output_path = os.path.join(output_dir, '01_time_features.csv')
with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    # 时间特征只有单表头（无ID映射），但为了一致性保留双行表头
    writer.writerow(['data_date'] + feature_names)  # ID row (use feature name as ID)
    writer.writerow(['data_date'] + feature_names)  # Name row
    for i in range(n):
        date_str = date_col.iloc[i].strftime('%Y-%m-%d')
        row = [date_str]
        for name in feature_names:
            val = features[name][i]
            if np.isnan(val):
                row.append('')
            else:
                row.append(f"{val:.6f}")
        writer.writerow(row)

fsize = os.path.getsize(output_path) / 1024
print(f"已保存: {os.path.basename(output_path)} ({fsize:.1f} KB)")
print(f"特征列数: {len(feature_names)}")
print(f"数据行数: {n}")

print("\n" + "=" * 80)
print("Step 2 完成！")
print("=" * 80)

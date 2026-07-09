import pandas as pd
import numpy as np
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_path = os.path.join(base_dir, 'outputs', 'merged_factors_daily_aligned.csv')

print("=" * 80)
print("分析不同时间段的数据完整性")
print("=" * 80)

# 读取数据（跳过第二行表头）
df = pd.read_csv(input_path, header=[0, 1], skiprows=0, encoding='utf-8-sig')
dates = pd.to_datetime(df.iloc[:, 0])
values_df = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

print(f"\n总时间范围: {dates.min()} 至 {dates.max()}")
print(f"总天数: {len(dates)}")
print(f"总列数: {values_df.shape[1]}")

# 定义时间断点
breakpoints = [
    ('2015-12-25', '2016-12-31', '2016年及以前'),
    ('2017-01-01', '2017-12-31', '2017年'),
    ('2018-01-01', '2018-12-31', '2018年'),
    ('2019-01-01', '2019-12-31', '2019年'),
    ('2020-01-01', '2020-12-31', '2020年'),
    ('2021-01-01', '2021-12-31', '2021年'),
    ('2022-01-01', '2022-12-31', '2022年'),
    ('2023-01-01', '2023-12-31', '2023年'),
    ('2024-01-01', '2024-12-31', '2024年'),
    ('2018-01-01', '2024-12-31', '2018-2024年'),
]

print("\n" + "=" * 80)
print("各时间段的数据完整性统计")
print("=" * 80)

for start, end, label in breakpoints:
    mask = (dates >= start) & (dates <= end)
    period_df = values_df.loc[mask]

    total_cells = period_df.size
    valid_cells = period_df.notna().sum().sum()
    missing_cells = total_cells - valid_cells
    completeness_rate = valid_cells / total_cells * 100

    print(f"\n{label}:")
    print(f"  天数: {len(period_df)}")
    print(f"  完整率: {completeness_rate:.2f}%")
    print(f"  有效单元格: {valid_cells:,} / {total_cells:,}")

# 分析每一列的数据起始时间
print("\n" + "=" * 80)
print("各列数据起始时间分布")
print("=" * 80)

col_start_dates = []
for col_idx in range(values_df.shape[1]):
    col_data = values_df.iloc[:, col_idx]
    first_valid_idx = col_data.first_valid_index()
    if first_valid_idx is not None:
        col_start_dates.append(dates.iloc[first_valid_idx])
    else:
        col_start_dates.append(pd.NaT)

col_start_series = pd.Series(col_start_dates)
year_counts = col_start_series.dt.year.value_counts().sort_index()

print("\n数据起始年份分布:")
for year, count in year_counts.items():
    print(f"  {int(year)}年: {count} 列")

# 计算2018-01-01时已有多少列有数据
ref_date = pd.to_datetime('2018-01-01')
cols_with_data_by_2018 = sum(1 for d in col_start_dates if pd.notna(d) and d <= ref_date)
print(f"\n2018-01-01 前已有数据的列数: {cols_with_data_by_2018} / {values_df.shape[1]}")

# 检查标的列的情况
print("\n" + "=" * 80)
print("标的列数据完整性")
print("=" * 80)

# 前27列是标的数据（从之前的输出可知）
target_cols = values_df.columns[:27]
target_df = values_df[target_cols]

print(f"\n标的列数: {len(target_cols)}")
for start, end, label in breakpoints:
    mask = (dates >= start) & (dates <= end)
    period_target = target_df.loc[mask]
    completeness = period_target.notna().sum().sum() / period_target.size * 100
    print(f"{label} 标的完整率: {completeness:.2f}%")

# 输出建议
print("\n" + "=" * 80)
print("建议")
print("=" * 80)

print("""
根据数据完整性分析，可以考虑以下方案：

方案A：使用2018-2024年数据
- 优点：数据更完整，因子覆盖度高
- 缺点：样本量减少

方案B：使用全部可用数据（2015-2024）
- 优点：样本量最大
- 缺点：早期数据有较多缺失

方案C：混合方案
- 2015-2017：只用数据完整的列（主要是标的+早期因子）
- 2018-2024：用全部列
""")

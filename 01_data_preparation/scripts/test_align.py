import pandas as pd
import numpy as np
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'data')
output_dir = os.path.join(base_dir, 'outputs')

print("读取标的数据...")
target_path = os.path.join(data_dir, '标的-铁矿石库存预测标的数据.csv')
with open(target_path, 'r', encoding='utf-8-sig') as f:
    lines = [l.rstrip() for l in f]

row1 = lines[0].split(',')
row2 = lines[1].split(',')

print(f"标的数据列数: {len(row1)}")
print(f"第一列: {row1[0]} - {row2[0]}")
print(f"前5个日期行:")
for i, line in enumerate(lines[2:7]):
    print(f"  {i+1}: {line[:80]}...")

# 解析所有日期
print(f"\n解析日期...")
dates = []
for line in lines[2:]:
    if not line:
        continue
    parts = line.split(',')
    dt_str = parts[0]
    try:
        dt = pd.to_datetime(dt_str.replace(' 00:00:00', ''))
        dates.append(dt)
    except:
        pass

print(f"标的日期范围: {min(dates)} 至 {max(dates)}")
print(f"数据点: {len(dates)}")

# 创建日度时间轴
print(f"\n创建日度时间轴...")
start_date = pd.Timestamp('2015-12-25')
end_date = pd.Timestamp('2026-05-08')
target_index = pd.date_range(start=start_date, end=end_date, freq='D')
print(f"目标时间轴: {target_index[0]} 至 {target_index[-1]}")
print(f"总天数: {len(target_index)}")

print("\n测试完成，思路正确!")

#!/usr/bin/env python3
"""分析数据清洗的实际效果"""

import numpy as np
import pandas as pd
from pathlib import Path

base_dir = Path(__file__).parent.parent
daily_data_path = base_dir / "03_data_cleaning" / "outputs" / "cleaned_data_daily.csv"

# 加载数据
print("加载日频数据...")
df = pd.read_csv(daily_data_path, header=[0], skiprows=[1], encoding="utf-8-sig", low_memory=False)
df["data_date"] = pd.to_datetime(df["data_date"])

targets = ["ID01517011", "ID01516873", "ID01517220"]

for target_id in targets:
    print(f"\n{'='*60}")
    print(f"分析标的: {target_id}")
    print('='*60)

    series = df.set_index("data_date")[target_id].dropna()

    # 统计信息
    mean_val = series.mean()
    std_val = series.std()
    lower_3sigma = mean_val - 3 * std_val
    upper_3sigma = mean_val + 3 * std_val

    print(f"统计信息:")
    print(f"  样本数: {len(series)}")
    print(f"  均值: {mean_val:.2f}")
    print(f"  标准差: {std_val:.2f}")
    print(f"  最小值: {series.min():.2f}")
    print(f"  最大值: {series.max():.2f}")
    print(f"  ±3σ 范围: [{lower_3sigma:.2f}, {upper_3sigma:.2f}]")

    # 检查有多少值超出±3σ
    outlier_mask = (series < lower_3sigma) | (series > upper_3sigma)
    outlier_count = outlier_mask.sum()

    print(f"\n异常值分析:")
    print(f"  超出±3σ的样本数: {outlier_count} ({outlier_count/len(series)*100:.2f}%)")

    if outlier_count > 0:
        outliers = series[outlier_mask]
        print(f"\n  异常值详情:")
        for date, val in outliers.items():
            print(f"    {date.date()}: {val:.2f}")

    # Winsorize处理
    winsorized = series.clip(lower_3sigma, upper_3sigma)

    print(f"\nWinsorize后:")
    print(f"  最小值: {winsorized.min():.2f}")
    print(f"  最大值: {winsorized.max():.2f}")

    # 检查是否有变化
    changes = (series != winsorized).sum()
    print(f"  修改的样本数: {changes}")

    # 检查接近0的值
    near_zero = (series < 1.0).sum()
    print(f"\n  小于1.0的样本数: {near_zero} ({near_zero/len(series)*100:.2f}%)")
    if near_zero > 0:
        near_zero_samples = series[series < 1.0].head()
        print(f"  前几个小值样本:")
        for date, val in near_zero_samples.items():
            print(f"    {date.date()}: {val:.4f}")

print(f"\n{'='*60}")
print("分析完成")
print('='*60)

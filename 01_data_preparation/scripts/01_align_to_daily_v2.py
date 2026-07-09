import pandas as pd
import numpy as np
from datetime import datetime
import os
import csv

def parse_file(filepath):
    """解析文件，返回表头和数据字典"""
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        lines = list(reader)

    header_row1 = lines[0]  # 指标id
    header_row2 = lines[1]  # 指标名称

    # 解析数据
    date_to_values = {}
    dates = []

    for row in lines[2:]:
        if not row or not row[0].strip():
            continue
        date_str = row[0]
        try:
            dt = pd.to_datetime(date_str.replace(' 00:00:00', ''))
        except:
            continue

        date_to_values[dt] = row[1:]
        dates.append(dt)

    return header_row1, header_row2, date_to_values, sorted(dates)

def align_column_to_target(dates, date_to_values, col_idx, target_index):
    """将某一列对齐到目标时间轴，前向填充"""
    # 先构建该列的完整时间序列
    col_data = {}
    for dt in dates:
        if col_idx < len(date_to_values[dt]):
            val_str = date_to_values[dt][col_idx]
            if val_str:
                try:
                    col_data[dt] = float(val_str)
                except:
                    pass

    # 对齐到目标时间轴，前向填充
    aligned = {}
    last_val = None
    for target_dt in target_index:
        # 找这个日期之前的最近数据
        found_val = None
        # 从新到旧遍历源日期
        for dt in sorted(dates, reverse=True):
            if dt <= target_dt and dt in col_data:
                found_val = col_data[dt]
                break

        if found_val is not None:
            last_val = found_val
        aligned[target_dt] = last_val

    return aligned

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    output_dir = os.path.join(base_dir, 'outputs')

    # 文件路径
    files = [
        ('target', os.path.join(data_dir, '标的-铁矿石库存预测标的数据.csv')),
        ('daily', os.path.join(output_dir, '结果表格_日度(1).csv')),
        ('weekly', os.path.join(output_dir, '结果表格_周度(1).csv')),
        ('monthly', os.path.join(output_dir, '结果表格_月度(1).csv')),
        ('quarterly', os.path.join(output_dir, '结果表格_季度(1).csv')),
        ('annual', os.path.join(output_dir, '结果表格_年度(1).csv'))
    ]

    print("=" * 80)
    print("步骤1: 创建目标日度时间轴 (V2 - 修复前向填充)")
    print("=" * 80)

    # 目标时间轴
    start_date = pd.Timestamp('2015-12-25')
    end_date = pd.Timestamp('2026-05-08')
    target_index = pd.date_range(start=start_date, end=end_date, freq='D')
    print(f"目标时间轴: {target_index[0]} 至 {target_index[-1]}")
    print(f"总天数: {len(target_index)}")

    # 存储所有列数据
    all_columns_data = {}
    final_header1 = ['data_date']
    final_header2 = ['data_date']

    # 处理每个文件
    for name, filepath in files:
        print(f"\n{'=' * 80}")
        print(f"处理 {name} 数据")
        print(f"{'=' * 80}")

        if not os.path.exists(filepath):
            print(f"文件不存在: {filepath}")
            continue

        print(f"加载文件: {os.path.basename(filepath)}")
        header1, header2, date_to_values, dates = parse_file(filepath)

        print(f"日期范围: {dates[0]} 至 {dates[-1]}")
        print(f"数据行数: {len(dates)}")
        print(f"列数（不含日期）: {len(header1) - 1}")

        # 处理每一列
        for col_idx in range(1, len(header1)):
            col_id = header1[col_idx]
            col_name = header2[col_idx] if col_idx < len(header2) else col_id

            # 如果这个列ID已经存在，跳过（避免重复列）
            if col_id in all_columns_data:
                continue

            # 对齐
            aligned_col = align_column_to_target(dates, date_to_values, col_idx - 1, target_index)

            all_columns_data[col_id] = aligned_col
            final_header1.append(col_id)
            final_header2.append(col_name)

    print(f"\n{'=' * 80}")
    print("生成输出文件")
    print(f"{'=' * 80}")
    print(f"总列数: {len(final_header1)}")

    # 构建输出行
    output_rows = []
    output_rows.append(final_header1)
    output_rows.append(final_header2)

    for dt in target_index:
        row = [dt.strftime('%Y-%m-%d')]
        for col_id in final_header1[1:]:
            val = all_columns_data[col_id][dt]
            if val is None:
                row.append('')
            else:
                row.append(str(val))
        output_rows.append(row)

    output_path = os.path.join(output_dir, 'merged_factors_daily_aligned.csv')
    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(output_rows)

    print(f"输出文件: {os.path.basename(output_path)}")
    print(f"总行数: {len(output_rows)}")
    print("完成!")

if __name__ == '__main__':
    main()

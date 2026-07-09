import pandas as pd
import numpy as np
from datetime import datetime
import os

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    output_dir = os.path.join(base_dir, 'outputs')

    # 文件路径
    files = {
        'target': os.path.join(data_dir, '标的-铁矿石库存预测标的数据.csv'),
        'daily': os.path.join(output_dir, '结果表格_日度(1).csv'),
        'weekly': os.path.join(output_dir, '结果表格_周度(1).csv'),
        'monthly': os.path.join(output_dir, '结果表格_月度(1).csv'),
        'quarterly': os.path.join(output_dir, '结果表格_季度(1).csv'),
        'annual': os.path.join(output_dir, '结果表格_年度(1).csv')
    }

    print("=" * 80)
    print("步骤1: 创建目标日度时间轴")
    print("=" * 80)

    # 目标时间轴：2015-12-25 至 2026-05-08
    start_date = pd.Timestamp('2015-12-25')
    end_date = pd.Timestamp('2026-05-08')
    target_index = pd.date_range(start=start_date, end=end_date, freq='D')
    print(f"目标时间轴: {target_index[0]} 至 {target_index[-1]}")
    print(f"总天数: {len(target_index)}")

    # 用于存储所有列
    all_cols = {}
    header_row1 = ['data_date']  # 指标id行
    header_row2 = ['data_date']  # 指标名称行

    # 逐个处理文件
    for name, filepath in files.items():
        print(f"\n{'=' * 80}")
        print(f"处理 {name} 数据: {os.path.basename(filepath)}")
        print(f"{'=' * 80}")

        if not os.path.exists(filepath):
            print(f"文件不存在!")
            continue

        # 直接读取所有内容
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            lines = [line.rstrip('\n') for line in f]

        # 前两行为表头
        row1 = lines[0].split(',')
        row2 = lines[1].split(',')

        # 解析数据
        data_dict = {}
        data_dates = []

        for line in lines[2:]:
            if not line.strip():
                continue
            parts = line.split(',')
            date_str = parts[0]

            # 解析日期
            try:
                dt = pd.to_datetime(date_str.replace(' 00:00:00', ''))
            except:
                continue

            data_dates.append(dt)
            data_dict[dt] = parts[1:]

        print(f"  数据日期范围: {min(data_dates)} 至 {max(data_dates)}")
        print(f"  数据行数: {len(data_dates)}")
        print(f"  列数（不含日期）: {len(row1)-1}")

        # 对齐每一列到目标时间轴
        for col_idx in range(1, len(row1)):
            col_id = row1[col_idx]
            col_name = row2[col_idx] if col_idx < len(row2) else ''

            # 构建该列的时间序列
            series_data = {}
            for dt in data_dates:
                if col_idx < len(data_dict[dt]):
                    val = data_dict[dt][col_idx]
                    if val:
                        try:
                            series_data[dt] = float(val)
                        except:
                            pass

            # 对齐到目标时间轴
            aligned_series = {}
            last_val = None
            for target_dt in target_index:
                # 找这个日期之前最近的数据点
                val = None
                # 从最新往旧找，找第一个<=target_dt的
                for dt in sorted(data_dates, reverse=True):
                    if dt <= target_dt:
                        if dt in series_data:
                            val = series_data[dt]
                            break
                aligned_series[target_dt] = val

            all_cols[f"{col_id}"] = aligned_series
            header_row1.append(col_id)
            header_row2.append(col_name)

    print(f"\n{'=' * 80}")
    print("合并并输出")
    print(f"{'=' * 80}")
    print(f"总列数: {len(all_cols)+1}")

    # 构建输出
    output_rows = []
    output_rows.append(header_row1)
    output_rows.append(header_row2)

    for dt in target_index:
        row = [dt.strftime('%Y-%m-%d')]
        for col_id in header_row1[1:]:
            val = all_cols[col_id][dt]
            if val is None:
                row.append('')
            else:
                row.append(str(val))
        output_rows.append(row)

    output_path = os.path.join(output_dir, 'merged_factors_daily_aligned.csv')
    with open(output_path, 'w', encoding='utf-8-sig') as f:
        for row in output_rows:
            f.write(','.join(row) + '\n')

    print(f"输出文件: {os.path.basename(output_path)}")
    print(f"总行数（含表头）: {len(output_rows)}")
    print("完成!")

if __name__ == '__main__':
    main()

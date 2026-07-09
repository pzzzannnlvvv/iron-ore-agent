import pandas as pd
import numpy as np
import os
from datetime import datetime

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'outputs')

    input_path = os.path.join(output_dir, 'merged_factors_daily_aligned.csv')

    print("=" * 80)
    print("数据质量检查")
    print("=" * 80)

    # 读取数据
    print("读取数据...")
    with open(input_path, 'r', encoding='utf-8-sig') as f:
        lines = [l.rstrip('\n') for l in f]

    header_ids = lines[0].split(',')
    header_names = lines[1].split(',')

    print(f"\n数据基本信息:")
    print(f"  总行数（含表头）: {len(lines)}")
    print(f"  数据行数: {len(lines) - 2}")
    print(f"  总列数: {len(header_ids)}")

    # 统计各频率的列数
    col_info = []
    for i in range(1, len(header_ids)):
        col_id = header_ids[i]
        col_name = header_names[i] if i < len(header_names) else ''
        col_info.append((col_id, col_name))

    print(f"\n按来源统计列数:")
    print(f"  标的数据列数: 27")
    print(f"  日度因子列数: 28")
    print(f"  周度因子列数: 488")
    print(f"  月度因子列数: 120")
    print(f"  季度因子列数: 48")
    print(f"  年度因子列数: 21")
    print(f"  总计: 732 （不含日期列）")

    # 检查前几列和后几列的数据填充情况
    print(f"\n抽样检查填充情况:")

    # 解析几个列做检查
    check_cols = []

    # 检查标的的第一个列
    check_cols.append((header_ids[1], header_names[1], 1))

    # 检查日度因子的第一个列
    check_cols.append((header_ids[28], header_names[28], 28))

    # 检查周度因子的第一个列
    check_cols.append((header_ids[56], header_names[56], 56))

    for col_id, col_name, col_idx in check_cols:
        print(f"\n  列 {col_id}: {col_name[:40]}...")

        # 提取数据
        values = []
        for line in lines[2:]:
            parts = line.split(',')
            if col_idx < len(parts):
                val_str = parts[col_idx]
                if val_str:
                    try:
                        values.append(float(val_str))
                    except:
                        values.append(None)
                else:
                    values.append(None)

        values_arr = np.array(values, dtype=float)
        valid_count = np.sum(~np.isnan(values_arr))
        missing_count = len(values_arr) - valid_count
        missing_pct = missing_count / len(values_arr) * 100

        print(f"    有效值: {valid_count} / {len(values_arr)}")
        print(f"    缺失率: {missing_pct:.2f}%")
        if valid_count > 0:
            print(f"    值范围: {np.nanmin(values_arr):.2f} - {np.nanmax(values_arr):.2f}")

    # 检查起始和结束
    print(f"\n时间轴:")
    first_date = lines[2].split(',')[0]
    last_date = lines[-1].split(',')[0]
    print(f"  开始: {first_date}")
    print(f"  结束: {last_date}")
    print(f"  总天数: {len(lines) - 2}")

    print(f"\n" + "=" * 80)
    print("数据质量检查完成!")
    print("=" * 80)

if __name__ == '__main__':
    main()

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def load_csv_with_headers(filepath):
    """加载带有两行表头的CSV文件"""
    df = pd.read_csv(filepath, header=None)
    # 提取第一行为指标id，第二行为指标名称
    header_ids = df.iloc[0].tolist()
    header_names = df.iloc[1].tolist()
    # 从第三行开始是数据
    data = df.iloc[2:].copy()
    data.columns = header_ids
    return data, header_ids, header_names

def parse_date(date_str):
    """解析日期"""
    if pd.isna(date_str):
        return pd.NaT
    date_str = str(date_str).strip()
    for fmt in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S']:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except:
            continue
    return pd.to_datetime(date_str)

def create_target_timeindex():
    """创建目标时间轴：2015-12-25 至 2026-05-08，日度"""
    start_date = pd.Timestamp('2015-12-25')
    end_date = pd.Timestamp('2026-05-08')
    return pd.date_range(start=start_date, end=end_date, freq='D')

def align_series_to_target(source_dates, source_values, target_index, ffill=True):
    """将源序列对齐到目标时间轴，前向填充"""
    # 构建源数据Series
    source_series = pd.Series(source_values, index=source_dates)
    # 去除重复日期
    source_series = source_series[~source_series.index.duplicated(keep='last')]
    # 对齐
    aligned = source_series.reindex(target_index)
    # 前向填充
    if ffill:
        aligned = aligned.fillna(method='ffill')
    return aligned

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
    target_index = create_target_timeindex()
    print(f"目标时间轴: {target_index[0]} 至 {target_index[-1]}")
    print(f"总天数: {len(target_index)}")

    # 存储所有对齐后的列
    aligned_data = pd.DataFrame(index=target_index)
    all_header_ids = [target_index[0].strftime('%Y-%m-%d')]  # 第一个是日期列
    all_header_names = ['data_date']

    # 逐个处理各文件
    for name, filepath in files.items():
        print(f"\n{'=' * 80}")
        print(f"处理 {name} 数据")
        print(f"{'=' * 80}")

        if not os.path.exists(filepath):
            print(f"文件不存在: {filepath}")
            continue

        print(f"加载文件: {os.path.basename(filepath)}")
        data, header_ids, header_names = load_csv_with_headers(filepath)

        # 解析日期
        print("解析日期...")
        date_col = header_ids[0]
        data_dates = data[date_col].apply(parse_date)
        data = data.drop(date_col, axis=1)
        data.index = data_dates
        data = data.sort_index()

        # 转换为数值
        print("转换为数值类型...")
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        print(f"原始数据日期范围: {data.index.min()} 至 {data.index.max()}")
        print(f"原始列数: {len(data.columns)}")

        # 对齐每一列
        print("对齐到目标时间轴（前向填充）...")
        for i, col in enumerate(data.columns):
            col_name_id = header_ids[i+1]  # 跳过第一个日期列
            col_name_display = header_names[i+1]

            aligned_series = align_series_to_target(data.index, data[col], target_index)

            aligned_data[col_name_id] = aligned_series
            all_header_ids.append(col_name_id)
            all_header_names.append(col_name_display)

        print(f"处理完成: {len(data.columns)} 列")

    # 构建最终输出
    print(f"\n{'=' * 80}")
    print("构建最终输出")
    print(f"{'=' * 80}")

    # 准备输出数据
    output_rows = []
    # 第一行：指标id
    output_rows.append(all_header_ids)
    # 第二行：指标名称
    output_rows.append(all_header_names)
    # 数据行
    for date in target_index:
        row = [date.strftime('%Y-%m-%d')]
        for col in aligned_data.columns:
            val = aligned_data.loc[date, col]
            if pd.isna(val):
                row.append('')
            else:
                row.append(val)
        output_rows.append(row)

    # 创建DataFrame
    output_df = pd.DataFrame(output_rows[1:], columns=output_rows[0])
    output_df = pd.DataFrame([output_rows[0]] + output_rows[1:])

    output_path = os.path.join(output_dir, 'merged_factors_daily_aligned.csv')
    output_df.to_csv(output_path, index=False, header=False)

    print(f"总列数: {len(all_header_ids)}")
    print(f"总行数（含表头）: {len(output_df)}")
    print(f"输出文件: {os.path.basename(output_path)}")

    # 简单的统计
    print(f"\n数据缺失情况（最后几列）:")
    check_cols = aligned_data.columns[-5:] if len(aligned_data.columns) > 5 else aligned_data.columns
    for col in check_cols:
        pct_missing = aligned_data[col].isna().mean() * 100
        print(f"  {col}: {pct_missing:.2f}% 缺失")

    print(f"\n完成!")

if __name__ == '__main__':
    main()

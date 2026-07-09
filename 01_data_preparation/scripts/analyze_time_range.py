import pandas as pd
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'data')
output_dir = os.path.join(base_dir, 'outputs')

files = {
    '标的数据': os.path.join(data_dir, '标的-铁矿石库存预测标的数据.csv'),
    '日度因子': os.path.join(output_dir, '结果表格_日度(1).csv'),
    '周度因子': os.path.join(output_dir, '结果表格_周度(1).csv'),
    '月度因子': os.path.join(output_dir, '结果表格_月度(1).csv'),
    '季度因子': os.path.join(output_dir, '结果表格_季度(1).csv'),
    '年度因子': os.path.join(output_dir, '结果表格_年度(1).csv'),
}

print("=" * 80)
print("各数据集时间范围分析")
print("=" * 80)

for name, file_path in files.items():
    if not os.path.exists(file_path):
        print(f"\n{name}: 文件不存在")
        continue

    df = pd.read_csv(file_path)

    # 前两行是表头，从第三行开始是数据
    if df.shape[0] > 2:
        # 直接提取日期列，手动处理
        date_values = df.iloc[2:, 0].values
        dates = []
        for d in date_values:
            try:
                dt = pd.to_datetime(d)
                dates.append(dt)
            except:
                pass

        if len(dates) > 0:
            dates = pd.Series(dates)
            min_date = dates.min()
            max_date = dates.max()
            print(f"\n{name}:")
            print(f"  数据行数: {len(dates)}")
            print(f"  最早日期: {min_date.strftime('%Y-%m-%d')}")
            print(f"  最晚日期: {max_date.strftime('%Y-%m-%d')}")
            print(f"  是否倒序: {dates.iloc[0] > dates.iloc[-1]}")
            print(f"  前3个日期: {[d.strftime('%Y-%m-%d') for d in dates[:3]]}")
            print(f"  后3个日期: {[d.strftime('%Y-%m-%d') for d in dates[-3:]]}")
        else:
            print(f"\n{name}: 无法解析日期")

print("\n" + "=" * 80)

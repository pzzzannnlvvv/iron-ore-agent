import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'data')
output_dir = os.path.join(base_dir, 'outputs')

files = {
    '标的数据': os.path.join(data_dir, '标的-铁矿石库存预测标的数据.csv'),
    '日度因子': os.path.join(output_dir, '结果表格_日度(1).csv'),
    '周度因子': os.path.join(output_dir, '结果表格_周度(1).csv'),
    '月度因子': os.path.join(output_dir, '结果表格_月度(1).csv'),
}

print("=" * 80)
print("各数据集时间范围分析")
print("=" * 80)

for name, file_path in files.items():
    if not os.path.exists(file_path):
        continue

    with open(file_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

    dates = []
    # 跳过前两行（表头）
    for line in lines[2:]:
        parts = line.strip().split(',')
        if parts and parts[0]:
            dates.append(parts[0])

    if dates:
        print(f"\n{name}:")
        print(f"  数据行数: {len(dates)}")
        print(f"  前3个日期: {dates[:3]}")
        print(f"  后3个日期: {dates[-3:]}")

print("\n" + "=" * 80)

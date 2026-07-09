import pandas as pd
import numpy as np
import os
import csv

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
input_path = os.path.join(base_dir, '03_data_cleaning', 'outputs', 'cleaned_data_daily.csv')
output_dir = os.path.join(base_dir, '05_feature_engineering', 'outputs')
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("Step 1: 数据理解与列分组")
print("=" * 80)

# 读取数据
print("\n读取数据...")
df = pd.read_csv(input_path, header=[0, 1], encoding='utf-8-sig')
date_col = df.iloc[:, 0]
values_df = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

col_ids = list(df.columns.get_level_values(0))[1:]
col_names = list(df.columns.get_level_values(1))[1:]

print(f"数据形状: {df.shape}")
print(f"日期范围: {date_col.iloc[0]} 至 {date_col.iloc[-1]}")
print(f"因子列数: {len(col_ids)}")

# ==========================================
# 分组规则（优先级从高到低）
# ==========================================
group_rules = [
    # (组名, 关键词列表, ALL关键词必须匹配)
    # 标的(Y变量) — 必须是45港总库存
    ('标的(Y变量)', ['45个港口', '库存'], False),

    # 供应侧
    ('供应-发运', ['发运'], False),
    ('供应-发货量', ['发货量'], False),
    ('供应-到港', ['到港'], False),
    ('供应-产量', ['产量'], False),
    ('供应-矿山', ['矿山', '澳洲', '巴西', '淡水河谷', '力拓', 'FMG', 'RoyHill', 'BHP', '必和必拓'], False),

    # 需求侧
    ('需求-疏港', ['疏港'], False),
    ('需求-开工率', ['开工率', '产能利用率'], False),
    ('需求-生铁粗钢', ['生铁', '粗钢', '铁水'], False),
    ('需求-消费', ['消费', '消耗量', '用量', '日耗'], False),

    # 价格侧
    ('价格-矿石价格', ['价格', '铁矿石'], False),
    ('价格-价差利润', ['价差', '利润', '现金利润'], False),
    ('价格-关联商品', ['螺纹钢', '热卷', '焦炭', '焦煤', '废钢'], False),
    ('价格-海运', ['运费', '海运费', 'C3', 'C5'], False),

    # 库存(因子)
    ('库存-港口', ['库存', '港口'], False),
    ('库存-钢厂', ['库存', '钢厂'], False),
    ('库存-其他', ['库存'], False),

    # 宏观
    ('宏观-PMI', ['PMI'], False),
    ('宏观-投资', ['投资'], False),
    ('宏观-货币', ['M1', 'M2', '社融', '利率', '汇率', '货币', '信贷'], False),
    ('宏观-GDP', ['GDP', 'CPI', 'PPI'], False),
    ('宏观-地产', ['房地产', '拿地', '新开工', '施工面积', '竣工面积', '房价'], False),
    ('宏观-基建', ['基建', '专项债'], False),

    # 供需平衡
    ('供需-海漂压港', ['海漂', '压港'], False),
    ('供需-配比', ['配比', '入炉比', '入炉品位', '可用天数'], False),
]

# 识别核心因子（来自《预测目标指南》）
core_keywords = [
    # 四大矿山
    '澳洲BHP', '澳洲力拓', '澳洲FMG', '澳洲RoyHill', '巴西淡水河谷',
    '力拓', '必和必拓', 'BHP', 'FMG', 'RoyHill', '淡水河谷',
    # 发运
    '澳洲铁矿石发运量', '巴西铁矿石发运量', '全球铁矿石发运量',
    '全球铁矿石对中国发运量',
    # 库存
    '45个港口', '铁矿石：进口：库存：45个港口',
    '247家钢厂', '钢厂库存',
    # 产能利用
    '高炉产能利用率', '高炉开工率',
    # 铁水产量
    '铁水产量', '生铁产量',
    # 疏港
    '疏港量', '日均疏港量',
    # 价格
    '普氏', '铁矿石价格指数', '铁矿石：价格',
    # 利润
    '钢厂利润', '现金利润',
    # 宏观
    '制造业PMI',
    # 粗钢
    '粗钢产量',
    # 出口
    '铁矿石：出口',
    # 国内产量
    '国内铁矿石', '铁矿石：产量：国内',
]


def match_group(col_name):
    """按优先级匹配列分组"""
    for group_name, keywords, require_all in group_rules:
        if require_all:
            if all(kw in col_name for kw in keywords):
                return group_name
        else:
            if any(kw in col_name for kw in keywords):
                return group_name
    return '其他'


def is_core_factor(col_name):
    """判断是否为核心因子"""
    name_lower = col_name.lower()
    for kw in core_keywords:
        if kw.lower() in name_lower:
            return True
    return False


# ==========================================
# 执行分组
# ==========================================
print("\n开始列分组...")

mapping = []
group_counts = {}

for idx, (col_id, col_name) in enumerate(zip(col_ids, col_names)):
    group = match_group(col_name)
    core = is_core_factor(col_name)

    mapping.append({
        'col_id': col_id,
        'col_name': col_name,
        'group': group,
        'is_core': core,
        'valid_count': int(values_df.iloc[:, idx].notna().sum()),
        'mean': values_df.iloc[:, idx].mean(),
    })

    group_counts[group] = group_counts.get(group, 0) + 1

    if idx % 100 == 0:
        print(f"  已处理 {idx}/{len(col_ids)} 列")

# ==========================================
# 统计摘要
# ==========================================
print("\n" + "=" * 80)
print("分组统计")
print("=" * 80)

group_order = [g[0] for g in group_rules] + ['其他']
for g in group_order:
    if g in group_counts:
        core_count = sum(1 for m in mapping if m['group'] == g and m['is_core'])
        print(f"  {g:<20s}: {group_counts[g]:>4d} 列  (核心: {core_count})")

total_core = sum(1 for m in mapping if m['is_core'])
print(f"\n  总计: {len(mapping)} 列")
print(f"  核心因子: {total_core} 列")
print(f"  其他因子: {len(mapping) - total_core} 列")

# ==========================================
# 保存 feature_group_mapping.csv
# ==========================================
print("\n保存分组映射...")

mapping_path = os.path.join(output_dir, 'feature_group_mapping.csv')
with open(mapping_path, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['列ID', '列名', '分组', '是否核心因子', '有效样本数', '均值'])
    for m in mapping:
        writer.writerow([
            m['col_id'],
            m['col_name'],
            m['group'],
            '是' if m['is_core'] else '否',
            m['valid_count'],
            f"{m['mean']:.6f}" if pd.notna(m['mean']) else ''
        ])

# ==========================================
# 保存 core_factors_list.csv
# ==========================================
print("保存核心因子列表...")

core_path = os.path.join(output_dir, 'core_factors_list.csv')
with open(core_path, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['列ID', '列名', '分组', '有效样本数'])
    for m in mapping:
        if m['is_core']:
            writer.writerow([
                m['col_id'],
                m['col_name'],
                m['group'],
                m['valid_count']
            ])

# ==========================================
# 打印核心因子详情
# ==========================================
print(f"\n核心因子详情 ({total_core}列):")
print("-" * 80)
for m in mapping:
    if m['is_core']:
        print(f"  [{m['group']}] {m['col_id']}: {m['col_name'][:60]}")

# ==========================================
# 保存分组摘要
# ==========================================
summary_path = os.path.join(output_dir, 'feature_group_summary.csv')
with open(summary_path, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['分组', '总列数', '核心列数', '非核心列数'])
    for g in group_order:
        if g in group_counts:
            core_c = sum(1 for m in mapping if m['group'] == g and m['is_core'])
            writer.writerow([g, group_counts[g], core_c, group_counts[g] - core_c])

print(f"\n分组映射已保存: {os.path.basename(mapping_path)}")
print(f"核心因子已保存: {os.path.basename(core_path)}")
print(f"分组摘要已保存: {os.path.basename(summary_path)}")
print("\n" + "=" * 80)
print("Step 1 完成！")
print("=" * 80)

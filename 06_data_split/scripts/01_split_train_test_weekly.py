"""
数据切分脚本（周频版）：按时间顺序划分训练集和测试集（无验证集）

切分逻辑：
- 训练集: 2015-12-25 至 2024-12-31
- Gap: 14周
- 测试集: 2024-12-31 + 14周 之后
"""

import csv
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = PROJECT_ROOT / "05_feature_engineering/outputs/09_features_final_weekly.csv"
OUTPUT_DIR = PROJECT_ROOT / "06_data_split/outputs"

TRAIN_VAL_CUTOFF = "2024-12-31"
GAP_WEEKS = 14


def read_dual_header_csv(path):
    """读取双表头 CSV，返回 header 行和 data 行列表"""
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header1 = next(reader)
        header2 = next(reader)
        rows = list(reader)

    return header1, header2, rows


def add_weeks(dt, weeks):
    """给日期添加指定周数"""
    return dt + timedelta(weeks=weeks)


def main():
    print("=" * 70)
    print("数据切分（周频版）: 训练集 + 测试集（无验证集）")
    print("=" * 70)

    # 读取数据
    header_ids, header_names, rows = read_dual_header_csv(INPUT_PATH)
    date_idx = header_ids.index("data_date")

    # 解析并排序
    rows_with_date = []
    for row in rows:
        d = row[date_idx].strip()
        if d:
            rows_with_date.append((d, row))

    rows_with_date.sort(key=lambda x: x[0])
    print(f"\n总数据行数: {len(rows_with_date)}")
    print(f"日期范围: {rows_with_date[0][0]} ~ {rows_with_date[-1][0]}")

    # 切分训练集
    cutoff_dt = datetime.strptime(TRAIN_VAL_CUTOFF, "%Y-%m-%d")
    train_rows = [(d, r) for d, r in rows_with_date if d <= TRAIN_VAL_CUTOFF]
    print(f"\n训练集: {len(train_rows)} 行, {train_rows[0][0]} ~ {train_rows[-1][0]}")

    # 计算测试集起始：cutoff + 14周
    test_start_dt = add_weeks(cutoff_dt, GAP_WEEKS)
    test_start = test_start_dt.strftime("%Y-%m-%d")

    # 找到第一个 >= test_start 的日期
    test_rows = []
    for d, r in rows_with_date:
        if d >= test_start:
            test_rows.append((d, r))

    if test_rows:
        print(f"测试集: {len(test_rows)} 行, {test_rows[0][0]} ~ {test_rows[-1][0]}")
    else:
        print(f"警告: 没有找到 >= {test_start} 的数据！")

    # 保存
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 保存训练集
    train_path = OUTPUT_DIR / "train_weekly.csv"
    with open(train_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header_ids)
        writer.writerow(header_names)
        writer.writerows(r for _, r in train_rows)
    print(f"\n已保存: {train_path}")

    # 保存测试集
    if test_rows:
        test_path = OUTPUT_DIR / "test_weekly.csv"
        with open(test_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(header_ids)
            writer.writerow(header_names)
            writer.writerows(r for _, r in test_rows)
        print(f"已保存: {test_path}")

    # 保存切分信息
    split_info = {
        "input_file": str(INPUT_PATH),
        "train_cutoff": TRAIN_VAL_CUTOFF,
        "gap_weeks": GAP_WEEKS,
        "test_start": test_start,
        "splits": {
            "train": {"rows": len(train_rows), "start": train_rows[0][0], "end": train_rows[-1][0]},
            "test": {"rows": len(test_rows), "start": test_rows[0][0] if test_rows else None,
                     "end": test_rows[-1][0] if test_rows else None}
        }
    }

    info_path = OUTPUT_DIR / "split_info_weekly.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    print(f"已保存: {info_path}")

    # 验证明细
    print("\n" + "=" * 70)
    print("切分明细")
    print("=" * 70)
    print(f"训练集截止: {TRAIN_VAL_CUTOFF}")
    print(f"Gap: {GAP_WEEKS} 周")
    print(f"测试集起始: {test_start}")
    print(f"\n数据总量: {len(rows_with_date)} = 训练{len(train_rows)} + 测试{len(test_rows)}")

    if test_rows:
        gap_start = datetime.strptime(train_rows[-1][0], "%Y-%m-%d")
        gap_end = datetime.strptime(test_rows[0][0], "%Y-%m-%d")
        actual_gap_days = (gap_end - gap_start).days
        actual_gap_weeks = actual_gap_days / 7
        print(f"实际Gap: {actual_gap_weeks:.1f} 周 ({actual_gap_days} 天)")

    print("\n" + "=" * 70)
    print("切分完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()


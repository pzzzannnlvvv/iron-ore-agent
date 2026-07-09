"""
数据切分脚本：按时间顺序划分训练集/验证集/测试集

切分逻辑：
- 训练集：2015-12-25 ~ 2024-12-31 的前 80%（时间顺序）
- 验证集：2015-12-25 ~ 2024-12-31 的后 20%（时间顺序）
- 14 周 Gap：2024-12-31 + 98 天 ≈ 2025-04-08
- 测试集：2025-04-08 之后的所有数据
"""

import csv
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = PROJECT_ROOT / "05_feature_engineering/outputs/09_features_final.csv"
OUTPUT_DIR = PROJECT_ROOT / "06_data_split/outputs"

GAP_DAYS = 14 * 7  # 14 周
TRAIN_VAL_CUTOFF = "2024-12-31"
TRAIN_RATIO = 0.8


def read_dual_header_csv(path):
    """读取双表头 CSV，返回 header 行和 data 行列表"""
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header1 = next(reader)
        header2 = next(reader)
        rows = list(reader)

    # 使用第一行作为列名
    return header1, rows


def main():
    header, rows = read_dual_header_csv(INPUT_PATH)

    date_idx = header.index("data_date")

    # 解析并排序
    rows_with_date = []
    for row in rows:
        d = row[date_idx].strip()
        if d:
            rows_with_date.append((d, row))

    rows_with_date.sort(key=lambda x: x[0])

    print(f"总数据行数: {len(rows_with_date)}")
    print(f"日期范围: {rows_with_date[0][0]} ~ {rows_with_date[-1][0]}")

    # 切分训练/验证区间
    train_val_rows = [(d, r) for d, r in rows_with_date if d <= TRAIN_VAL_CUTOFF]
    n_tv = len(train_val_rows)
    n_train = int(n_tv * TRAIN_RATIO)

    train_rows = train_val_rows[:n_train]
    val_rows = train_val_rows[n_train:]

    # 测试集：gap 之后
    from datetime import datetime, timedelta
    cutoff_dt = datetime.strptime(TRAIN_VAL_CUTOFF, "%Y-%m-%d")
    test_start_dt = cutoff_dt + timedelta(days=GAP_DAYS)
    test_start = test_start_dt.strftime("%Y-%m-%d")

    test_rows = [(d, r) for d, r in rows_with_date if d >= test_start]

    # 输出
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, subset in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        path = OUTPUT_DIR / f"{name}.csv"
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(r for _, r in subset)
        print(f"{name}.csv: {len(subset)} 行, {subset[0][0]} ~ {subset[-1][0]}")

    # 保存切分信息
    split_info = {
        "input_file": str(INPUT_PATH),
        "train_val_cutoff": TRAIN_VAL_CUTOFF,
        "train_ratio": TRAIN_RATIO,
        "gap_days": GAP_DAYS,
        "test_start": test_start,
        "splits": {
            "train": {"rows": len(train_rows), "start": train_rows[0][0], "end": train_rows[-1][0]},
            "val": {"rows": len(val_rows), "start": val_rows[0][0], "end": val_rows[-1][0]},
            "test": {"rows": len(test_rows), "start": test_rows[0][0], "end": test_rows[-1][0]},
        },
    }

    info_path = OUTPUT_DIR / "split_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    print(f"\nsplit_info.json 已保存")

    # 验证明细
    print(f"\n--- 验证明细 ---")
    print(f"训练集+验证集截止: {TRAIN_VAL_CUTOFF}")
    print(f"Gap 区间: {TRAIN_VAL_CUTOFF} + {GAP_DAYS}天 → 测试集起始: {test_start}")
    print(f"数据总量: {len(rows_with_date)} = 训练{len(train_rows)} + 验证{len(val_rows)} + 测试{len(test_rows)}")
    print(f"训练/验证比: {len(train_rows)}/{len(val_rows)} = {len(train_rows)/len(val_rows):.1f}:1")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
三个问题标的特殊处理：数据切分
为每个标的分别切分训练集和测试集
策略：
- 训练集：到2024-12-31
- Gap：14周
- 测试集：2025-04-11之后
"""

import os
import json
import csv
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# ==================== 路径配置 ====================
base_dir = Path(__file__).resolve().parent.parent.parent
config_path = base_dir / "three_special_targets" / "config" / "target_list.json"
output_dir = base_dir / "three_special_targets" / "outputs"

TRAIN_VAL_CUTOFF = "2024-12-31"
GAP_WEEKS = 14


def load_target_list():
    """加载标的配置"""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def add_weeks(dt, weeks):
    """给日期添加指定周数"""
    return dt + timedelta(weeks=weeks)


def split_single_target(target_id, target_name):
    """为单个标的切分数据"""
    print(f"\n  处理标的: {target_id} - {target_name}")

    feature_path = output_dir / target_id / "features_weekly.csv"
    if not feature_path.exists():
        print(f"    [警告] 特征文件不存在: {feature_path}")
        return None

    # 读取特征
    df = pd.read_csv(feature_path, encoding="utf-8-sig")

    # 构造目标y：target_diff_1向后平移1周（预测下一期的变化量）
    if "target_diff_1" not in df.columns:
        print(f"    [警告] 未找到 target_diff_1 列")
        return None

    df["_y"] = df["target_diff_1"].shift(-1)
    df = df.dropna(subset=["_y"]).copy()

    # 切分
    cutoff_dt = datetime.strptime(TRAIN_VAL_CUTOFF, "%Y-%m-%d")
    test_start_dt = add_weeks(cutoff_dt, GAP_WEEKS)
    test_start = test_start_dt.strftime("%Y-%m-%d")

    train_mask = df["data_date"] <= TRAIN_VAL_CUTOFF
    test_mask = df["data_date"] >= test_start

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    print(f"    训练集: {len(train_df)} 行")
    print(f"    测试集: {len(test_df)} 行")

    # 保存
    target_output_dir = output_dir / target_id
    train_path = target_output_dir / "train_weekly.csv"
    test_path = target_output_dir / "test_weekly.csv"

    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    # 保存切分信息
    split_info = {
        "target_id": target_id,
        "target_name": target_name,
        "train_cutoff": TRAIN_VAL_CUTOFF,
        "gap_weeks": GAP_WEEKS,
        "test_start": test_start,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_date_range": [train_df["data_date"].iloc[0], train_df["data_date"].iloc[-1]] if len(train_df) > 0 else None,
        "test_date_range": [test_df["data_date"].iloc[0], test_df["data_date"].iloc[-1]] if len(test_df) > 0 else None
    }

    info_path = target_output_dir / "split_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)

    print(f"    已保存: {train_path}")
    print(f"    已保存: {test_path}")
    print(f"    已保存: {info_path}")

    return split_info


def main():
    print("=" * 80)
    print("三个问题标的特殊处理 - Phase 2: 数据切分")
    print("=" * 80)

    # 加载配置
    target_config = load_target_list()
    targets = target_config["targets"]

    print(f"\n切分策略:")
    print(f"  训练集截止: {TRAIN_VAL_CUTOFF}")
    print(f"  Gap: {GAP_WEEKS} 周")

    # 处理每个标的
    print("\n处理标的...")
    split_records = {}

    for i, target in enumerate(targets):
        target_id = target["target_id"]
        target_name = target["target_name"]

        split_info = split_single_target(target_id, target_name)
        if split_info:
            split_records[target_id] = split_info

    # 保存总记录
    print("\n保存总记录...")
    record = {
        "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_cutoff": TRAIN_VAL_CUTOFF,
        "gap_weeks": GAP_WEEKS,
        "splits": split_records
    }
    record_path = output_dir / "split_record.json"
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    print(f"\n已保存: {record_path}")
    print(f"\n数据切分完成！处理了 {len(split_records)} 个标的")
    print("=" * 80)


if __name__ == "__main__":
    main()

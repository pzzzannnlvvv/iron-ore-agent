#!/usr/bin/env python3
"""
对比新旧训练结果，生成可视化报告
"""

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

base_dir = Path(__file__).resolve().parent.parent.parent
improved_output_dir = base_dir / "multi_target_improved" / "outputs"
config_path = base_dir / "multi_target_improved" / "config" / "underperforming_targets.json"


def load_all_records():
    """加载所有标的的训练记录"""
    records = {}
    for target_dir in improved_output_dir.iterdir():
        if target_dir.is_dir():
            record_path = target_dir / "training_record.json"
            if record_path.exists():
                with open(record_path, "r", encoding="utf-8") as f:
                    records[target_dir.name] = json.load(f)
    return records


def generate_comparison(records):
    """生成对比数据"""
    rows = []
    for tid, rec in records.items():
        rows.append({
            "target_id": tid,
            "target_name": rec.get("target_name", ""),
            "group": rec.get("group", ""),
            "original_mda": rec.get("original_mda", 0),
            "new_test_mda": rec.get("new_test_mda", 0),
            "improvement": rec.get("mda_improvement", 0),
            "new_test_mape": rec.get("new_test_mape", 0),
            "feature_count": rec.get("feature_count", 0),
            "train_samples": rec.get("train_samples", 0),
            "optuna_trials": rec.get("optuna_trials", 0),
            "validation_mda": rec.get("new_validation_mda", 0),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("improvement", ascending=False)
    return df


def print_report(df):
    """打印对比报告"""
    print("=" * 100)
    print("多标的MDA提升 — 对比报告")
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

    # 逐标的对比
    print(f"\n{'标的ID':<15s} {'分组':<5s} {'原MDA':<10s} {'新MDA':<10s} {'提升':<10s} {'新MAPE':<10s} {'特征数':<8s}")
    print("-" * 78)
    for _, row in df.iterrows():
        mark = "✓" if row["improvement"] > 0 else "✗"
        print(f"{row['target_id']:<15s} {row['group']:<5s} {row['original_mda']:<10.4f} {row['new_test_mda']:<10.4f} {row['improvement']:+.4f} {mark} {row['new_test_mape']:<10.4f} {row['feature_count']:<8d}")

    # 分组统计
    print("\n" + "=" * 100)
    print("分组统计")
    print("=" * 100)
    for group in sorted(df["group"].unique()):
        group_df = df[df["group"] == group]
        avg_orig = group_df["original_mda"].mean()
        avg_new = group_df["new_test_mda"].mean()
        avg_imp = group_df["improvement"].mean()
        improved_count = (group_df["improvement"] > 0).sum()
        total = len(group_df)
        print(f"\nGroup {group} ({total} 个标的):")
        print(f"  平均原MDA: {avg_orig:.4f}")
        print(f"  平均新MDA: {avg_new:.4f}")
        print(f"  平均提升:  {avg_imp:+.4f}")
        print(f"  提升标的数: {improved_count}/{total}")

    # 总体统计
    print("\n" + "=" * 100)
    print("总体统计")
    print("=" * 100)
    total_orig = df["original_mda"].mean()
    total_new = df["new_test_mda"].mean()
    total_imp = df["improvement"].mean()
    total_improved = (df["improvement"] > 0).sum()
    total_count = len(df)

    print(f"  总标的数: {total_count}")
    print(f"  平均原MDA: {total_orig:.4f}")
    print(f"  平均新MDA: {total_new:.4f}")
    print(f"  平均提升: {total_imp:+.4f}")
    print(f"  提升标的数: {total_improved}/{total_count}")
    print(f"  达到70%标的数: {(df['new_test_mda'] >= 0.70).sum()}/{total_count}")

    # 达标情况
    print("\n" + "=" * 100)
    print("达标情况（MDA ≥ 70%）")
    print("=" * 100)
    above_70 = df[df["new_test_mda"] >= 0.70]
    below_70 = df[df["new_test_mda"] < 0.70]

    if len(above_70) > 0:
        print(f"\n已达标 ({len(above_70)} 个):")
        for _, row in above_70.iterrows():
            print(f"  {row['target_id']} {row['target_name'][:30]:<30s} MDA={row['new_test_mda']:.4f}")

    if len(below_70) > 0:
        print(f"\n仍未达标 ({len(below_70)} 个):")
        for _, row in below_70.iterrows():
            print(f"  {row['target_id']} {row['target_name'][:30]:<30s} MDA={row['new_test_mda']:.4f}")


def save_report(df):
    """保存报告为JSON"""
    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_targets": len(df),
        "overall": {
            "avg_original_mda": float(df["original_mda"].mean()),
            "avg_new_mda": float(df["new_test_mda"].mean()),
            "avg_improvement": float(df["improvement"].mean()),
            "targets_improved": int((df["improvement"] > 0).sum()),
            "targets_above_70": int((df["new_test_mda"] >= 0.70).sum()),
        },
        "by_group": {},
        "targets": []
    }

    for group in sorted(df["group"].unique()):
        group_df = df[df["group"] == group]
        report["by_group"][group] = {
            "count": len(group_df),
            "avg_original_mda": float(group_df["original_mda"].mean()),
            "avg_new_mda": float(group_df["new_test_mda"].mean()),
            "avg_improvement": float(group_df["improvement"].mean()),
            "targets_above_70": int((group_df["new_test_mda"] >= 0.70).sum()),
        }

    for _, row in df.iterrows():
        report["targets"].append({
            "target_id": row["target_id"],
            "target_name": row["target_name"],
            "group": row["group"],
            "original_mda": float(row["original_mda"]),
            "new_test_mda": float(row["new_test_mda"]),
            "improvement": float(row["improvement"]),
            "new_test_mape": float(row["new_test_mape"]),
            "feature_count": int(row["feature_count"]),
        })

    report_path = improved_output_dir / "comparison_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n报告已保存: {report_path}")


def main():
    print("=" * 100)
    print("对比新旧训练结果")
    print("=" * 100)

    records = load_all_records()
    if not records:
        print("未找到训练记录，请先运行 02_retrain.py")
        return

    print(f"找到 {len(records)} 个标的的训练记录")

    df = generate_comparison(records)
    print_report(df)
    save_report(df)

    print("\n" + "=" * 100)
    print("对比完成！")
    print("=" * 100)


if __name__ == "__main__":
    main()

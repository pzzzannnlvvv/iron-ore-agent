#!/usr/bin/env python3
"""
逐标的特征筛选脚本
对每个MDA不达标的标的，用LightGBM importance筛选最有用的特征子集

原理：
- 当前3184个特征是为ID00186052（45港口总库存）量身选的
- 对其他标的，大量特征是噪声，导致过拟合
- 通过逐标的feature importance筛选，保留对每个标的真正有用的特征
"""

import os
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ==================== 路径配置 ====================
base_dir = Path(__file__).resolve().parent.parent.parent
config_path = base_dir / "multi_target_improved" / "config" / "underperforming_targets.json"
multi_target_output_dir = base_dir / "multi_target" / "outputs"
ref_params_path = base_dir / "07_model_training" / "outputs" / "best_params_weekly.json"
output_dir = base_dir / "multi_target_improved" / "outputs"

SEED = 42


def load_config():
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_reference_params():
    with open(ref_params_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_train_data(target_id):
    """加载训练数据，构造 X 和 y"""
    train_path = multi_target_output_dir / target_id / "train_weekly.csv"
    if not train_path.exists():
        print(f"    [错误] 训练文件不存在: {train_path}")
        return None, None, None

    df = pd.read_csv(train_path, encoding="utf-8-sig", low_memory=False)

    # 构造目标：使用 target_diff_1（预测下一期变化量）
    if "target_diff_1" not in df.columns:
        print(f"    [错误] 未找到 target_diff_1 列")
        return None, None, None

    df["_y"] = df["target_diff_1"].shift(-1)
    df = df.dropna(subset=["_y"]).copy()

    # 只排除 target_diff_*、target_pct_* 和 _y，保留 target_lag_* 作为特征
    target_cols = [c for c in df.columns
                   if c.startswith("target_diff_") or c.startswith("target_pct_") or c == "_y"]
    drop_cols = ["data_date"] + target_cols

    X = df.drop(columns=drop_cols, errors="ignore")
    y = df["_y"]
    dates = df["data_date"]

    return X, y, dates


def select_features_for_target(target_id, target_name, group, max_features, ref_params):
    """对单个标的进行特征筛选"""
    print(f"\n  {'='*60}")
    print(f"  标的: {target_id} - {target_name}")
    print(f"  分组: {group}, 目标特征数: ≤{max_features}")

    # 加载数据
    X, y, dates = load_train_data(target_id)
    if X is None:
        return None

    print(f"    数据形状: {X.shape}")
    print(f"    原始特征数: {X.shape[1]}")

    # 用参考参数快速训练一棵LightGBM获取feature importance
    probe_params = {
        "num_leaves": ref_params.get("num_leaves", 63),
        "max_depth": min(ref_params.get("max_depth", 8), 6),  # 限制深度防过拟合
        "learning_rate": 0.05,
        "n_estimators": 200,
        "min_child_samples": max(ref_params.get("min_child_samples", 20), 20),
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": SEED,
        "verbosity": -1,
        "n_jobs": -1,
    }

    print(f"    训练探测模型（200棵树）...")
    model = lgb.LGBMRegressor(**probe_params)
    model.fit(X, y)

    # 提取feature importance
    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    # 筛选：保留importance > 0的特征，最多保留max_features个
    nonzero = importance[importance["importance"] > 0]
    selected = nonzero.head(max_features)

    print(f"    importance > 0 的特征数: {len(nonzero)}")
    print(f"    保留特征数: {len(selected)}")
    print(f"    Top 10 特征:")
    for _, row in selected.head(10).iterrows():
        print(f"      {row['feature']:<40s} importance={row['importance']}")

    # 保存结果
    target_output_dir = output_dir / target_id
    target_output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "target_id": target_id,
        "target_name": target_name,
        "group": group,
        "original_features": X.shape[1],
        "selected_features": len(selected),
        "probe_params": probe_params,
        "feature_list": selected["feature"].tolist(),
        "feature_importance": {
            row["feature"]: int(row["importance"])
            for _, row in selected.iterrows()
        }
    }

    save_path = target_output_dir / "selected_features.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"    已保存: {save_path}")
    return result


def main():
    print("=" * 80)
    print("逐标的特征筛选 - Phase 1")
    print("=" * 80)

    # 加载配置
    config = load_config()
    targets = config["targets"]
    fs_config = config["feature_selection_config"]
    ref_params = load_reference_params()

    print(f"\n待筛选标的数: {len(targets)}")
    print(f"参考参数已加载: {ref_params_path}")

    # 按分组确定最大特征数
    max_features_map = {
        "A": fs_config["max_features_group_A"],
        "B": fs_config["max_features_group_B"],
        "C": fs_config["max_features_group_C"],
        "D": fs_config["max_features_group_D"],
    }

    # 处理每个标的
    results = {}
    for i, target in enumerate(targets):
        target_id = target["target_id"]
        target_name = target["target_name"]
        group = target["group"]
        max_features = max_features_map.get(group, 500)

        print(f"\n[{i+1}/{len(targets)}] 处理中...")
        result = select_features_for_target(
            target_id, target_name, group, max_features, ref_params
        )
        if result:
            results[target_id] = result

    # 保存总记录
    print("\n" + "=" * 80)
    print("保存总记录...")
    summary = {
        "processed_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_targets": len(targets),
        "processed_targets": len(results),
        "results": {
            tid: {
                "target_name": r["target_name"],
                "group": r["group"],
                "original_features": r["original_features"],
                "selected_features": r["selected_features"]
            }
            for tid, r in results.items()
        }
    }

    summary_path = output_dir / "feature_selection_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n已保存: {summary_path}")

    # 打印总结
    print("\n" + "=" * 80)
    print("特征筛选总结")
    print("=" * 80)
    print(f"{'标的ID':<15s} {'分组':<5s} {'原始特征':<10s} {'筛选后':<10s} {'缩减比':<10s}")
    print("-" * 50)
    for tid, r in results.items():
        ratio = r["selected_features"] / r["original_features"] * 100
        print(f"{tid:<15s} {r['group']:<5s} {r['original_features']:<10d} {r['selected_features']:<10d} {ratio:.1f}%")

    print("\n特征筛选完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
lgbm_infer_reproduce_weekly.py — 周频模型复现推理脚本
====================================================
读取最优超参数，用 train 训练，在 test 集上预测并评估。

使用方法：
    python lgbm_infer_reproduce_weekly.py

前置条件：
    必须先运行 optuna_lgbm_optimize_weekly.py 生成 outputs/best_params_weekly.json

输出：
    outputs/test_predictions_weekly.csv — 测试集预测结果（含 MDA/MAPE）
"""

import json
import os
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

warnings.filterwarnings("ignore")

# ============================================================
# 全局随机种子 — 必须与优化脚本一致
# ============================================================
SEED = 42

np.random.seed(SEED)
import random
random.seed(SEED)

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "..", "06_data_split", "outputs")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "train_weekly.csv")
TEST_PATH = os.path.join(DATA_DIR, "test_weekly.csv")
BEST_PARAMS_PATH = os.path.join(OUTPUT_DIR, "best_params_weekly.json")


def mean_directional_accuracy(y_true, y_pred):
    """MDA — 平均方向准确率（与优化脚本完全一致）。"""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    actual_dir = np.sign(y_true[1:] - y_true[:-1])
    pred_dir = np.sign(y_pred[1:] - y_pred[:-1])

    mask = actual_dir != 0
    if mask.sum() == 0:
        return 0.5
    return np.mean(actual_dir[mask] == pred_dir[mask])


def load_data():
    """加载 train/test，分离 X 和 y。"""
    train = pd.read_csv(TRAIN_PATH, header=[0, 1])
    test = pd.read_csv(TEST_PATH, header=[0, 1])

    # 提取列名
    train.columns = list(train.columns.get_level_values(0))
    test.columns = list(test.columns.get_level_values(0))

    # 构造目标列 y[t] = target_lag_1[t+1]
    train["_y"] = train["target_lag_1"].shift(-1)
    test["_y"] = test["target_lag_1"].shift(-1)

    # 丢弃最后一行（y 为 NaN）
    train = train.dropna(subset=["_y"])
    test = test.dropna(subset=["_y"])

    # 分离 X 和 y
    target_cols = [c for c in train.columns if c.startswith("target_") or c == "_y"]
    drop_cols = ["data_date"] + target_cols

    X_train = train.drop(columns=drop_cols, errors="ignore")
    y_train = train["_y"]
    X_test = test.drop(columns=drop_cols, errors="ignore")
    y_test = test["_y"]

    # 确保所有 X 列一致
    common_cols = X_train.columns.intersection(X_test.columns)
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    return X_train, y_train, X_test, y_test, test


def main():
    print("=" * 60)
    print("LGBM 周频复现推理")
    print("=" * 60)

    # 加载最优参数
    if not os.path.exists(BEST_PARAMS_PATH):
        print(f"[错误] 未找到最优参数文件: {BEST_PARAMS_PATH}")
        print("请先运行 optuna_lgbm_optimize_weekly.py")
        return

    with open(BEST_PARAMS_PATH, "r", encoding="utf-8") as f:
        best_params = json.load(f)
    print(f"\n加载最优参数 ({len(best_params)} 个):")
    for k, v in best_params.items():
        if k not in ("random_state", "deterministic", "verbosity"):
            print(f"  {k}: {v}")

    # 加载数据
    X_train, y_train, X_test, y_test, test_df = load_data()

    # 用最优参数训练最终模型
    print("\n训练最终模型...")
    model = lgb.LGBMRegressor(**best_params)
    model.fit(
        X_train, y_train,
    )

    # 预测
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # 计算训练集指标
    mda_train = mean_directional_accuracy(y_train.values, y_pred_train)
    mape_train = mean_absolute_percentage_error(y_train.values, y_pred_train)
    mae_train = np.mean(np.abs(y_train.values - y_pred_train))

    # 计算测试集指标
    mda_test = mean_directional_accuracy(y_test.values, y_pred_test)
    mape_test = mean_absolute_percentage_error(y_test.values, y_pred_test)
    mae_test = np.mean(np.abs(y_test.values - y_pred_test))

    print(f"\n{'='*60}")
    print(f"训练集评估结果")
    print(f"{'='*60}")
    print(f"  MDA:  {mda_train:.4f}  (方向准确率)")
    print(f"  MAPE: {mape_train:.4f}  ({mape_train*100:.2f}%)")
    print(f"  MAE:  {mae_train:.2f}  (万吨)")

    print(f"\n{'='*60}")
    print(f"测试集评估结果")
    print(f"{'='*60}")
    print(f"  MDA:  {mda_test:.4f}  (方向准确率)")
    print(f"  MAPE: {mape_test:.4f}  ({mape_test*100:.2f}%)")
    print(f"  MAE:  {mae_test:.2f}  (万吨)")

    # 保存测试集预测结果
    test_results = pd.DataFrame({
        "data_date": test_df["data_date"].values[:len(y_pred_test)],
        "y_true": y_test.values,
        "y_pred": y_pred_test,
        "abs_error": np.abs(y_test.values - y_pred_test),
    })
    test_pred_path = os.path.join(OUTPUT_DIR, "test_predictions_weekly.csv")
    test_results.to_csv(test_pred_path, index=False, encoding="utf-8-sig")
    print(f"\n[OK] 测试集预测结果已保存: {test_pred_path}")

    # 保存完整评估指标
    summary = {
        "train_mda": float(mda_train),
        "train_mape": float(mape_train),
        "train_mae": float(mae_train),
        "test_mda": float(mda_test),
        "test_mape": float(mape_test),
        "test_mae": float(mae_test),
        "y_range": [float(y_test.min()), float(y_test.max())],
        "train_rows": len(y_train),
        "test_rows": len(y_test),
    }
    summary_path = os.path.join(OUTPUT_DIR, "final_evaluation_weekly.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] 最终评估指标已保存: {summary_path}")
    print(f"\n{'='*60}")
    print("复现完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()


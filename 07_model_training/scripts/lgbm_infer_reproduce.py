#!/usr/bin/env python3
"""
lgbm_infer_reproduce.py — 模型复现推理脚本
===========================================
读取最优超参数，用 train+val 合并训练，在 test 集上预测并评估。

使用方法：
    python lgbm_infer_reproduce.py

前置条件：
    必须先运行 optuna_lgbm_optimize.py 生成 outputs/best_params.json

输出：
    outputs/test_predictions.csv — 测试集预测结果（含 MDA/MAPE）
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

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
VAL_PATH = os.path.join(DATA_DIR, "val.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
BEST_PARAMS_PATH = os.path.join(OUTPUT_DIR, "best_params.json")


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
    """加载 train/val/test，分离 X 和 y。"""
    train = pd.read_csv(TRAIN_PATH)
    val = pd.read_csv(VAL_PATH)
    test = pd.read_csv(TEST_PATH)

    # 构造目标列 y[t] = target_lag_1[t+1]
    train["_y"] = train["target_lag_1"].shift(-1)
    val["_y"] = val["target_lag_1"].shift(-1)
    test["_y"] = test["target_lag_1"].shift(-1)

    # 丢弃最后一行（y 为 NaN）
    train = train.dropna(subset=["_y"])
    val = val.dropna(subset=["_y"])
    test = test.dropna(subset=["_y"])

    # 分离 X 和 y
    target_cols = [c for c in train.columns if c.startswith("target_") or c == "_y"]
    drop_cols = ["data_date"] + target_cols

    X_train = train.drop(columns=drop_cols, errors="ignore")
    y_train = train["_y"]
    X_val = val.drop(columns=drop_cols, errors="ignore")
    y_val = val["_y"]
    X_test = test.drop(columns=drop_cols, errors="ignore")
    y_test = test["_y"]

    # 确保所有 X 列一致
    common_cols = X_train.columns.intersection(X_val.columns).intersection(X_test.columns)
    X_train = X_train[common_cols]
    X_val = X_val[common_cols]
    X_test = X_test[common_cols]

    # 合并 train + val 用于最终训练
    X_train_full = pd.concat([X_train, X_val], axis=0)
    y_train_full = pd.concat([y_train, y_val], axis=0)

    print(f"X_train_full: {X_train_full.shape}, y_train_full: {y_train_full.shape}")
    print(f"X_test:       {X_test.shape}, y_test:       {y_test.shape}")

    return X_train_full, y_train_full, X_test, y_test, test


def main():
    print("=" * 60)
    print("LGBM 复现推理")
    print("=" * 60)

    # 加载最优参数
    if not os.path.exists(BEST_PARAMS_PATH):
        print(f"[错误] 未找到最优参数文件: {BEST_PARAMS_PATH}")
        print("请先运行 optuna_lgbm_optimize.py")
        return

    with open(BEST_PARAMS_PATH, "r", encoding="utf-8") as f:
        best_params = json.load(f)
    print(f"\n加载最优参数 ({len(best_params)} 个):")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # 加载数据
    X_train, y_train, X_test, y_test, test_df = load_data()

    # 用最优参数训练最终模型
    print("\n训练最终模型...")
    model = lgb.LGBMRegressor(**best_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="l1",
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(100),
        ],
    )

    # 预测
    y_pred = model.predict(X_test)

    # 计算指标
    mda = mean_directional_accuracy(y_test.values, y_pred)
    mape = mean_absolute_percentage_error(y_test.values, y_pred)

    print(f"\n{'='*60}")
    print(f"测试集评估结果")
    print(f"{'='*60}")
    print(f"  MDA:  {mda:.4f}  (方向准确率)")
    print(f"  MAPE: {mape:.4f}  ({mape*100:.2f}%)")

    # 保存预测结果
    results = pd.DataFrame({
        "data_date": test_df["data_date"].values[:len(y_pred)],
        "y_true": y_test.values,
        "y_pred": y_pred,
        "abs_error": np.abs(y_test.values - y_pred),
    })
    pred_path = os.path.join(OUTPUT_DIR, "test_predictions.csv")
    results.to_csv(pred_path, index=False, encoding="utf-8-sig")
    print(f"\n[OK] 测试集预测结果已保存: {pred_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
optuna_lgbm_optimize.py — Optuna 超参优化脚本
==============================================
用 Optuna (TPESampler) 自动搜索 LightGBM 最优超参数，
以 MDA（平均方向准确率）为优化目标。

使用方法：
    python optuna_lgbm_optimize.py

输出：
    outputs/optuna_hyperparam_results.csv  — 所有试验的超参 + MDA/MAPE
    outputs/best_params.json               — 最优超参组合
    outputs/optimization_summary.json      — 优化摘要
"""

import json
import os
import time
import warnings

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_percentage_error

warnings.filterwarnings("ignore")

# ============================================================
# 全局随机种子 — 保证可复现
# ============================================================
SEED = 42
N_TRIALS = 200  # 快速验证用 50 次，正式跑改为 200~500

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


def load_data():
    """加载训练集和验证集，分离特征 X 和目标 y。

    目标 y 由 target_lag_1 向后平移 1 天构造：
        y[t] = target_lag_1[t+1]
    最后一行 y 为 NaN，需丢弃。
    """
    train = pd.read_csv(TRAIN_PATH)
    val = pd.read_csv(VAL_PATH)

    # 构造目标列
    train["_y"] = train["target_lag_1"].shift(-1)
    val["_y"] = val["target_lag_1"].shift(-1)

    # 丢弃最后一行（y 为 NaN）
    train = train.dropna(subset=["_y"])
    val = val.dropna(subset=["_y"])

    # 分离 X 和 y：排除 data_date 和所有 target_ 开头的列
    target_cols = [c for c in train.columns if c.startswith("target_") or c == "_y"]
    drop_cols = ["data_date"] + target_cols

    X_train = train.drop(columns=drop_cols, errors="ignore")
    y_train = train["_y"]
    X_val = val.drop(columns=drop_cols, errors="ignore")
    y_val = val["_y"]

    # 确保 X_train 和 X_val 列一致
    common_cols = X_train.columns.intersection(X_val.columns)
    X_train = X_train[common_cols]
    X_val = X_val[common_cols]

    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape}, y_val:   {y_val.shape}")

    return X_train, y_train, X_val, y_val


def mean_directional_accuracy(y_true, y_pred):
    """MDA — 平均方向准确率。

    方向正确 = sign(预测变化) == sign(实际变化)
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    actual_dir = np.sign(y_true[1:] - y_true[:-1])
    pred_dir = np.sign(y_pred[1:] - y_pred[:-1])

    # 排除方向为 0（不变）的情况
    mask = actual_dir != 0
    if mask.sum() == 0:
        return 0.5
    return np.mean(actual_dir[mask] == pred_dir[mask])


def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna 目标函数 — 返回 1-MDA（越小越好）。"""
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": SEED,
        "deterministic": True,
        "verbosity": -1,
        "n_jobs": -1,
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="l1",
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(0),
        ],
    )

    y_pred = model.predict(X_val)
    mda = mean_directional_accuracy(y_val.values, y_pred)

    # 同时记录 MAPE 供参考
    mape = mean_absolute_percentage_error(y_val.values, y_pred)

    trial.set_user_attr("mda", float(mda))
    trial.set_user_attr("mape", float(mape))
    trial.set_user_attr("best_iteration", model.best_iteration_)

    return 1.0 - mda  # 转换为最小化问题


def main():
    print("=" * 60)
    print("Optuna + LightGBM 超参优化")
    print(f"优化目标: MDA  试验次数: {N_TRIALS}  随机种子: {SEED}")
    print("=" * 60)

    # 加载数据
    X_train, y_train, X_val, y_val = load_data()

    # 创建 Optuna study
    sampler = TPESampler(seed=SEED)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name="lgbm_mda_optimization",
    )

    # 优化
    start_time = time.time()
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )
    elapsed = time.time() - start_time

    # ---- 保存结果 ----
    # 1. 全部试验结果 CSV
    results_df = study.trials_dataframe()
    # 添加 user_attrs 列
    for attr in ["mda", "mape", "best_iteration"]:
        results_df[attr] = [
            t.user_attrs.get(attr, None) if t.user_attrs else None
            for t in study.trials
        ]
    results_path = os.path.join(OUTPUT_DIR, "optuna_hyperparam_results.csv")
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    print(f"\n[OK] 试验结果已保存: {results_path}")

    # 2. 最优参数
    best_params = study.best_params
    best_params["random_state"] = SEED
    best_params["deterministic"] = True
    best_params["verbosity"] = -1

    best_path = os.path.join(OUTPUT_DIR, "best_params.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)
    print(f"[OK] 最优参数已保存: {best_path}")

    # 3. 优化摘要
    best_trial = study.best_trial
    summary = {
        "study_name": "lgbm_mda_optimization",
        "seed": SEED,
        "n_trials": N_TRIALS,
        "n_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "elapsed_seconds": round(elapsed, 1),
        "best_mda": round(float(best_trial.user_attrs.get("mda", 1 - study.best_value)), 6),
        "best_mape": round(float(best_trial.user_attrs.get("mape", None)), 6),
        "best_value": round(study.best_value, 6),
        "best_params": best_params,
        "feature_count": X_train.shape[1],
        "train_rows": len(X_train),
        "val_rows": len(X_val),
    }
    summary_path = os.path.join(OUTPUT_DIR, "optimization_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[OK] 优化摘要已保存: {summary_path}")

    # ---- 打印最优结果 ----
    print("\n" + "=" * 60)
    print("最优结果")
    print("=" * 60)
    print(f"  MDA:  {summary['best_mda']:.4f}  (方向准确率)")
    print(f"  MAPE: {summary['best_mape']:.4f}")
    print(f"  完成试验: {summary['n_completed']}  剪枝: {summary['n_pruned']}")
    print(f"  耗时: {elapsed/60:.1f} 分钟")
    print(f"\n最优超参:")
    for k, v in best_params.items():
        if k not in ("random_state", "deterministic", "verbosity"):
            print(f"    {k}: {v}")


if __name__ == "__main__":
    main()

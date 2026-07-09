#!/usr/bin/env python3
"""
optuna_lgbm_optimize_weekly.py - 周频版本的Optuna超参优化

改动点：
- 使用周频数据（train_weekly.csv）
- 无验证集，只用训练集
- 去掉 MedianPruner
- 在训练集上做简单评估

使用方法：
    python optuna_lgbm_optimize_weekly.py

输出：
    outputs/optuna_hyperparam_results_weekly.csv
    outputs/best_params_weekly.json
    outputs/optimization_summary_weekly.json
"""

import json
import os
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import mean_absolute_percentage_error

warnings.filterwarnings("ignore")

# 全局随机种子
SEED = 42
N_TRIALS = 200

np.random.seed(SEED)
import random
random.seed(SEED)

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "..", "06_data_split/outputs")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "train_weekly.csv")


def load_data():
    """加载训练数据，分离特征 X 和目标 y

    目标 y 由 target_lag_1 向后平移 1 周构造：
        y[t] = target_lag_1[t+1]
    最后一行 y 为 NaN，需丢弃。
    """
    train = pd.read_csv(TRAIN_PATH, header=[0, 1])

    # 提取列名
    col_ids = list(train.columns.get_level_values(0))
    col_names = list(train.columns.get_level_values(1))
    train.columns = col_ids

    # 构造目标列
    train["_y"] = train["target_lag_1"].shift(-1)

    # 丢弃最后一行（y 为 NaN）
    train = train.dropna(subset=["_y"])

    # 分离 X 和 y：排除 data_date 和所有 target_ 开头的列
    target_cols = [c for c in train.columns if c.startswith("target_") or c == "_y"]
    drop_cols = ["data_date"] + target_cols

    X_train = train.drop(columns=drop_cols, errors="ignore")
    y_train = train["_y"]

    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")

    return X_train, y_train


def mean_directional_accuracy(y_true, y_pred):
    """MDA - 平均方向准确率

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


def objective(trial, X_train, y_train):
    """Optuna目标函数 - 返回 1-MDA（越小越好）

    周频版本：只用训练集，做训练-验证简单拆分
    """
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

    # 简单的训练-验证拆分：最后 10% 做验证
    n = len(X_train)
    split_idx = int(n * 0.9)

    X_tr = X_train.iloc[:split_idx]
    y_tr = y_train.iloc[:split_idx]
    X_val = X_train.iloc[split_idx:]
    y_val = y_train.iloc[split_idx:]

    model = lgb.LGBMRegressor(**params)
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_val)
    mda = mean_directional_accuracy(y_val.values, y_pred)

    # 同时记录 MAPE 供参考
    mape = mean_absolute_percentage_error(y_val.values, y_pred)

    trial.set_user_attr("mda", float(mda))
    trial.set_user_attr("mape", float(mape))

    return 1.0 - mda


def main():
    print("=" * 70)
    print("Optuna + LightGBM 超参优化（周频版）")
    print(f"优化目标: MDA  试验次数: {N_TRIALS}  随机种子: {SEED}")
    print("=" * 70)

    # 加载数据
    X_train, y_train = load_data()

    # 创建 Optuna study - 无 Pruner（因为验证集很小）
    sampler = optuna.samplers.TPESampler(seed=SEED)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name="lgbm_mda_optimization_weekly",
    )

    # 优化
    start_time = time.time()
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )
    elapsed = time.time() - start_time

    # 保存结果
    results_df = study.trials_dataframe()
    for attr in ["mda", "mape"]:
        results_df[attr] = [
            t.user_attrs.get(attr, None) if t.user_attrs else None
            for t in study.trials
        ]
    results_path = os.path.join(OUTPUT_DIR, "optuna_hyperparam_results_weekly.csv")
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    print(f"\n已保存: {results_path}")

    # 保存最优参数
    best_params = study.best_params
    best_params["random_state"] = SEED
    best_params["deterministic"] = True
    best_params["verbosity"] = -1

    best_path = os.path.join(OUTPUT_DIR, "best_params_weekly.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)
    print(f"已保存: {best_path}")

    # 保存优化摘要
    best_trial = study.best_trial
    summary = {
        "study_name": "lgbm_mda_optimization_weekly",
        "seed": SEED,
        "n_trials": N_TRIALS,
        "n_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "elapsed_seconds": round(elapsed, 1),
        "best_mda": round(float(best_trial.user_attrs.get("mda", 1 - study.best_value)), 6),
        "best_mape": round(float(best_trial.user_attrs.get("mape", None)), 6),
        "best_value": round(study.best_value, 6),
        "best_params": best_params,
        "feature_count": X_train.shape[1],
        "train_rows": len(X_train),
    }
    summary_path = os.path.join(OUTPUT_DIR, "optimization_summary_weekly.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"已保存: {summary_path}")

    # 打印最优结果
    print("\n" + "=" * 70)
    print("最优结果")
    print("=" * 70)
    print(f"  MDA: {summary['best_mda']:.4f}  (方向准确率)")
    print(f"  MAPE: {summary['best_mape']:.4f}")
    print(f"  完成试验: {summary['n_completed']}")
    print(f"  耗时: {elapsed/60:.1f} 分钟")
    print(f"\n最优超参:")
    for k, v in best_params.items():
        if k not in ("random_state", "deterministic", "verbosity"):
            print(f"    {k}: {v}")
    print("\n" + "=" * 70)
    print("优化完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()


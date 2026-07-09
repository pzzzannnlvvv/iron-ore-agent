#!/usr/bin/env python3
"""
多标的模型训练脚本
为每个标的训练独立的LightGBM模型
策略：
- 使用参考标的的最优超参数作为起点（warm start）
- 对每个标的进行小范围Optuna搜索（50次）
- 优化目标：MDA（方向准确率）
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna

warnings.filterwarnings("ignore")

# ==================== 路径配置 ====================
base_dir = Path(__file__).resolve().parent.parent.parent
config_path = base_dir / "multi_target" / "config" / "target_list.json"
ref_best_params_path = base_dir / "07_model_training" / "outputs" / "best_params_weekly.json"
output_dir = base_dir / "multi_target" / "outputs"

# 全局配置
SEED = 42
OPTUNA_TRIALS = 50  # 减少搜索次数，使用warm start


def load_target_list():
    """加载标的配置"""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_reference_best_params():
    """加载参考标的的最优超参数"""
    if ref_best_params_path.exists():
        with open(ref_best_params_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def mean_directional_accuracy(y_true, y_pred):
    """MDA - 平均方向准确率"""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    actual_dir = np.sign(y_true[1:] - y_true[:-1])
    pred_dir = np.sign(y_pred[1:] - y_pred[:-1])

    mask = actual_dir != 0
    if mask.sum() == 0:
        return 0.5
    return np.mean(actual_dir[mask] == pred_dir[mask])


def load_train_data(target_id):
    """加载单个标的的训练数据"""
    train_path = output_dir / target_id / "train_weekly.csv"
    if not train_path.exists():
        return None, None

    df = pd.read_csv(train_path, encoding="utf-8-sig", low_memory=False)

    # 只排除 target_diff_*、target_pct_* 和 _y，保留 target_lag_* 作为特征
    target_cols = [c for c in df.columns
                   if c.startswith("target_diff_") or c.startswith("target_pct_") or c == "_y"]
    drop_cols = ["data_date"] + target_cols

    X = df.drop(columns=drop_cols, errors="ignore")
    y = df["_y"]

    return X, y


def objective(trial, X_train, y_train, ref_params):
    """Optuna目标函数"""
    # 基于参考参数构建搜索空间
    params = {
        "num_leaves": trial.suggest_int("num_leaves",
            max(15, ref_params.get("num_leaves", 63) - 30),
            min(255, ref_params.get("num_leaves", 63) + 30)),
        "max_depth": trial.suggest_int("max_depth",
            max(3, ref_params.get("max_depth", 8) - 3),
            min(15, ref_params.get("max_depth", 8) + 3)),
        "learning_rate": trial.suggest_float("learning_rate",
            max(0.005, ref_params.get("learning_rate", 0.05) * 0.3),
            min(0.3, ref_params.get("learning_rate", 0.05) * 3.0),
            log=True),
        "n_estimators": trial.suggest_int("n_estimators",
            max(100, ref_params.get("n_estimators", 500) - 300),
            min(2000, ref_params.get("n_estimators", 500) + 300)),
        "min_child_samples": trial.suggest_int("min_child_samples",
            max(5, ref_params.get("min_child_samples", 20) - 10),
            min(100, ref_params.get("min_child_samples", 20) + 30)),
        "subsample": trial.suggest_float("subsample",
            max(0.5, ref_params.get("subsample", 0.8) - 0.2),
            min(1.0, ref_params.get("subsample", 0.8) + 0.2)),
        "colsample_bytree": trial.suggest_float("colsample_bytree",
            max(0.5, ref_params.get("colsample_bytree", 0.8) - 0.2),
            min(1.0, ref_params.get("colsample_bytree", 0.8) + 0.2)),
        "reg_alpha": trial.suggest_float("reg_alpha",
            max(1e-8, ref_params.get("reg_alpha", 0.1) * 0.1),
            min(10.0, ref_params.get("reg_alpha", 0.1) * 10.0),
            log=True),
        "reg_lambda": trial.suggest_float("reg_lambda",
            max(1e-8, ref_params.get("reg_lambda", 0.1) * 0.1),
            min(10.0, ref_params.get("reg_lambda", 0.1) * 10.0),
            log=True),
        "random_state": SEED,
        "deterministic": True,
        "verbosity": -1,
        "n_jobs": -1,
    }

    # 简单的训练-验证拆分：最后10%做验证
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

    trial.set_user_attr("mda", float(mda))

    return 1.0 - mda


def train_single_target(target_id, target_name, ref_params):
    """训练单个标的的模型"""
    print(f"\n  训练标的: {target_id} - {target_name}")

    # 加载训练数据
    X_train, y_train = load_train_data(target_id)
    if X_train is None or len(X_train) < 50:
        print(f"    [警告] 训练数据不足，跳过")
        return None

    print(f"    训练数据: {X_train.shape}")

    # 加载测试数据用于最终评估
    test_path = output_dir / target_id / "test_weekly.csv"
    test_df = None
    if test_path.exists():
        test_df = pd.read_csv(test_path, encoding="utf-8-sig", low_memory=False)

    # 确保参考参数存在基础配置
    if ref_params is None:
        ref_params = {
            "num_leaves": 63,
            "max_depth": 8,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        }

    # Optuna优化
    print(f"    Optuna优化: {OPTUNA_TRIALS} 次试验...")
    start_time = time.time()

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=f"lgbm_{target_id}"
    )

    study.optimize(
        lambda trial: objective(trial, X_train, y_train, ref_params),
        n_trials=OPTUNA_TRIALS,
        show_progress_bar=True
    )

    elapsed = time.time() - start_time
    print(f"    优化耗时: {elapsed/60:.1f} 分钟")

    # 获取最优参数
    best_params = study.best_params
    best_params["random_state"] = SEED
    best_params["deterministic"] = True
    best_params["verbosity"] = -1
    best_params["n_jobs"] = -1

    best_trial = study.best_trial
    best_mda = best_trial.user_attrs.get("mda", 1.0 - study.best_value)

    print(f"    最优验证MDA: {best_mda:.4f}")

    # 用最优参数在全部训练集上重训
    print(f"    在全部训练集上重训最终模型...")
    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(X_train, y_train)

    # 在测试集上评估
    test_mda = None
    test_rmse = None
    test_preds = None

    if test_df is not None and len(test_df) > 0:
        # 只排除 target_diff_*、target_pct_* 和 _y，保留 target_lag_* 作为特征
        target_cols = [c for c in test_df.columns
                       if c.startswith("target_diff_") or c.startswith("target_pct_") or c == "_y"]
        drop_cols = ["data_date"] + target_cols
        X_test = test_df.drop(columns=drop_cols, errors="ignore")
        y_test = test_df["_y"]

        y_pred_test = final_model.predict(X_test)
        test_mda = mean_directional_accuracy(y_test.values, y_pred_test)

        # 计算RMSE（diff目标不适合用MAPE）
        from sklearn.metrics import mean_squared_error
        test_rmse = float(np.sqrt(mean_squared_error(y_test.values, y_pred_test)))

        test_preds = test_df[["data_date", "_y"]].copy()
        test_preds["prediction"] = y_pred_test

        print(f"    测试集MDA: {test_mda:.4f}")
        print(f"    测试集RMSE: {test_rmse:.4f}")

    # 保存结果
    target_output_dir = output_dir / target_id

    # 保存模型（用Python写入避免中文路径问题）
    model_path = target_output_dir / "model.txt"
    model_str = final_model.booster_.model_to_string()
    with open(model_path, "w", encoding="utf-8") as f:
        f.write(model_str)

    # 保存最优参数
    params_path = target_output_dir / "best_params.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)

    # 保存测试预测
    if test_preds is not None:
        pred_path = target_output_dir / "test_predictions.csv"
        test_preds.to_csv(pred_path, index=False, encoding="utf-8-sig")

    # 保存训练记录
    train_record = {
        "target_id": target_id,
        "target_name": target_name,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "optuna_trials": OPTUNA_TRIALS,
        "optimization_time_seconds": elapsed,
        "train_samples": len(X_train),
        "feature_count": X_train.shape[1],
        "validation_mda": float(best_mda),
        "test_mda": float(test_mda) if test_mda is not None else None,
        "test_rmse": float(test_rmse) if test_rmse is not None else None,
        "best_params": best_params
    }

    record_path = target_output_dir / "training_record.json"
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(train_record, f, indent=2, ensure_ascii=False)

    print(f"    已保存模型: {model_path}")
    print(f"    已保存记录: {record_path}")

    return train_record


def main():
    print("=" * 80)
    print("多标的模型训练 - Phase 3")
    print("=" * 80)

    # 加载配置
    target_config = load_target_list()
    targets = target_config["targets"]
    ref_params = load_reference_best_params()

    print(f"\nOptuna试验次数: {OPTUNA_TRIALS} (使用warm start)")
    if ref_params:
        print(f"参考参数已加载: {ref_best_params_path}")

    # 检查是否有指定要训练的单个标的
    target_to_train = None
    if len(sys.argv) > 1:
        target_to_train = sys.argv[1]
        print(f"\n指定训练单个标的: {target_to_train}")

    # 处理标的
    print("\n训练模型...")
    train_records = {}

    for i, target in enumerate(targets):
        target_id = target["target_id"]
        target_name = target["target_name"]
        status = target.get("status", "pending")

        # 如果指定了单个标的，只处理那个
        if target_to_train and target_id != target_to_train:
            continue

        if status == "already_trained":
            print(f"\n[{i+1}/{len(targets)}] 跳过已训练标的: {target_id}")
            continue

        # 检查是否已经训练过
        record_path = output_dir / target_id / "training_record.json"
        if record_path.exists():
            print(f"\n[{i+1}/{len(targets)}] 检测到已训练，跳过: {target_id}")
            with open(record_path, "r", encoding="utf-8") as f:
                train_records[target_id] = json.load(f)
            continue

        print(f"\n[{i+1}/{len(targets)}] 准备训练: {target_id} - {target_name}")

        record = train_single_target(target_id, target_name, ref_params)
        if record:
            train_records[target_id] = record

    # 保存总记录
    print("\n保存总记录...")
    summary_record = {
        "summary_created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_targets": len(targets),
        "trained_targets": len(train_records),
        "results": train_records
    }

    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_record, f, indent=2, ensure_ascii=False)

    print(f"\n已保存: {summary_path}")

    # 打印总结
    print("\n" + "=" * 80)
    print("训练总结")
    print("=" * 80)
    print(f"总标的数: {len(targets)}")
    print(f"已训练数: {len(train_records)}")

    if train_records:
        print("\n各标的测试MDA:")
        for tid, rec in train_records.items():
            tname = rec.get("target_name", tid)[:40]
            mda = rec.get("test_mda", "N/A")
            mda_str = f"{mda:.4f}" if mda is not None else "N/A"
            print(f"  {tid:<12s} {tname:<40s} MDA={mda_str}")

    print("\n" + "=" * 80)
    print("训练完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()

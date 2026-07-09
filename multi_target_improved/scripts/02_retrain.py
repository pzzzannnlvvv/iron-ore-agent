#!/usr/bin/env python3
"""
用筛选后的特征重新训练模型
策略：
- 加载逐标的特征筛选结果
- 用筛选后的特征子集训练
- Optuna 100次，增强正则化搜索空间
- 与原始MDA对比
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
config_path = base_dir / "multi_target_improved" / "config" / "underperforming_targets.json"
multi_target_output_dir = base_dir / "multi_target" / "outputs"
output_dir = base_dir / "multi_target_improved" / "outputs"

SEED = 42
OPTUNA_TRIALS = 100


def load_config():
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def load_data(target_id, selected_features=None):
    """加载训练和测试数据，可选特征过滤"""
    train_path = multi_target_output_dir / target_id / "train_weekly.csv"
    test_path = multi_target_output_dir / target_id / "test_weekly.csv"

    if not train_path.exists() or not test_path.exists():
        return None, None, None, None

    train_df = pd.read_csv(train_path, encoding="utf-8-sig", low_memory=False)
    test_df = pd.read_csv(test_path, encoding="utf-8-sig", low_memory=False)

    # 构造目标：使用 target_diff_1（预测下一期变化量）
    for df in [train_df, test_df]:
        if "target_diff_1" in df.columns:
            df["_y"] = df["target_diff_1"].shift(-1)

    train_df = train_df.dropna(subset=["_y"]).copy()
    test_df = test_df.dropna(subset=["_y"]).copy()

    # 只排除 target_diff_*、target_pct_* 和 _y，保留 target_lag_* 作为特征
    target_cols = [c for c in train_df.columns
                   if c.startswith("target_diff_") or c.startswith("target_pct_") or c == "_y"]
    drop_cols = ["data_date"] + target_cols

    X_train = train_df.drop(columns=drop_cols, errors="ignore")
    y_train = train_df["_y"]
    X_test = test_df.drop(columns=drop_cols, errors="ignore")
    y_test = test_df["_y"]

    # 特征筛选
    if selected_features is not None:
        # 只保留筛选后的特征（需要与X_train的列名做交集）
        available = [f for f in selected_features if f in X_train.columns]
        X_train = X_train[available]
        X_test = X_test[available]
        print(f"    特征已筛选: {len(available)}/{len(selected_features)} 个可用")

    return X_train, y_train, X_test, y_test


def objective(trial, X_train, y_train, search_space):
    """Optuna目标函数 - 增强正则化"""
    params = {
        "num_leaves": trial.suggest_int("num_leaves", search_space["num_leaves"][0], search_space["num_leaves"][1]),
        "max_depth": trial.suggest_int("max_depth", search_space["max_depth"][0], search_space["max_depth"][1]),
        "learning_rate": trial.suggest_float("learning_rate", search_space["learning_rate"][0], search_space["learning_rate"][1], log=True),
        "n_estimators": trial.suggest_int("n_estimators", search_space["n_estimators"][0], search_space["n_estimators"][1]),
        "min_child_samples": trial.suggest_int("min_child_samples", search_space["min_child_samples"][0], search_space["min_child_samples"][1]),
        "subsample": trial.suggest_float("subsample", search_space["subsample"][0], search_space["subsample"][1]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", search_space["colsample_bytree"][0], search_space["colsample_bytree"][1]),
        "reg_alpha": trial.suggest_float("reg_alpha", search_space["reg_alpha"][0], search_space["reg_alpha"][1], log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", search_space["reg_lambda"][0], search_space["reg_lambda"][1], log=True),
        "random_state": SEED,
        "deterministic": True,
        "verbosity": -1,
        "n_jobs": -1,
    }

    # 时序验证：最后10%做验证
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


def retrain_single_target(target_id, target_name, group, current_mda, search_space):
    """对单个标的重新训练"""
    print(f"\n  {'='*60}")
    print(f"  标的: {target_id} - {target_name}")
    print(f"  分组: {group}, 当前MDA: {current_mda:.4f}")

    # 加载筛选后的特征列表
    feature_path = output_dir / target_id / "selected_features.json"
    selected_features = None
    if feature_path.exists():
        with open(feature_path, "r", encoding="utf-8") as f:
            feature_data = json.load(f)
        selected_features = feature_data["feature_list"]
        print(f"    使用筛选后特征: {len(selected_features)} 个")
    else:
        print(f"    [警告] 未找到特征筛选结果，使用全部特征")

    # 加载数据
    X_train, y_train, X_test, y_test = load_data(target_id, selected_features)
    if X_train is None:
        return None

    print(f"    训练集: {X_train.shape}, 测试集: {X_test.shape}")

    # Optuna优化
    print(f"    Optuna优化: {OPTUNA_TRIALS} 次试验...")
    start_time = time.time()

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=f"lgbm_improved_{target_id}"
    )

    study.optimize(
        lambda trial: objective(trial, X_train, y_train, search_space),
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

    best_mda = study.best_trial.user_attrs.get("mda", 1.0 - study.best_value)
    print(f"    最优验证MDA: {best_mda:.4f}")

    # 用最优参数在全量训练集上重训
    print(f"    在全量训练集上重训...")
    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(X_train, y_train)

    # 在测试集上评估
    y_pred_test = final_model.predict(X_test)
    test_mda = mean_directional_accuracy(y_test.values, y_pred_test)

    from sklearn.metrics import mean_squared_error
    test_rmse = float(np.sqrt(mean_squared_error(y_test.values, y_pred_test)))

    print(f"    测试集MDA: {test_mda:.4f} (原来: {current_mda:.4f}, 变化: {test_mda - current_mda:+.4f})")
    print(f"    测试集RMSE: {test_rmse:.4f}")

    # 保存结果
    target_output_dir = output_dir / target_id
    target_output_dir.mkdir(parents=True, exist_ok=True)

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
    test_preds = pd.DataFrame({
        "data_date": pd.read_csv(multi_target_output_dir / target_id / "test_weekly.csv",
                                  encoding="utf-8-sig")["data_date"].iloc[-len(y_test):].values,
        "y_true": y_test.values,
        "y_pred": y_pred_test
    })
    pred_path = target_output_dir / "test_predictions.csv"
    test_preds.to_csv(pred_path, index=False, encoding="utf-8-sig")

    # 保存训练记录
    record = {
        "target_id": target_id,
        "target_name": target_name,
        "group": group,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "optuna_trials": OPTUNA_TRIALS,
        "optimization_time_seconds": elapsed,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "feature_count": X_train.shape[1],
        "used_feature_selection": selected_features is not None,
        "original_mda": float(current_mda),
        "new_validation_mda": float(best_mda),
        "new_test_mda": float(test_mda),
        "new_test_rmse": float(test_rmse),
        "mda_improvement": float(test_mda - current_mda),
        "best_params": best_params
    }

    record_path = target_output_dir / "training_record.json"
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    print(f"    已保存: {record_path}")
    return record


def main():
    print("=" * 80)
    print("多标的MDA提升 - Phase 2: 重新训练")
    print("=" * 80)

    # 加载配置
    config = load_config()
    targets = config["targets"]
    search_space = config["optuna_config"]["search_space"]

    print(f"\n待训练标的数: {len(targets)}")
    print(f"Optuna试验次数: {OPTUNA_TRIALS}")
    print(f"搜索空间: reg_alpha={search_space['reg_alpha']}, reg_lambda={search_space['reg_lambda']}")
    print(f"          num_leaves={search_space['num_leaves']}, max_depth={search_space['max_depth']}")

    # 处理每个标的
    records = {}
    for i, target in enumerate(targets):
        target_id = target["target_id"]
        target_name = target["target_name"]
        group = target["group"]
        current_mda = target["current_mda"]

        # 检查是否已处理
        record_path = output_dir / target_id / "training_record.json"
        if record_path.exists():
            print(f"\n[{i+1}/{len(targets)}] 检测到已训练，跳过: {target_id}")
            with open(record_path, "r", encoding="utf-8") as f:
                records[target_id] = json.load(f)
            continue

        print(f"\n[{i+1}/{len(targets)}] 训练中...")
        record = retrain_single_target(
            target_id, target_name, group, current_mda, search_space
        )
        if record:
            records[target_id] = record

    # 保存总报告
    print("\n" + "=" * 80)
    print("保存总报告...")
    summary = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_targets": len(targets),
        "trained_targets": len(records),
        "optuna_trials": OPTUNA_TRIALS,
        "improvements": {}
    }

    for tid, rec in records.items():
        summary["improvements"][tid] = {
            "target_name": rec["target_name"],
            "group": rec["group"],
            "original_mda": rec["original_mda"],
            "new_test_mda": rec["new_test_mda"],
            "improvement": rec["mda_improvement"],
            "feature_count": rec["feature_count"]
        }

    summary_path = output_dir / "improvement_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n已保存: {summary_path}")

    # 打印总结
    print("\n" + "=" * 80)
    print("MDA提升总结")
    print("=" * 80)
    print(f"{'标的ID':<15s} {'分组':<5s} {'原MDA':<10s} {'新MDA':<10s} {'提升':<10s} {'特征数':<8s}")
    print("-" * 58)

    improvements = []
    for tid, rec in records.items():
        imp = rec["mda_improvement"]
        improvements.append(imp)
        mark = "✓" if imp > 0 else "✗"
        print(f"{tid:<15s} {rec['group']:<5s} {rec['original_mda']:<10.4f} {rec['new_test_mda']:<10.4f} {imp:+.4f} {mark} {rec['feature_count']:<8d}")

    if improvements:
        avg_imp = np.mean(improvements)
        print(f"\n平均提升: {avg_imp:+.4f}")
        print(f"提升标的数: {sum(1 for x in improvements if x > 0)}/{len(improvements)}")

    print("\n重新训练完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()

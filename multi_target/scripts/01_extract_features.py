#!/usr/bin/env python3
"""
多标的特征提取脚本
为每个标的生成独立的周频特征文件
策略：
- 复用所有因子特征和时间特征
- 替换标的滞后特征（target_lag_*、target_diff_*、target_pct_*）
- 周频聚合：取每周最后一个交易日
"""

import os
import json
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ==================== 路径配置 ====================
base_dir = Path(__file__).resolve().parent.parent.parent
config_path = base_dir / "multi_target" / "config" / "target_list.json"
daily_data_path = base_dir / "03_data_cleaning" / "outputs" / "cleaned_data_daily.csv"
weekly_features_path = base_dir / "05_feature_engineering" / "outputs" / "09_features_final_weekly.csv"
feature_list_path = base_dir / "05_feature_engineering" / "outputs" / "final_feature_list.csv"
output_dir = base_dir / "multi_target" / "outputs"


def load_target_list():
    """加载标的配置"""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_daily_data():
    """加载日频清洗数据"""
    print("  读取日频清洗数据...")
    # 双表头：跳过第二行（中文名称行），只保留第一行ID作为列名
    df = pd.read_csv(daily_data_path, header=[0], skiprows=[1], encoding="utf-8-sig", low_memory=False)
    # 统一日期格式
    df["data_date"] = pd.to_datetime(df["data_date"]).dt.strftime("%Y-%m-%d")
    print(f"  日频数据形状: {df.shape}")
    return df


def load_feature_list():
    """加载特征列表，区分特征类型"""
    print("  读取特征列表...")
    feature_df = pd.read_csv(feature_list_path, encoding="utf-8-sig")
    target_cols = feature_df[feature_df["类型"] == "target"]["列ID"].tolist()
    time_cols = feature_df[feature_df["类型"] == "time"]["列ID"].tolist()
    factor_cols = feature_df[feature_df["类型"] == "factor"]["列ID"].tolist()
    print(f"  标的特征数: {len(target_cols)}")
    print(f"  时间特征数: {len(time_cols)}")
    print(f"  因子特征数: {len(factor_cols)}")
    return target_cols, time_cols, factor_cols


def load_weekly_template():
    """加载周频特征作为模板（不含标的特征）"""
    print("  读取周频特征模板...")
    # 双表头：跳过第二行，只保留第一行ID作为列名
    template_df = pd.read_csv(weekly_features_path, header=[0], skiprows=[1], encoding="utf-8-sig", low_memory=False)
    # 统一日期格式
    template_df["data_date"] = pd.to_datetime(template_df["data_date"]).dt.strftime("%Y-%m-%d")
    print(f"  周频模板形状: {template_df.shape}")
    return template_df


def generate_target_features(target_series, lags_short, lags_mid, lags_long):
    """
    为单个标的生成滞后、差分、变化率特征
    输入是日频的Series（index是日期）
    """
    n = len(target_series)
    target_data = target_series.values.astype(np.float64)
    features = {}

    # 短期滞后
    for lag in lags_short:
        fid = f"target_lag_{lag}"
        arr = np.full(n, np.nan, dtype=np.float32)
        if lag < n:
            arr[lag:] = target_data[:-lag]
        features[fid] = arr

    # 中期滞后
    for lag in lags_mid:
        fid = f"target_lag_{lag}"
        arr = np.full(n, np.nan, dtype=np.float32)
        if lag < n:
            arr[lag:] = target_data[:-lag]
        features[fid] = arr

    # 长期滞后
    for lag in lags_long:
        fid = f"target_lag_{lag}"
        arr = np.full(n, np.nan, dtype=np.float32)
        if lag < n:
            arr[lag:] = target_data[:-lag]
        features[fid] = arr

    # 差分特征
    for lag in [1, 7]:
        fid = f"target_diff_{lag}"
        arr = np.full(n, np.nan, dtype=np.float32)
        if lag < n:
            arr[lag:] = target_data[lag:] - target_data[:-lag]
        features[fid] = arr

    # 变化率
    for lag in [1, 7]:
        fid = f"target_pct_{lag}"
        arr = np.full(n, np.nan, dtype=np.float32)
        if lag < n:
            safe_denom = np.where(np.abs(target_data[:-lag]) > 1e-10, target_data[:-lag], np.nan)
            arr[lag:] = (target_data[lag:] - target_data[:-lag]) / safe_denom
        features[fid] = arr

    return features


def find_weekly_end_dates(dates):
    """找到每周的最后一个交易日（优先周五）"""
    week_map = {}
    for d_str in dates:
        try:
            # 尝试多种日期格式
            try:
                dt = datetime.strptime(d_str, "%Y-%m-%d")
            except ValueError:
                try:
                    dt = datetime.strptime(d_str, "%Y/%m/%d")
                except ValueError:
                    dt = pd.to_datetime(d_str).to_pydatetime()
        except Exception:
            continue

        week_key = (dt.isocalendar().year, dt.isocalendar().week)
        if week_key not in week_map:
            week_map[week_key] = []
        week_map[week_key].append((dt, d_str))

    selected = []
    for week_key in sorted(week_map.keys()):
        dates_in_week = week_map[week_key]
        # 优先选周五
        fridays = [(dt, d_str) for dt, d_str in dates_in_week if dt.weekday() == 4]
        if fridays:
            selected.append(fridays[-1][1])
        else:
            selected.append(dates_in_week[-1][1])
    return selected


def process_single_target(target_id, target_name, daily_df, weekly_template, target_cols, time_cols, factor_cols):
    """处理单个标的"""
    print(f"\n  处理标的: {target_id} - {target_name}")

    # 1. 提取该标的的日频数据
    if target_id not in daily_df.columns:
        print(f"    [警告] 标的 {target_id} 不在日频数据中，跳过")
        return None

    target_series = daily_df.set_index("data_date")[target_id]

    # 2. 生成日频标的特征
    daily_target_features = generate_target_features(
        target_series,
        lags_short=[1, 2, 3, 5, 7],
        lags_mid=[14, 21, 30],
        lags_long=[60, 90, 180, 365]
    )

    # 3. 创建日频完整特征（仅标的特征，用于聚合）
    daily_feat_df = pd.DataFrame(index=target_series.index, data=daily_target_features)
    daily_feat_df.index.name = "data_date"

    # 4. 找到周频日期
    weekly_dates = find_weekly_end_dates(daily_feat_df.index.tolist())

    # 5. 聚合到周频（取每周最后一天的值）
    weekly_target_feat = daily_feat_df.loc[weekly_dates].copy()
    weekly_target_feat.reset_index(inplace=True)
    print(f"    周频特征行数: {len(weekly_target_feat)}")

    # 6. 与模板合并：保留模板的时间特征、因子特征，替换标的特征
    result_df = weekly_template.copy()

    # 替换标的特征列
    for col in target_cols:
        if col in weekly_target_feat.columns:
            result_df[col] = weekly_target_feat[col].values

    # 7. 保存
    target_output_dir = output_dir / target_id
    target_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_output_dir / "features_weekly.csv"

    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    fsize = os.path.getsize(output_path) / 1024 / 1024
    print(f"    已保存: {output_path} ({fsize:.1f} MB)")

    return output_path


def main():
    print("=" * 80)
    print("多标的特征提取 - Phase 1")
    print("=" * 80)

    # 加载配置
    target_config = load_target_list()
    targets = target_config["targets"]

    print(f"\n总标的数: {len(targets)}")

    # 加载数据
    print("\n[1/5] 加载数据...")
    daily_df = load_daily_data()
    weekly_template = load_weekly_template()
    target_cols, time_cols, factor_cols = load_feature_list()

    # 处理每个标的
    print("\n[2/5] 处理标的...")
    output_paths = {}

    for i, target in enumerate(targets):
        target_id = target["target_id"]
        target_name = target["target_name"]
        status = target.get("status", "pending")

        if status == "already_trained":
            print(f"\n[{i+1}/{len(targets)}] 跳过已训练标的: {target_id}")
            continue

        output_path = process_single_target(
            target_id, target_name, daily_df, weekly_template,
            target_cols, time_cols, factor_cols
        )
        if output_path:
            output_paths[target_id] = str(output_path)

    # 保存处理结果记录
    print("\n[3/5] 保存处理记录...")
    record = {
        "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "outputs": output_paths
    }
    record_path = output_dir / "feature_extraction_record.json"
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    print(f"\n已保存记录: {record_path}")
    print(f"\n特征提取完成！处理了 {len(output_paths)} 个标的")
    print("=" * 80)


if __name__ == "__main__":
    main()

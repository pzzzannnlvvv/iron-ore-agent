#!/usr/bin/env python3
"""
三个问题标的特殊处理：数据清洗+特征提取
集成Winsorize缩尾处理极端值
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
config_path = base_dir / "three_special_targets" / "config" / "target_list.json"
daily_data_path = base_dir / "03_data_cleaning" / "outputs" / "cleaned_data_daily.csv"
weekly_features_path = base_dir / "05_feature_engineering" / "outputs" / "09_features_final_weekly.csv"
feature_list_path = base_dir / "05_feature_engineering" / "outputs" / "final_feature_list.csv"
output_dir = base_dir / "three_special_targets" / "outputs"

# ==================== 数据清洗函数 ====================
def winsorize_series(series, method="percentile", lower=0.01, upper=0.99, n_std=3):
    """
    Winsorize缩尾处理

    参数:
        series: 要处理的Series
        method: "percentile" (分位数) 或 "std" (标准差)
        lower: 下分位数 (percentile方法)
        upper: 上分位数 (percentile方法)
        n_std: 标准差倍数 (std方法)

    返回:
        缩尾后的Series
    """
    # 先去除NaN
    valid_series = series.dropna()
    if len(valid_series) == 0:
        return series

    if method == "percentile":
        lower_val = valid_series.quantile(lower)
        upper_val = valid_series.quantile(upper)
    elif method == "std":
        mean_val = valid_series.mean()
        std_val = valid_series.std()
        lower_val = mean_val - n_std * std_val
        upper_val = mean_val + n_std * std_val
    else:
        raise ValueError(f"Unknown method: {method}")

    return series.clip(lower_val, upper_val)


def clean_target_series(target_series, target_id, method="winsorize"):
    """
    专门针对三个问题标的的数据清洗

    参数:
        target_series: 原始标的序列
        target_id: 标的ID
        method: "winsorize" (缩尾) 或 "fill" (前向填充)

    返回:
        清洗后的Series
    """
    # 只处理这三个标的
    if target_id not in ["ID01517011", "ID01516873", "ID01517220"]:
        return target_series

    cleaned = target_series.copy()

    print(f"    [清洗前] 均值: {cleaned.mean():.2f}, 标准差: {cleaned.std():.2f}, "
          f"最小值: {cleaned.min():.2f}, 最大值: {cleaned.max():.2f}")

    if method == "fill":
        # 方法1：跳变检测+前向填充
        pct_change = cleaned.pct_change()
        abnormal_mask = (pct_change.abs() > 0.5)  # 变化超过50%
        near_zero_mask = (cleaned < 1.0)  # 接近0的值
        full_mask = abnormal_mask | near_zero_mask

        modified_count = full_mask.sum()
        print(f"    [填充法] 检测到 {modified_count} 个异常点")

        cleaned[full_mask] = np.nan
        cleaned = cleaned.fillna(method="ffill")

        # 剩余的用中位数填充
        median_val = cleaned.median()
        cleaned = cleaned.fillna(median_val)

    elif method == "winsorize":
        # 方法2：Winsorize缩尾（推荐）
        # 先对原始值进行缩尾（±3σ）
        cleaned = winsorize_series(cleaned, method="std", n_std=3)
        print(f"    [Winsorize缩尾] 原始值已用±3σ缩尾处理")

    print(f"    [清洗后] 均值: {cleaned.mean():.2f}, 标准差: {cleaned.std():.2f}, "
          f"最小值: {cleaned.min():.2f}, 最大值: {cleaned.max():.2f}")

    return cleaned


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


def generate_target_features(target_series, target_id, lags_short, lags_mid, lags_long):
    """
    为单个标的生成滞后、差分、变化率特征（包含Winsorize处理）
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

    # 百分比变化特征（带Winsorize缩尾）
    for lag in [1, 7]:
        fid = f"target_pct_{lag}"
        arr = np.full(n, np.nan, dtype=np.float32)
        if lag < n:
            safe_denom = np.where(np.abs(target_data[:-lag]) > 1e-10, target_data[:-lag], np.nan)
            arr[lag:] = (target_data[lag:] - target_data[:-lag]) / safe_denom

        # 对百分比变化特征进行Winsorize缩尾（±50%）
        arr = np.clip(arr, -0.5, 0.5)
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
    """处理单个标的（包含数据清洗）"""
    print(f"\n  处理标的: {target_id} - {target_name}")

    # 1. 提取该标的的日频数据
    if target_id not in daily_df.columns:
        print(f"    [警告] 标的 {target_id} 不在日频数据中，跳过")
        return None

    target_series = daily_df.set_index("data_date")[target_id]

    # 2. 数据清洗（关键步骤！）
    print(f"  [数据清洗] 开始处理...")
    cleaned_series = clean_target_series(target_series, target_id, method="winsorize")

    # 3. 生成日频标的特征（使用清洗后的数据）
    daily_target_features = generate_target_features(
        cleaned_series,
        target_id,
        lags_short=[1, 2, 3, 5, 7],
        lags_mid=[14, 21, 30],
        lags_long=[60, 90, 180, 365]
    )

    # 4. 创建日频完整特征（仅标的特征，用于聚合）
    daily_feat_df = pd.DataFrame(index=cleaned_series.index, data=daily_target_features)
    daily_feat_df.index.name = "data_date"

    # 5. 找到周频日期
    weekly_dates = find_weekly_end_dates(daily_feat_df.index.tolist())

    # 6. 聚合到周频（取每周最后一天的值）
    weekly_target_feat = daily_feat_df.loc[weekly_dates].copy()
    weekly_target_feat.reset_index(inplace=True)
    print(f"    周频特征行数: {len(weekly_target_feat)}")

    # 7. 与模板合并：保留模板的时间特征、因子特征，替换标的特征
    result_df = weekly_template.copy()

    # 替换标的特征列
    for col in target_cols:
        if col in weekly_target_feat.columns:
            result_df[col] = weekly_target_feat[col].values

    # 8. 保存
    target_output_dir = output_dir / target_id
    target_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_output_dir / "features_weekly.csv"

    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    fsize = os.path.getsize(output_path) / 1024 / 1024
    print(f"    已保存: {output_path} ({fsize:.1f} MB)")

    return output_path


def main():
    print("=" * 80)
    print("三个问题标的特殊处理 - Phase 1: 数据清洗+特征提取")
    print("=" * 80)

    # 加载配置
    target_config = load_target_list()
    targets = target_config["targets"]

    print(f"\n清洗策略: {target_config.get('cleaning_strategy', {}).get('description', 'Winsorize缩尾')}")
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
        "cleaning_strategy": target_config.get("cleaning_strategy", {}),
        "outputs": output_paths
    }
    record_path = output_dir / "feature_extraction_record.json"
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    print(f"\n已保存: {record_path}")
    print(f"\n特征提取完成！处理了 {len(output_paths)} 个标的")
    print("=" * 80)


if __name__ == "__main__":
    main()

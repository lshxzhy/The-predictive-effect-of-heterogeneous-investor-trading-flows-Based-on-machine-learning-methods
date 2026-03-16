import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import AssetRuntimeContext, get_asset_context
from stats_utils import save_descriptive_stats


@dataclass(frozen=True)
class SharedSplitPlan:
    """共享交易日历上的时间切分方案。"""

    train_end_date: pd.Timestamp
    valid_end_date: pd.Timestamp


def load_merged_raw_panel(ctx: AssetRuntimeContext) -> pd.DataFrame:
    """读取单个资产的原始合并层。"""
    df = pd.read_csv(ctx.paths.merged_raw_panel_file(), parse_dates=["Date"])
    df = df.loc[:, ctx.config.feature_input_required_cols()].copy()
    return df.sort_values("Date").reset_index(drop=True)


def add_required_features(df: pd.DataFrame, ctx: AssetRuntimeContext) -> pd.DataFrame:
    """计算派生特征和多期限标签。"""
    df = df.copy()
    config = ctx.config

    df["cred"] = df["FirmBondAA10Y"] - df["ChBond10Y"]
    df["liqu"] = df["R6M"] - df["ChBond3M"]
    df[config.interaction_feature_name] = df["ITVvar"] * df["dolsha"]

    log_close = np.log(df["close"].replace(0, np.nan))
    daily_log_return = log_close.diff()

    for horizon in config.market_lag_horizons:
        df[config.market_log_return_col(horizon)] = daily_log_return.shift(horizon)
        df[config.market_log_return_abs_col(horizon)] = daily_log_return.abs().shift(horizon)

    for horizon in config.market_stat_horizons:
        df[config.market_vol_col(horizon)] = daily_log_return.rolling(
            horizon,
            min_periods=horizon,
        ).std()
        df[config.market_mom_col(horizon)] = np.log(df["close"] / df["close"].shift(horizon))

    for horizon in config.label_horizons:
        target_return_col = config.target_return_col(horizon)
        target_label_col = config.target_label_col(horizon)

        df[target_return_col] = np.log(df["close"].shift(-horizon) / df["close"])

        target_label = pd.Series(pd.NA, index=df.index, dtype="Int64")
        valid_target_mask = df[target_return_col].notna()
        target_label.loc[valid_target_mask] = (
            df.loc[valid_target_mask, target_return_col] > 0
        ).astype(int)
        df[target_label_col] = target_label

    return df


def trim_common_sample_rows(df: pd.DataFrame, ctx: AssetRuntimeContext) -> pd.DataFrame:
    """在生成全部 Y 后，按固定期限统一裁样本。"""
    df = df.copy()
    common_horizon = ctx.config.common_sample_horizon()
    common_required_cols = [
        ctx.config.target_return_col(common_horizon),
        ctx.config.target_label_col(common_horizon),
    ]
    common_mask = df.loc[:, common_required_cols].notna().all(axis=1)
    trimmed_df = df.loc[common_mask].reset_index(drop=True)
    if trimmed_df.empty:
        raise ValueError(f"{ctx.asset_alias} 在按 {common_horizon}d 统一裁样本后为空。")

    label_check_cols: list[str] = []
    for horizon in ctx.config.label_horizons:
        label_check_cols.extend(
            [
                ctx.config.target_return_col(horizon),
                ctx.config.target_label_col(horizon),
            ]
        )

    missing_counts = trimmed_df.loc[:, label_check_cols].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if not missing_counts.empty:
        raise ValueError(
            f"{ctx.asset_alias} 在按 {common_horizon}d 统一裁样本后仍存在 Y 缺失："
            f"{missing_counts.to_dict()}"
        )

    for horizon in ctx.config.label_horizons:
        target_label_col = ctx.config.target_label_col(horizon)
        trimmed_df[target_label_col] = trimmed_df[target_label_col].astype(int)

    return trimmed_df


def build_shared_common_sample_dates(
    merged_df: pd.DataFrame,
    ctx: AssetRuntimeContext,
) -> pd.Series:
    """基于共享交易日历生成统一裁样本后的日期序列。"""
    common_horizon = ctx.config.common_sample_horizon()
    shared_dates = merged_df["Date"].iloc[:-common_horizon].reset_index(drop=True)
    if shared_dates.empty:
        raise ValueError("共享交易日历为空，无法生成统一样本日期。")
    return shared_dates


def validate_common_sample_dates(
    feature_df: pd.DataFrame,
    shared_dates: pd.Series,
    ctx: AssetRuntimeContext,
) -> None:
    """校验统一样本日期必须属于共享交易日历。"""
    feature_dates = feature_df["Date"].reset_index(drop=True)
    if feature_dates.duplicated().any():
        raise ValueError(f"{ctx.asset_alias} 在统一裁样本后存在重复日期。")

    invalid_dates = feature_dates.loc[~feature_dates.isin(shared_dates)]
    if invalid_dates.empty:
        return

    raise ValueError(
        f"{ctx.asset_alias} 在统一裁样本后存在共享交易日历之外的日期；"
        f"异常日期数 {invalid_dates.nunique()}。"
    )


def drop_research_excluded_columns(
    df: pd.DataFrame,
    ctx: AssetRuntimeContext,
) -> pd.DataFrame:
    """删除只用于中间计算的原始列。"""
    df = df.copy()
    drop_cols = [col for col in ctx.config.rate_source_cols if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def build_shared_split_plan(
    shared_dates: pd.Series,
    ctx: AssetRuntimeContext,
) -> SharedSplitPlan:
    """基于统一样本日期按时间顺序构造全局切分边界。"""
    usable_dates = shared_dates.reset_index(drop=True)
    if usable_dates.empty:
        raise ValueError("统一样本日期为空，无法构造 train/valid/test 切分边界。")

    n_dates = len(usable_dates)
    train_count = int(n_dates * ctx.config.train_ratio)
    valid_count = int(n_dates * ctx.config.valid_ratio)
    test_count = n_dates - train_count - valid_count
    if train_count <= 0 or valid_count <= 0 or test_count <= 0:
        raise ValueError("统一样本长度不足以按 7:1.5:1.5 划分。")

    train_end_idx = train_count - 1
    valid_end_idx = train_count + valid_count - 1

    return SharedSplitPlan(
        train_end_date=pd.Timestamp(usable_dates.iloc[train_end_idx]),
        valid_end_date=pd.Timestamp(usable_dates.iloc[valid_end_idx]),
    )


def apply_shared_split_plan(
    df: pd.DataFrame,
    split_plan: SharedSplitPlan,
    ctx: AssetRuntimeContext,
) -> pd.DataFrame:
    """按共享时间边界固定 train、valid、test。"""
    df = df.copy()
    train_mask = df["Date"] <= split_plan.train_end_date
    valid_mask = (df["Date"] > split_plan.train_end_date) & (df["Date"] <= split_plan.valid_end_date)
    test_mask = df["Date"] > split_plan.valid_end_date

    split_labels = pd.Series(index=df.index, dtype="object")
    split_labels.loc[train_mask] = "train"
    split_labels.loc[valid_mask] = "valid"
    split_labels.loc[test_mask] = "test"
    df["split"] = split_labels
    if df["split"].isna().any():
        raise ValueError(f"{ctx.asset_alias} 在固定 split 时存在未分配样本。")

    split_counts = df["split"].value_counts().to_dict()
    for split_name in ["train", "valid", "test"]:
        if split_counts.get(split_name, 0) == 0:
            raise ValueError(f"{ctx.asset_alias} 在固定 split 后缺少 {split_name} 样本。")

    return df.reset_index(drop=True)


def save_feature_panel(feature_df: pd.DataFrame, ctx: AssetRuntimeContext) -> None:
    """保存统一特征层。"""
    feature_df.to_csv(
        ctx.paths.feature_panel_file(),
        index=False,
        encoding="utf-8-sig",
    )


def save_feature_stats(feature_df: pd.DataFrame, ctx: AssetRuntimeContext) -> None:
    """保存特征层描述统计。"""
    exclude_cols = [
        *[ctx.config.target_return_col(h) for h in ctx.config.label_horizons],
        *[ctx.config.target_label_col(h) for h in ctx.config.label_horizons],
    ]
    save_descriptive_stats(
        df=feature_df,
        desc_file=ctx.paths.feature_desc_stats_file(),
        exclude_cols=exclude_cols,
    )


def impute_missing_with_asset_train_median(
    df: pd.DataFrame,
    feature_cols: list[str],
    ctx: AssetRuntimeContext,
) -> pd.DataFrame:
    """在单个资产内部用训练集的中位数填充筛选特征缺失。"""
    df = df.copy()
    train_mask = df["split"] == "train"
    if train_mask.sum() == 0:
        raise ValueError(f"{ctx.asset_alias} 的筛选输入缺少 train 样本，无法计算中位数。")

    for col in feature_cols:
        if not df[col].isna().any():
            continue

        median_val = df.loc[train_mask, col].median()
        fill_value = 0.0 if pd.isna(median_val) else median_val
        df[col] = df[col].fillna(fill_value)

    return df


def validate_no_missing_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    stage_name: str,
) -> None:
    """校验填充后特征不再存在缺失。"""
    missing_counts = df.loc[:, feature_cols].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if not missing_counts.empty:
        raise ValueError(f"{stage_name} 仍存在特征缺失：{missing_counts.to_dict()}")


def standardize_with_asset_train_stats(
    df: pd.DataFrame,
    feature_cols: list[str],
    ctx: AssetRuntimeContext,
) -> pd.DataFrame:
    """在单个资产内部用训练集统计量标准化筛选特征。"""
    df = df.copy()
    train_mask = df["split"] == "train"

    for col in feature_cols:
        if col in ctx.config.normalized_cols:
            continue

        mean_val = df.loc[train_mask, col].mean()
        std_val = df.loc[train_mask, col].std()
        if pd.isna(std_val) or std_val == 0:
            continue
        df[col] = (df[col] - mean_val) / std_val

    return df


def build_screening_ready_panel(
    feature_df: pd.DataFrame,
    ctx: AssetRuntimeContext,
) -> pd.DataFrame:
    """生成已经在资产内完成预处理的筛选输入面板。"""
    config = ctx.config
    screening_horizon = config.selected_feature_source_horizon()
    feature_cols = config.screening_feature_cols()

    keep_cols = [
        *config.screening_id_cols,
        "split",
        *feature_cols,
        config.target_label_col(screening_horizon),
    ]
    screening_df = feature_df.loc[:, keep_cols].copy()
    screening_df = impute_missing_with_asset_train_median(screening_df, feature_cols, ctx)
    validate_no_missing_features(
        screening_df,
        feature_cols,
        f"{ctx.asset_alias} 的 screening_ready_panel",
    )
    screening_df = standardize_with_asset_train_stats(screening_df, feature_cols, ctx)
    screening_df[config.target_label_col(screening_horizon)] = screening_df[
        config.target_label_col(screening_horizon)
    ].astype(int)
    return screening_df


def save_screening_ready_panel(
    screening_df: pd.DataFrame,
    ctx: AssetRuntimeContext,
) -> None:
    """保存资产级筛选输入表。"""
    screening_df.to_csv(
        ctx.paths.screening_ready_panel_file(),
        index=False,
        encoding="utf-8-sig",
    )


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True)
    return parser.parse_args()


def main() -> None:
    """执行统一特征层和资产级筛选输入的生成。"""
    args = parse_args()
    ctx = get_asset_context(asset_alias=args.asset)

    merged_df = load_merged_raw_panel(ctx)
    feature_df = add_required_features(merged_df, ctx)
    feature_df = trim_common_sample_rows(feature_df, ctx)
    feature_df = drop_research_excluded_columns(feature_df, ctx)
    shared_dates = build_shared_common_sample_dates(merged_df, ctx)
    validate_common_sample_dates(feature_df, shared_dates, ctx)
    split_plan = build_shared_split_plan(shared_dates, ctx)
    feature_df = apply_shared_split_plan(feature_df, split_plan, ctx)
    ctx.paths.ensure_asset_dirs()
    save_feature_panel(feature_df, ctx)
    save_feature_stats(feature_df, ctx)

    screening_df = build_screening_ready_panel(feature_df, ctx)
    save_screening_ready_panel(screening_df, ctx)

    print("统一特征层和资产级筛选输入已保存：")
    print(ctx.paths.feature_panel_file())
    print(ctx.paths.feature_desc_stats_file())
    print(ctx.paths.screening_ready_panel_file())


if __name__ == "__main__":
    main()

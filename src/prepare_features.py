import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import AssetRuntimeContext, get_asset_context
from stats_utils import save_descriptive_stats


@dataclass(frozen=True)
class AssetSplitPlan:
    """单个资产统一样本期上的时间切分方案。"""

    train_end_date: pd.Timestamp
    valid_end_date: pd.Timestamp


# merged_raw_panel 进入特征层前必须具备完整源字段，并按 Date 升序整理成统一输入口径。
def load_merged_raw_panel(ctx: AssetRuntimeContext) -> pd.DataFrame:
    """读取单个资产的原始合并层。"""
    df = pd.read_csv(ctx.paths.merged_raw_panel_file(), parse_dates=["Date"])
    df = df.loc[:, ctx.config.feature_input_required_cols()].copy()
    if df["Date"].isna().any():
        raise ValueError(f"{ctx.asset_alias} 的 merged_raw_panel 存在空 Date。")
    if df["Date"].duplicated().any():
        duplicate_dates = df.loc[df["Date"].duplicated(), "Date"].dt.strftime("%Y-%m-%d").tolist()
        raise ValueError(f"{ctx.asset_alias} 的 merged_raw_panel 存在重复 Date：{duplicate_dates[:10]}")
    return df.sort_values("Date").reset_index(drop=True)


# 对数运算只接受正价格；上市前空档或异常价格先保留成 NaN，
# 让它们通过标签可用性和后续特征校验自然暴露，而不是提前被默认值吞掉。
def build_log_math_close(df: pd.DataFrame, ctx: AssetRuntimeContext) -> pd.Series:
    """返回仅供对数运算使用的 close 序列。"""
    if "close" not in df.columns:
        raise ValueError(f"{ctx.asset_alias} 的 merged_raw_panel 缺少 close。")
    return df["close"].where(df["close"] > 0)


# 统一裁样本之后，保留下来的样本必须全部有有效 close，
# 否则说明这部分无效价格已经实质进入了最终训练口径。
def validate_close_series(df: pd.DataFrame, ctx: AssetRuntimeContext) -> None:
    """校验最终保留样本上的 close 可以安全用于对数变换。"""
    invalid_mask = df["close"].isna() | (df["close"] <= 0)
    if not invalid_mask.any():
        return

    invalid_dates = df.loc[invalid_mask, "Date"].dt.strftime("%Y-%m-%d").tolist()
    preview = invalid_dates[:10]
    raise ValueError(
        f"{ctx.asset_alias} 在最终保留样本上仍存在缺失或非正 close；"
        f"异常日期前 10 个：{preview}"
    )


# daily_log_return_t 定义为 log(close_t / close_t-1)，它本身已经是 Date=t 上收盘后可见的信息。
def build_daily_log_return(df: pd.DataFrame, ctx: AssetRuntimeContext) -> pd.Series:
    """基于 close 构造按 Date=t 对齐的一日对数收益。"""
    log_close = np.log(build_log_math_close(df, ctx))
    return log_close.diff()


# lagged daily return 变量记录的是“过去某一日的一日收益”，
# 因此 lag1 不平移，lag5 平移 4 行，lag22 平移 21 行。
def add_lagged_daily_return_features(
    df: pd.DataFrame,
    daily_log_return: pd.Series,
    ctx: AssetRuntimeContext,
) -> pd.DataFrame:
    """添加一日对数收益及其绝对值的多滞后特征。"""
    df = df.copy()
    for horizon in ctx.config.market_lag_horizons:
        shift_steps = ctx.config.market_lag_shift_steps(horizon)
        df[ctx.config.market_log_return_col(horizon)] = daily_log_return.shift(shift_steps)
        df[ctx.config.market_log_return_abs_col(horizon)] = daily_log_return.abs().shift(shift_steps)
    return df


# vol_h*d 和 mom_h*d 都使用截至 Date=t 收盘已经观察到的历史窗口信息。
def add_market_state_features(
    df: pd.DataFrame,
    daily_log_return: pd.Series,
    ctx: AssetRuntimeContext,
) -> pd.DataFrame:
    """添加滚动波动率和累计动量特征。"""
    df = df.copy()
    close_for_log = build_log_math_close(df, ctx)
    for horizon in ctx.config.market_stat_horizons:
        df[ctx.config.market_vol_col(horizon)] = daily_log_return.rolling(
            window=horizon,
            min_periods=horizon,
        ).std()
        df[ctx.config.market_mom_col(horizon)] = np.log(close_for_log / close_for_log.shift(horizon))
    return df


# 所有 Y_h 都提前挂在 Date=t，统一定义为从 t 收盘到 t+h 收盘的累计对数收益。
def add_future_targets(df: pd.DataFrame, ctx: AssetRuntimeContext) -> pd.DataFrame:
    """添加全部期限的未来累计收益和方向标签。"""
    df = df.copy()
    close_for_log = build_log_math_close(df, ctx)
    for horizon in ctx.config.label_horizons:
        target_return_col = ctx.config.target_return_col(horizon)
        target_label_col = ctx.config.target_label_col(horizon)
        df[target_return_col] = np.log(close_for_log.shift(-horizon) / close_for_log)

        target_label = pd.Series(pd.NA, index=df.index, dtype="Int64")
        valid_mask = df[target_return_col].notna()
        target_label.loc[valid_mask] = (df.loc[valid_mask, target_return_col] > 0).astype(int)
        df[target_label_col] = target_label
    return df


# Date=t 上一次性生成全部 X_t 和全部 Y_h，保证后面所有期限共享同一套时间对齐口径。
def add_required_features(df: pd.DataFrame, ctx: AssetRuntimeContext) -> pd.DataFrame:
    """计算派生特征和多期限标签。"""
    df = df.copy()
    config = ctx.config
    df["cred"] = df["FirmBondAA10Y"] - df["ChBond10Y"]
    df["liqu"] = df["R6M"] - df["ChBond3M"]
    df[config.interaction_feature_name] = df["ITVvar"] * df["dolsha"]

    daily_log_return = build_daily_log_return(df, ctx)
    df = add_lagged_daily_return_features(df, daily_log_return, ctx)
    df = add_market_state_features(df, daily_log_return, ctx)
    df = add_future_targets(df, ctx)
    return df


# 先生成全部 Y_h，再按 99d 的标签可用性统一裁样本，确保保留下来的每一行都共享全期限标签。
def trim_common_sample_rows(df: pd.DataFrame, ctx: AssetRuntimeContext) -> pd.DataFrame:
    """按固定最长期限统一裁样本。"""
    df = df.copy()
    common_horizon = ctx.config.common_sample_horizon()
    required_cols = [
        ctx.config.target_return_col(common_horizon),
        ctx.config.target_label_col(common_horizon),
    ]
    trimmed_df = df.loc[df.loc[:, required_cols].notna().all(axis=1)].reset_index(drop=True)
    if trimmed_df.empty:
        raise ValueError(f"{ctx.asset_alias} 在按 {common_horizon}d 统一裁样本后为空。")

    label_cols: list[str] = []
    for horizon in ctx.config.label_horizons:
        label_cols.extend(
            [
                ctx.config.target_return_col(horizon),
                ctx.config.target_label_col(horizon),
            ]
        )

    missing_counts = trimmed_df.loc[:, label_cols].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if not missing_counts.empty:
        raise ValueError(
            f"{ctx.asset_alias} 在按 {common_horizon}d 统一裁样本后仍存在标签缺失："
            f"{missing_counts.to_dict()}"
        )

    for horizon in ctx.config.label_horizons:
        target_label_col = ctx.config.target_label_col(horizon)
        trimmed_df[target_label_col] = trimmed_df[target_label_col].astype(int)

    return trimmed_df


# 每个资产按自己的 99d 可用样本继续往下走，不再和其他资产做日期交集。
def validate_asset_sample_dates(
    feature_df: pd.DataFrame,
    ctx: AssetRuntimeContext,
) -> None:
    """校验当前资产统一样本期内的日期序列干净且递增。"""
    feature_dates = feature_df["Date"].reset_index(drop=True)
    if feature_dates.isna().any():
        raise ValueError(f"{ctx.asset_alias} 在统一裁样本后仍存在空 Date。")
    if feature_dates.duplicated().any():
        raise ValueError(f"{ctx.asset_alias} 在统一裁样本后存在重复日期。")
    if not feature_dates.is_monotonic_increasing:
        raise ValueError(f"{ctx.asset_alias} 在统一裁样本后的日期未按升序排列。")


# close 和利率原始列只用于中间计算，feature_panel 不允许再保留它们。
def drop_research_excluded_columns(
    df: pd.DataFrame,
    ctx: AssetRuntimeContext,
) -> pd.DataFrame:
    """删除只用于中间计算的原始列。"""
    df = df.copy()
    return df.drop(columns=ctx.config.feature_intermediate_drop_cols())


# 每个资产都在自己按 99d 裁好的样本上按 7:1.5:1.5 划分，
# 不再要求和其他资产共享完全相同的日期边界。
def build_asset_split_plan(
    asset_dates: pd.Series,
    ctx: AssetRuntimeContext,
) -> AssetSplitPlan:
    """基于当前资产的统一样本日期构造时间边界。"""
    usable_dates = asset_dates.reset_index(drop=True)
    if usable_dates.empty:
        raise ValueError("统一样本日期为空，无法构造 train/valid/test 边界。")

    n_dates = len(usable_dates)
    train_count = int(n_dates * ctx.config.train_ratio)
    valid_count = int(n_dates * ctx.config.valid_ratio)
    test_count = n_dates - train_count - valid_count
    if train_count <= 0 or valid_count <= 0 or test_count <= 0:
        raise ValueError("统一样本长度不足以按 7:1.5:1.5 划分。")

    train_end_idx = train_count - 1
    valid_end_idx = train_count + valid_count - 1
    return AssetSplitPlan(
        train_end_date=pd.Timestamp(usable_dates.iloc[train_end_idx]),
        valid_end_date=pd.Timestamp(usable_dates.iloc[valid_end_idx]),
    )


# split 一旦固定，下游不允许再重新切分。
def apply_asset_split_plan(
    df: pd.DataFrame,
    split_plan: AssetSplitPlan,
    ctx: AssetRuntimeContext,
) -> pd.DataFrame:
    """按当前资产自己的时间边界固定 train、valid、test。"""
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


# feature_panel 是统一样本和固定 split 之后的资产级主表。
def save_feature_panel(feature_df: pd.DataFrame, ctx: AssetRuntimeContext) -> None:
    """保存统一特征层。"""
    feature_df.to_csv(
        ctx.paths.feature_panel_file(),
        index=False,
        encoding="utf-8-sig",
    )


# 描述统计只针对数值型 X，显式排除全部标签列。
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


# 资产内缺失填充只允许使用 train 中位数；如果 train 整列全空导致中位数不可得，
# 则按当前约定回填 0，并把这个“训练期无可用信息”的状态显式保留下来。
def impute_missing_with_asset_train_median(
    df: pd.DataFrame,
    feature_cols: list[str],
    ctx: AssetRuntimeContext,
) -> pd.DataFrame:
    """在单资产内使用训练集的中位数填充筛选特征缺失。"""
    df = df.copy()
    train_mask = df["split"] == "train"
    if train_mask.sum() == 0:
        raise ValueError(f"{ctx.asset_alias} 的 screening_ready_panel 缺少 train 样本。")

    for col in feature_cols:
        if not df[col].isna().any():
            continue

        median_val = df.loc[train_mask, col].median()
        if pd.isna(median_val):
            median_val = 0.0
        df[col] = df[col].fillna(median_val)

    return df


# 只要还有缺失，就直接报错停下。
def validate_no_missing_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    stage_name: str,
) -> None:
    """校验特征列已经没有缺失。"""
    missing_counts = df.loc[:, feature_cols].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if not missing_counts.empty:
        raise ValueError(f"{stage_name} 仍存在特征缺失：{missing_counts.to_dict()}")


# 资产内标准化只允许使用 train 均值和标准差；
# 如果某列在训练集上的标准差为 0，则整列标准化结果按 0 处理。
def standardize_with_asset_train_stats(
    df: pd.DataFrame,
    feature_cols: list[str],
    ctx: AssetRuntimeContext,
) -> pd.DataFrame:
    """在单资产内使用训练集统计量标准化筛选特征。"""
    df = df.copy()
    train_mask = df["split"] == "train"

    for col in feature_cols:
        if col in ctx.config.normalized_cols:
            continue

        mean_val = df.loc[train_mask, col].mean()
        std_val = df.loc[train_mask, col].std()
        if pd.isna(std_val):
            raise ValueError(
                f"{ctx.asset_alias} 的 {col} 在训练集上的标准差无效，"
                "无法继续标准化，请先检查上游特征构造。"
            )
        if std_val == 0:
            df[col] = 0.0
            continue
        df[col] = (df[col] - mean_val) / std_val

    return df


# screening_ready_panel 只保留“标识列 + split + 3 个固定保留变量 + 候选变量 + 1d 标签”，
# 并且必须在资产内完成“先按 train 中位数填充、再按 train 统计量标准化”。
def build_screening_ready_panel(
    feature_df: pd.DataFrame,
    ctx: AssetRuntimeContext,
) -> pd.DataFrame:
    """生成资产内完成预处理后的筛选输入表。"""
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


# screening_ready_panel 进入统一筛选前不再重复填充或标准化。
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


# 特征层脚本仍然要求显式传入资产简称。
def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True)
    return parser.parse_args()


# prepare_features 串起“特征生成 -> 统一裁样本 -> 固定 split -> 资产内筛选预处理”整条主线。
def main() -> None:
    """执行统一特征层和资产级筛选输入的生成。"""
    args = parse_args()
    ctx = get_asset_context(asset_alias=args.asset)

    merged_df = load_merged_raw_panel(ctx)
    feature_df = add_required_features(merged_df, ctx)
    feature_df = trim_common_sample_rows(feature_df, ctx)
    validate_close_series(feature_df, ctx)
    feature_df = drop_research_excluded_columns(feature_df, ctx)
    validate_asset_sample_dates(feature_df, ctx)
    split_plan = build_asset_split_plan(feature_df["Date"], ctx)
    feature_df = apply_asset_split_plan(feature_df, split_plan, ctx)

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

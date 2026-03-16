import argparse

import pandas as pd

from config import RuntimeContext, get_runtime_context


def load_feature_panel(ctx: RuntimeContext) -> pd.DataFrame:
    """读取单个资产的统一特征层。"""
    df = pd.read_csv(ctx.paths.feature_panel_file(), parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def parse_bool_series(series: pd.Series) -> pd.Series:
    """把筛选标记统一转成布尔值。"""
    true_values = {"1", "true", "True", "TRUE", "yes", "Yes", "YES"}
    return series.astype(str).isin(true_values)


def load_selected_supplementary_features(ctx: RuntimeContext) -> list[str]:
    """读取统一筛选后的补充控制变量清单。"""
    selected_file = ctx.paths.selected_features_file()
    selected_df = pd.read_csv(selected_file).loc[:, ["feature_name", "selected_for_model"]].copy()

    if selected_df["feature_name"].isna().any():
        raise ValueError(f"{selected_file} 中存在空的 feature_name。")
    if selected_df["feature_name"].duplicated().any():
        dup_cols = selected_df.loc[selected_df["feature_name"].duplicated(), "feature_name"].tolist()
        raise ValueError(f"{selected_file} 中存在重复特征：{dup_cols}")

    selected_cols = selected_df.loc[
        parse_bool_series(selected_df["selected_for_model"]),
        "feature_name",
    ].tolist()

    allowed_cols = set(ctx.config.all_screening_supplementary_cols())
    invalid_cols = [col for col in selected_cols if col not in allowed_cols]
    if invalid_cols:
        raise ValueError(f"筛选结果中存在不属于统一候选特征的列：{invalid_cols}")

    return selected_cols


def collect_model_feature_columns(ctx: RuntimeContext, feature_df: pd.DataFrame) -> list[str]:
    """收集单期限建模使用的特征列。"""
    selected_feature_names = load_selected_supplementary_features(ctx)
    feature_cols = list(dict.fromkeys([*ctx.config.fixed_core_feature_cols, *selected_feature_names]))
    missing_cols = [col for col in feature_cols if col not in feature_df.columns]
    if missing_cols:
        raise ValueError(f"筛选结果中的变量不存在于 feature_panel：{missing_cols}")
    return feature_cols


def select_model_ready_rows(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    ctx: RuntimeContext,
) -> pd.DataFrame:
    """从固定 split 的特征层中提取当前期限建模所需列。"""
    target_return_col = ctx.config.target_return_col(ctx.horizon)
    target_label_col = ctx.config.target_label_col(ctx.horizon)

    keep_cols = [
        "Date",
        "split",
        target_return_col,
        target_label_col,
        *feature_cols,
    ]
    ready_df = feature_df.loc[:, keep_cols].copy()
    missing_target_counts = ready_df.loc[:, [target_return_col, target_label_col]].isna().sum()
    missing_target_counts = missing_target_counts[missing_target_counts > 0]
    if not missing_target_counts.empty:
        raise ValueError(
            f"{ctx.scheme_name} 在固定 split 的 feature_panel 中仍存在当前期限 Y 缺失："
            f"{missing_target_counts.to_dict()}"
        )

    ready_df[target_label_col] = ready_df[target_label_col].astype(int)
    return ready_df


def validate_split_labels(df: pd.DataFrame, ctx: RuntimeContext) -> None:
    """校验特征层中已经固定好的 split 是否完整。"""
    split_values = set(df["split"].dropna().unique().tolist())
    invalid_splits = split_values.difference({"train", "valid", "test"})
    if invalid_splits:
        raise ValueError(f"{ctx.scheme_name} 存在非法 split：{sorted(invalid_splits)}")

    for split_name in ["train", "valid", "test"]:
        if (df["split"] == split_name).sum() == 0:
            raise ValueError(f"{ctx.scheme_name} 缺少 {split_name} 样本。")


def impute_missing_with_train_median(
    df: pd.DataFrame,
    feature_cols: list[str],
    ctx: RuntimeContext,
) -> pd.DataFrame:
    """在固定 split 后用训练集中位数填充特征缺失。"""
    df = df.copy()
    train_mask = df["split"] == "train"
    if train_mask.sum() == 0:
        raise ValueError(f"{ctx.scheme_name} 训练集为空，无法计算中位数填充。")

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
    ctx: RuntimeContext,
) -> None:
    """校验中位数填充后入模特征不再存在缺失。"""
    missing_counts = df.loc[:, feature_cols].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if not missing_counts.empty:
        raise ValueError(f"{ctx.scheme_name} 中位数填充后仍存在特征缺失：{missing_counts.to_dict()}")


def standardize_with_train_stats(
    df: pd.DataFrame,
    feature_cols: list[str],
    ctx: RuntimeContext,
) -> pd.DataFrame:
    """用训练集统计量标准化连续特征。"""
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


def build_model_input(
    prepared_df: pd.DataFrame,
    feature_cols: list[str],
    ctx: RuntimeContext,
) -> pd.DataFrame:
    """基于当前时点特征生成模型输入。"""
    target_label_col = ctx.config.target_label_col(ctx.horizon)
    keep_cols = ["Date", target_label_col, "split", *feature_cols]
    model_input_df = prepared_df.loc[:, keep_cols].copy()
    model_input_df[target_label_col] = model_input_df[target_label_col].astype(int)
    return model_input_df


def save_split_outputs(
    prepared_df: pd.DataFrame,
    model_input_df: pd.DataFrame,
    ctx: RuntimeContext,
) -> None:
    """保存单期限建模数据。"""
    for split_name in ["train", "valid", "test"]:
        split_prepared = prepared_df.loc[prepared_df["split"] == split_name].drop(columns=["split"])
        split_prepared.to_csv(
            ctx.paths.prepared_file(split_name),
            index=False,
            encoding="utf-8-sig",
        )

        split_model_input = model_input_df.loc[
            model_input_df["split"] == split_name
        ].drop(columns=["split"])
        split_model_input.to_csv(
            ctx.paths.model_input_file(split_name),
            index=False,
            encoding="utf-8-sig",
        )


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    """生成单个资产、单个期限的建模数据。"""
    args = parse_args()
    ctx = get_runtime_context(asset_alias=args.asset, horizon=args.horizon)

    feature_df = load_feature_panel(ctx)
    feature_cols = collect_model_feature_columns(ctx, feature_df)
    prepared_df = select_model_ready_rows(feature_df, feature_cols, ctx)
    validate_split_labels(prepared_df, ctx)
    prepared_df = impute_missing_with_train_median(prepared_df, feature_cols, ctx)
    validate_no_missing_features(prepared_df, feature_cols, ctx)
    prepared_df = standardize_with_train_stats(prepared_df, feature_cols, ctx)
    model_input_df = build_model_input(prepared_df, feature_cols, ctx)

    ctx.paths.ensure_horizon_dirs()
    save_split_outputs(
        prepared_df=prepared_df,
        model_input_df=model_input_df,
        ctx=ctx,
    )

    print("单期限建模数据已保存：")
    print(ctx.paths.prepared_file("train"))
    print(ctx.paths.prepared_file("valid"))
    print(ctx.paths.prepared_file("test"))
    print(ctx.paths.model_input_file("train"))
    print(ctx.paths.model_input_file("valid"))
    print(ctx.paths.model_input_file("test"))
    print(f"当前方案：{ctx.scheme_name}")


if __name__ == "__main__":
    main()

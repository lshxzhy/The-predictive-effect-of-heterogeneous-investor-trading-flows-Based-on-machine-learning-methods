import argparse

import pandas as pd

from config import RuntimeContext, get_runtime_context


# 单期限建模层直接复用已经固定好 split 的 feature_panel，不允许再切分或再裁样本。
def load_feature_panel(ctx: RuntimeContext) -> pd.DataFrame:
    """读取单个资产的统一特征层。"""
    df = pd.read_csv(ctx.paths.feature_panel_file(), parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


# selected_features.csv 中的布尔列可能来自 csv 文本，先统一转成布尔值。
def parse_bool_series(series: pd.Series) -> pd.Series:
    """把筛选标记统一转成布尔值。"""
    true_values = {"1", "true", "True", "TRUE", "yes", "Yes", "YES"}
    return series.astype(str).isin(true_values)


# 单期限训练只读取统一筛选之后仍然保留的补充候选变量。
def load_selected_supplementary_features(ctx: RuntimeContext) -> list[str]:
    """读取统一筛选保留下来的补充变量列表。"""
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


# 建模列由“固定保留 3 列 + 统一筛选保留列”组成，所有列都必须真实存在于 feature_panel。
def collect_model_feature_columns(ctx: RuntimeContext, feature_df: pd.DataFrame) -> list[str]:
    """收集单期限建模使用的特征列。"""
    selected_feature_names = load_selected_supplementary_features(ctx)
    feature_cols = list(dict.fromkeys([*ctx.config.fixed_core_feature_cols, *selected_feature_names]))
    missing_cols = [col for col in feature_cols if col not in feature_df.columns]
    if missing_cols:
        raise ValueError(f"筛选结果中的这些变量不存在于 feature_panel：{missing_cols}")
    return feature_cols


# 当前期限只从 feature_panel 抽取当前 Y 和最终建模特征列。
def select_model_ready_rows(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    ctx: RuntimeContext,
) -> pd.DataFrame:
    """从 feature_panel 提取当前期限建模所需列。"""
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
            f"{ctx.scheme_name} 的 feature_panel 仍存在当前期限标签缺失："
            f"{missing_target_counts.to_dict()}"
        )

    ready_df[target_label_col] = ready_df[target_label_col].astype(int)
    return ready_df


# split 必须已经在特征层固定好，单期限层不允许再切分。
def validate_split_labels(df: pd.DataFrame, ctx: RuntimeContext) -> None:
    """校验 split 标签是否完整。"""
    split_values = set(df["split"].dropna().unique().tolist())
    invalid_splits = split_values.difference({"train", "valid", "test"})
    if invalid_splits:
        raise ValueError(f"{ctx.scheme_name} 存在非法 split：{sorted(invalid_splits)}")

    for split_name in ["train", "valid", "test"]:
        if (df["split"] == split_name).sum() == 0:
            raise ValueError(f"{ctx.scheme_name} 缺少 {split_name} 样本。")


# 单期限缺失填充只允许使用 train 中位数；如果 train 整列全空导致中位数不可得，
# 则按当前约定回填 0，并把这个“训练期无可用信息”的状态显式保留下来。
def impute_missing_with_train_median(
    df: pd.DataFrame,
    feature_cols: list[str],
    ctx: RuntimeContext,
) -> pd.DataFrame:
    """使用训练集中的中位数填充特征缺失。"""
    df = df.copy()
    train_mask = df["split"] == "train"
    if train_mask.sum() == 0:
        raise ValueError(f"{ctx.scheme_name} 训练集为空，无法计算中位数。")

    for col in feature_cols:
        if not df[col].isna().any():
            continue

        median_val = df.loc[train_mask, col].median()
        if pd.isna(median_val):
            median_val = 0.0
        df[col] = df[col].fillna(median_val)

    return df


# 只要还有缺失就直接报错，避免错误特征继续进入训练。
def validate_no_missing_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    ctx: RuntimeContext,
) -> None:
    """校验中位数填充后入模特征不再存在缺失。"""
    missing_counts = df.loc[:, feature_cols].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if not missing_counts.empty:
        raise ValueError(f"{ctx.scheme_name} 填充后仍存在特征缺失：{missing_counts.to_dict()}")


# 标准化只允许使用 train 统计量；
# 如果某列在训练集上的标准差为 0，则整列标准化结果按 0 处理。
def standardize_with_train_stats(
    df: pd.DataFrame,
    feature_cols: list[str],
    ctx: RuntimeContext,
) -> pd.DataFrame:
    """使用训练集统计量标准化连续特征。"""
    df = df.copy()
    train_mask = df["split"] == "train"

    for col in feature_cols:
        if col in ctx.config.normalized_cols:
            continue

        mean_val = df.loc[train_mask, col].mean()
        std_val = df.loc[train_mask, col].std()
        if pd.isna(std_val):
            raise ValueError(
                f"{ctx.scheme_name} 的 {col} 在训练集上的标准差无效，"
                "无法继续标准化，请先检查上游特征构造。"
            )
        if std_val == 0:
            df[col] = 0.0
            continue
        df[col] = (df[col] - mean_val) / std_val

    return df


# model_input 只保留 Date、标签和最终标准化后的特征列。
def build_model_input(
    prepared_df: pd.DataFrame,
    feature_cols: list[str],
    ctx: RuntimeContext,
) -> pd.DataFrame:
    """生成当前期限的模型输入表。"""
    target_label_col = ctx.config.target_label_col(ctx.horizon)
    keep_cols = ["Date", target_label_col, "split", *feature_cols]
    model_input_df = prepared_df.loc[:, keep_cols].copy()
    model_input_df[target_label_col] = model_input_df[target_label_col].astype(int)
    return model_input_df


# 单期限层只落盘固定 split 下的 prepared 和 model_input 两组文件。
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


# 批量训练会重复调用这条准备链，所以这里把完整流程收口成一个可复用函数。
def prepare_model_data_for_context(ctx: RuntimeContext) -> None:
    """生成单资产单期限的建模数据并落盘。"""
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


# 单期限层脚本仍然要求显式传入资产和期限。
def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    return parser.parse_args()


# 主入口只负责当前资产、当前期限的数据准备和落盘。
def main() -> None:
    """生成单个资产、单个期限的建模数据。"""
    args = parse_args()
    ctx = get_runtime_context(asset_alias=args.asset, horizon=args.horizon)
    prepare_model_data_for_context(ctx)

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

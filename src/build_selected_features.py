import argparse

import pandas as pd

from config import HorizonRuntimeContext, get_horizon_context


# 正式删列必须显式给出 --drop-cols，且只能删统一筛选候选变量里的列。
def parse_drop_cols(raw_drop_cols: str, allowed_cols: list[str]) -> list[str]:
    """解析人工指定的删除变量列表。"""
    drop_cols = [item.strip() for item in raw_drop_cols.split(",") if item.strip()]
    if len(set(drop_cols)) != len(drop_cols):
        raise ValueError(f"--drop-cols 中存在重复变量：{drop_cols}")

    invalid_cols = [col for col in drop_cols if col not in allowed_cols]
    if invalid_cols:
        raise ValueError(f"--drop-cols 中存在不属于统一筛选候选的变量：{invalid_cols}")
    return drop_cols


# 读取 screening_long_panel 主要是强制校验文件存在和候选列集合完整，
# selected_features.csv 不再从别处猜测候选变量名单。
def load_screening_long_panel(ctx: HorizonRuntimeContext) -> pd.DataFrame:
    """读取统一筛选长面板并校验候选变量集合。"""
    return pd.read_csv(ctx.paths.screening_long_panel_file()).loc[
        :,
        ctx.config.all_screening_supplementary_cols(),
    ]


# 重要性表至少要有“变量名 + 重要性值”两列，
# 如果落地文件不再保存排名，就在这里按重要性重新生成 importance_rank。
def load_importance_report(ctx: HorizonRuntimeContext) -> pd.DataFrame:
    """读取筛选模型输出的变量重要性表。"""
    file_path = ctx.paths.screening_output_file("screening_lgbm_importance.csv")
    df = pd.read_csv(file_path)
    required_cols = {"feature_name", "importance"}
    missing_cols = sorted(required_cols.difference(df.columns))
    if missing_cols:
        raise ValueError(f"{file_path} 缺少这些关键列：{missing_cols}")

    df = df.loc[:, ["feature_name", "importance"]].copy()
    if df["feature_name"].duplicated().any():
        dup_cols = df.loc[df["feature_name"].duplicated(), "feature_name"].tolist()
        raise ValueError(f"{file_path} 中存在重复变量：{dup_cols}")
    df = df.sort_values(["importance", "feature_name"], ascending=[False, True]).reset_index(drop=True)
    df["importance_rank"] = df.index + 1
    return df


# 缺失报告只抽取正式删列需要的几个字段，
# 让 selected_features.csv 同时带上“变量重要性”和“填充前缺失率”两条依据。
def load_missing_report(ctx: HorizonRuntimeContext) -> pd.DataFrame:
    """读取筛选层填充前缺失比例报告。"""
    file_path = ctx.paths.screening_output_file("screening_missing_before_imputation.csv")
    df = pd.read_csv(file_path).loc[
        :,
        [
            "feature_name",
            "feature_group",
            "overall_missing_ratio",
            "train_missing_ratio",
            "valid_missing_ratio",
            "test_missing_ratio",
        ],
    ].copy()
    if df["feature_name"].duplicated().any():
        dup_cols = df.loc[df["feature_name"].duplicated(), "feature_name"].tolist()
        raise ValueError(f"{file_path} 中存在重复变量：{dup_cols}")
    return df


# selected_features.csv 只面向 30 个补充候选变量，
# 通过手动 drop 把“是否保留、为何删除、重要性、缺失率”整理到一张正式表里。
def build_selected_feature_frame(
    ctx: HorizonRuntimeContext,
    drop_cols: list[str],
) -> pd.DataFrame:
    """按人工删除列和筛选报告生成正式的 selected_features.csv。"""
    candidate_cols = ctx.config.all_screening_supplementary_cols()
    load_screening_long_panel(ctx)

    importance_df = load_importance_report(ctx)
    missing_df = load_missing_report(ctx)

    selected_df = pd.DataFrame({"feature_name": candidate_cols})
    selected_df = selected_df.merge(importance_df, on="feature_name", how="left")
    selected_df = selected_df.merge(missing_df, on="feature_name", how="left")

    unresolved_missing_report_cols = selected_df.loc[
        selected_df["feature_group"].isna(),
        "feature_name",
    ].tolist()
    if unresolved_missing_report_cols:
        raise ValueError(f"缺失比例报告缺少这些筛选候选变量：{unresolved_missing_report_cols}")

    unresolved_importance_cols = selected_df.loc[
        selected_df["importance"].isna(),
        "feature_name",
    ].tolist()
    if unresolved_importance_cols:
        raise ValueError(f"重要性文件缺少这些筛选候选变量：{unresolved_importance_cols}")

    selected_df["selected_for_model"] = ~selected_df["feature_name"].isin(drop_cols)
    selected_df["manually_dropped"] = selected_df["feature_name"].isin(drop_cols)
    selected_df["drop_reason"] = selected_df["manually_dropped"].map(
        lambda flag: "manual_drop" if flag else ""
    )
    selected_df = selected_df.sort_values(
        ["selected_for_model", "importance_rank", "feature_name"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    return selected_df


# 正式删列脚本必须显式给出 horizon 和 drop-cols。
def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--drop-cols", required=True)
    return parser.parse_args()


# 当前 selected_features.csv 固定从 1d 筛选结果生成，
# 这里只落正式删列结果，不重新训练模型。
def main() -> None:
    """基于人工删除列列表生成正式筛选结果文件。"""
    args = parse_args()
    ctx = get_horizon_context(horizon=args.horizon)
    screening_horizon = ctx.config.selected_feature_source_horizon()
    if args.horizon != screening_horizon:
        raise ValueError(
            f"正式筛选结果固定使用 {screening_horizon}d，请显式传入 --horizon {screening_horizon}"
        )

    candidate_cols = ctx.config.all_screening_supplementary_cols()
    drop_cols = parse_drop_cols(args.drop_cols, candidate_cols)
    selected_df = build_selected_feature_frame(ctx, drop_cols)

    ctx.paths.ensure_screening_data_dir()
    output_file = ctx.paths.selected_features_file()
    selected_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("正式筛选结果已保存：")
    print(output_file)


if __name__ == "__main__":
    main()

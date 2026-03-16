import argparse

import pandas as pd

from config import PipelineConfig, get_asset_context, get_horizon_context


def resolve_asset_aliases(asset_arg: str) -> list[str]:
    """解析需要统计的资产简称列表。"""
    asset_aliases = [item.strip() for item in asset_arg.split(",") if item.strip()]
    if not asset_aliases:
        raise ValueError("至少需要一个资产简称。")
    if len(set(asset_aliases)) != len(asset_aliases):
        raise ValueError(f"--assets 中存在重复资产简称：{asset_aliases}")
    return asset_aliases


def screening_feature_groups(config: PipelineConfig) -> list[tuple[str, str]]:
    """返回统一筛选候选变量及其分组。"""
    return [
        (col, "supplementary")
        for col in config.all_screening_supplementary_cols()
    ]


def load_feature_panel(asset_alias: str) -> pd.DataFrame:
    """读取单个资产的特征层文件。"""
    ctx = get_asset_context(asset_alias=asset_alias)
    df = pd.read_csv(ctx.paths.feature_panel_file(), parse_dates=["Date"])
    return df.loc[
        :,
        [
            "Date",
            "split",
            *ctx.config.all_screening_supplementary_cols(),
        ],
    ].copy()


def build_missing_row(
    df: pd.DataFrame,
    feature_name: str,
    feature_group: str,
    asset_alias: str | None,
) -> dict[str, object]:
    """统计单个变量在填充前的缺失比例。"""
    row: dict[str, object] = {
        "feature_name": feature_name,
        "feature_group": feature_group,
    }
    if asset_alias is not None:
        row["asset_alias"] = asset_alias

    total_count = len(df)
    missing_count = int(df[feature_name].isna().sum())
    row["overall_total_rows"] = total_count
    row["overall_missing_count"] = missing_count
    row["overall_missing_ratio"] = missing_count / total_count if total_count else 0.0

    for split_name in ["train", "valid", "test"]:
        split_df = df.loc[df["split"] == split_name]
        split_total = len(split_df)
        split_missing = int(split_df[feature_name].isna().sum())
        row[f"{split_name}_total_rows"] = split_total
        row[f"{split_name}_missing_count"] = split_missing
        row[f"{split_name}_missing_ratio"] = split_missing / split_total if split_total else 0.0

    return row


def build_missing_reports(asset_aliases: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成 pooled 和按资产拆分的缺失比例报告。"""
    config = PipelineConfig()
    grouped_cols = screening_feature_groups(config)

    pooled_frames: list[pd.DataFrame] = []
    by_asset_rows: list[dict[str, object]] = []
    for asset_alias in asset_aliases:
        asset_df = load_feature_panel(asset_alias)
        pooled_frames.append(asset_df.loc[:, ["split", *[col for col, _ in grouped_cols]]].copy())

        for feature_name, feature_group in grouped_cols:
            by_asset_rows.append(
                build_missing_row(
                    df=asset_df,
                    feature_name=feature_name,
                    feature_group=feature_group,
                    asset_alias=asset_alias,
                )
            )

    pooled_df = pd.concat(pooled_frames, ignore_index=True)
    pooled_rows = [
        build_missing_row(
            df=pooled_df,
            feature_name=feature_name,
            feature_group=feature_group,
            asset_alias=None,
        )
        for feature_name, feature_group in grouped_cols
    ]
    pooled_report_df = pd.DataFrame(pooled_rows).sort_values(
        ["feature_group", "overall_missing_ratio", "feature_name"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    by_asset_report_df = pd.DataFrame(by_asset_rows).sort_values(
        ["asset_alias", "feature_group", "overall_missing_ratio", "feature_name"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)
    return pooled_report_df, by_asset_report_df


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    """输出统一筛选前各变量的缺失比例报告。"""
    args = parse_args()
    ctx = get_horizon_context(horizon=args.horizon)
    screening_horizon = ctx.config.selected_feature_source_horizon()
    if args.horizon != screening_horizon:
        raise ValueError(
            f"缺失比例检查固定使用 {screening_horizon}d，请显式传入 --horizon {screening_horizon}"
        )

    asset_aliases = resolve_asset_aliases(args.assets)
    pooled_report_df, by_asset_report_df = build_missing_reports(asset_aliases)

    ctx.paths.ensure_screening_output_dir()
    pooled_file = ctx.paths.screening_output_file("screening_missing_before_imputation.csv")
    by_asset_file = ctx.paths.screening_output_file("screening_missing_before_imputation_by_asset.csv")
    legacy_file = ctx.paths.screening_output_file("screening_missing_before_imputation_by_future.csv")
    pooled_report_df.to_csv(pooled_file, index=False, encoding="utf-8-sig")
    by_asset_report_df.to_csv(by_asset_file, index=False, encoding="utf-8-sig")
    if legacy_file.exists():
        legacy_file.unlink()

    print("筛选层填充前缺失比例报告已保存：")
    print(pooled_file)
    print(by_asset_file)


if __name__ == "__main__":
    main()

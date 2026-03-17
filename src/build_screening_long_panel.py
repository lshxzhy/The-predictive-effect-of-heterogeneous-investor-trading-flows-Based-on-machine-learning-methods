import argparse

import pandas as pd

from config import HorizonRuntimeContext, PipelineConfig, get_asset_context, get_horizon_context


# 统一筛选层显式接收资产列表，并在入口就拦住空列表或重复资产。
def resolve_asset_aliases(asset_arg: str) -> list[str]:
    """解析需要拼接的资产简称列表。"""
    asset_aliases = [item.strip() for item in asset_arg.split(",") if item.strip()]
    if not asset_aliases:
        raise ValueError("至少需要一个资产简称。")
    if len(set(asset_aliases)) != len(asset_aliases):
        raise ValueError(f"--assets 中存在重复资产简称：{asset_aliases}")
    return asset_aliases


# 读取单资产 screening_ready_panel；这里默认资产内的缺失填充和标准化已经完成，
# 本脚本只负责复核后再做跨资产拼接。
def load_single_screening_panel(
    base_ctx: HorizonRuntimeContext,
    asset_alias: str,
) -> pd.DataFrame:
    """读取单个资产的筛选输入表。"""
    asset_ctx = get_asset_context(
        project_root=base_ctx.paths.project_root,
        asset_alias=asset_alias,
    )
    return pd.read_csv(asset_ctx.paths.screening_ready_panel_file(), parse_dates=["Date"])


# 先把每个资产按 train/valid/test 分桶，再分别收集三段面板，
# 这样统一长面板的顺序就是“全部 train -> 全部 valid -> 全部 test”。
def collect_split_frames(
    base_ctx: HorizonRuntimeContext,
    asset_aliases: list[str],
) -> dict[str, list[pd.DataFrame]]:
    """收集已经在资产内完成处理的 train、valid、test 面板。"""
    config = base_ctx.config
    screening_horizon = config.selected_feature_source_horizon()
    feature_cols = config.all_screening_supplementary_cols()
    label_col = config.target_label_col(screening_horizon)
    keep_cols = [
        *config.screening_id_cols,
        "split",
        *feature_cols,
        label_col,
    ]

    split_frames: dict[str, list[pd.DataFrame]] = {"train": [], "valid": [], "test": []}
    for asset_alias in asset_aliases:
        panel_df = load_single_screening_panel(base_ctx, asset_alias).loc[:, keep_cols].copy()
        split_values = set(panel_df["split"].dropna().unique().tolist())
        invalid_splits = split_values.difference({"train", "valid", "test"})
        if invalid_splits:
            raise ValueError(f"{asset_alias} 的 screening_ready_panel 存在非法 split：{sorted(invalid_splits)}")

        if panel_df[label_col].isna().any():
            raise ValueError(f"{asset_alias} 的 screening_ready_panel 存在 1d 标签缺失。")
        panel_df[label_col] = panel_df[label_col].astype(int)

        missing_counts = panel_df.loc[:, feature_cols].isna().sum()
        missing_counts = missing_counts[missing_counts > 0]
        if not missing_counts.empty:
            raise ValueError(
                f"{asset_alias} 的 screening_ready_panel 仍存在未处理的特征缺失："
                f"{missing_counts.to_dict()}"
            )

        for split_name in ["train", "valid", "test"]:
            split_df = panel_df.loc[panel_df["split"] == split_name].copy()
            if split_df.empty:
                raise ValueError(f"{asset_alias} 的 screening_ready_panel 缺少 {split_name} 样本。")
            split_frames[split_name].append(split_df)

    return split_frames


# 统一长面板是进入筛选模型的直接输入，因此在拼接后再做一次无缺失硬校验。
def validate_no_missing_features(df: pd.DataFrame, feature_cols: list[str]) -> None:
    """校验拼接后的统一筛选长面板不再存在特征缺失。"""
    missing_counts = df.loc[:, feature_cols].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if not missing_counts.empty:
        raise ValueError(f"统一筛选长面板仍存在特征缺失：{missing_counts.to_dict()}")


# 长面板只拼接补充候选变量，不带固定保留的 3 个交易流指标，
# 因为那 3 列不参与统一筛选淘汰，只在后续正式建模时强制保留。
def build_long_panel(
    base_ctx: HorizonRuntimeContext,
    asset_aliases: list[str],
) -> pd.DataFrame:
    """拼接已经在各资产内部预处理完成的统一筛选长面板。"""
    config = base_ctx.config
    screening_horizon = config.selected_feature_source_horizon()
    label_col = config.target_label_col(screening_horizon)
    split_frames = collect_split_frames(base_ctx, asset_aliases)

    long_panel_df = pd.concat(
        [*split_frames["train"], *split_frames["valid"], *split_frames["test"]],
        ignore_index=True,
    )
    feature_cols = config.all_screening_supplementary_cols()
    validate_no_missing_features(long_panel_df, feature_cols)
    long_panel_df[label_col] = long_panel_df[label_col].astype(int)
    return long_panel_df


# 长面板阶段同样要求显式给出 --assets。
def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", required=True)
    return parser.parse_args()


# 统一筛选固定使用 1d 标签，这里只负责把各资产的 screening_ready_panel 合成一张 pooled 长面板。
def main() -> None:
    """生成固定 1d 筛选期限的统一长面板。"""
    args = parse_args()
    screening_horizon = PipelineConfig().selected_feature_source_horizon()
    ctx = get_horizon_context(horizon=screening_horizon)

    asset_aliases = resolve_asset_aliases(args.assets)
    long_panel_df = build_long_panel(ctx, asset_aliases)

    ctx.paths.ensure_screening_data_dir()
    long_panel_df.to_csv(
        ctx.paths.screening_long_panel_file(),
        index=False,
        encoding="utf-8-sig",
    )

    print("统一筛选长面板已保存：")
    print(ctx.paths.screening_long_panel_file())
    print(f"筛选期限：{screening_horizon}d")
    print(f"拼接资产：{', '.join(asset_aliases)}")
    print(f"长面板形状：{long_panel_df.shape}")


if __name__ == "__main__":
    main()

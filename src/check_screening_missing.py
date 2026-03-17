import argparse

import matplotlib.pyplot as plt
import pandas as pd

from config import PipelineConfig, get_asset_context, get_horizon_context
from plot_utils import (
    MONO_BAR_COLOR,
    MONO_EDGE_COLOR,
    configure_monochrome_matplotlib,
)
from screening_feature_meta import build_feature_display_frame


# 统一筛选前的诊断脚本要求显式传入资产列表，入口先拦住空列表和重复资产。
def resolve_asset_aliases(asset_arg: str) -> list[str]:
    """解析需要统计的资产简称列表。"""
    asset_aliases = [item.strip() for item in asset_arg.split(",") if item.strip()]
    if not asset_aliases:
        raise ValueError("至少需要一个资产简称。")
    if len(set(asset_aliases)) != len(asset_aliases):
        raise ValueError(f"--assets 中存在重复资产简称：{asset_aliases}")
    return asset_aliases


# 缺失报告只跟统一筛选候选变量有关，不统计固定保留的 3 个交易流指标。
def screening_feature_groups(config: PipelineConfig) -> list[tuple[str, str]]:
    """返回统一筛选候选变量及其分组。"""
    return [(col, "supplementary") for col in config.all_screening_supplementary_cols()]


# 缺失诊断回到 feature_panel 做，因为这里只有填充前的原始缺失形态和 split 信息。
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


# 交易流相关性诊断回到 screening_ready_panel 做，因为这里已经完成资产内填充和标准化，
# 与统一筛选长面板的训练集口径一致，只是额外保留了 3 个固定交易流指标。
def load_screening_ready_panel(asset_alias: str) -> pd.DataFrame:
    """读取单个资产的筛选输入表。"""
    ctx = get_asset_context(asset_alias=asset_alias)
    keep_cols = [
        "split",
        *ctx.config.fixed_core_feature_cols,
        *ctx.config.all_screening_supplementary_cols(),
    ]
    df = pd.read_csv(ctx.paths.screening_ready_panel_file(), parse_dates=["Date"])
    return df.loc[:, keep_cols].copy()


# 读取统一筛选重要性表，并校验后续 merge 缺失统计所需的关键列。
def load_importance_report_for_enrichment(ctx) -> pd.DataFrame:
    """读取需要补充缺失统计的变量重要性表。"""
    file_path = ctx.paths.screening_output_file("screening_lgbm_importance.csv")
    df = pd.read_csv(file_path)
    required_cols = {"feature_name", "importance"}
    missing_cols = sorted(required_cols.difference(df.columns))
    if missing_cols:
        raise ValueError(f"{file_path} 缺少这些关键列：{missing_cols}")
    if df["feature_name"].duplicated().any():
        dup_cols = df.loc[df["feature_name"].duplicated(), "feature_name"].tolist()
        raise ValueError(f"{file_path} 中存在重复变量：{dup_cols}")
    return df


# 同时统计 overall/train/valid/test 三个切面，方便区分“整体缺失高”和“只在某一段缺失高”。
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


# pooled 报告把所有资产纵向合并后再统计，by_asset 报告则逐资产展开，
# 两张表一起用于判断某列是整体不可用还是只在个别资产上有问题。
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


# 先把每个资产训练集拼成 pooled 训练面板，再一次性计算 3 个交易流指标和全部候选变量的相关性。
# 缺失在这里不再视为报错；相关系数直接使用 pandas 的 pairwise complete 规则计算。
def build_trade_flow_corr_report(asset_aliases: list[str]) -> pd.DataFrame:
    """生成固定交易流指标与其他候选变量的相关性报告。"""
    config = PipelineConfig()
    fixed_cols = list(config.fixed_core_feature_cols)
    supplementary_cols = config.all_screening_supplementary_cols()

    pooled_train_frames: list[pd.DataFrame] = []
    for asset_alias in asset_aliases:
        asset_df = load_screening_ready_panel(asset_alias)
        train_df = asset_df.loc[asset_df["split"] == "train", [*fixed_cols, *supplementary_cols]].copy()
        if train_df.empty:
            raise ValueError(f"{asset_alias} 的 screening_ready_panel 缺少 train 样本。")
        pooled_train_frames.append(train_df)

    pooled_train_df = pd.concat(pooled_train_frames, ignore_index=True)
    corr_full_df = pooled_train_df.loc[:, [*fixed_cols, *supplementary_cols]].corr(method="pearson")
    corr_matrix_df = corr_full_df.loc[fixed_cols, supplementary_cols].fillna(0.0)

    corr_report_df = corr_matrix_df.T.reset_index().rename(columns={"index": "feature_name"})
    rename_map = {col: f"corr_with_{col}" for col in fixed_cols}
    corr_report_df = corr_report_df.rename(columns=rename_map)
    corr_cols = list(rename_map.values())
    corr_report_df["max_abs_corr_with_trade_flow"] = corr_report_df.loc[:, corr_cols].abs().max(axis=1)

    display_df = build_feature_display_frame(corr_report_df["feature_name"].tolist())
    corr_report_df = corr_report_df.merge(display_df, on="feature_name", how="left")
    corr_report_df = corr_report_df.loc[
        :,
        [
            "feature_name",
            "feature_chinese_name",
            "feature_chinese_meaning",
            *corr_cols,
            "max_abs_corr_with_trade_flow",
        ],
    ].sort_values(
        ["max_abs_corr_with_trade_flow", "feature_name"],
        ascending=[False, True],
    ).reset_index(drop=True)
    return corr_report_df


# 热力图底色使用绝对相关系数，数字保留原始相关系数符号，这样既能突出高相关，也不丢失方向信息。
def plot_trade_flow_corr_heatmap(corr_report_df: pd.DataFrame, output_file) -> None:
    """把交易流指标与候选变量的相关性画成黑白灰热力图。"""
    configure_monochrome_matplotlib()

    corr_cols = [col for col in corr_report_df.columns if col.startswith("corr_with_")]
    fixed_feature_names = [col.removeprefix("corr_with_") for col in corr_cols]
    row_meta = build_feature_display_frame(fixed_feature_names)
    row_labels = row_meta["feature_chinese_name"].tolist()
    col_labels = corr_report_df["feature_chinese_name"].tolist()
    heatmap_values = corr_report_df.loc[:, corr_cols].T.abs().to_numpy()
    annotation_values = corr_report_df.loc[:, corr_cols].T.to_numpy()

    fig_width = max(14, len(col_labels) * 0.55)
    fig_height = 5.2
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(heatmap_values, cmap="Greys", aspect="auto", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=75, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("统一筛选候选变量")
    ax.set_ylabel("固定保留交易流指标")
    ax.set_title("交易流指标与候选变量相关性热力图\n底色为绝对相关系数，单元格数字为原始相关系数")

    font_size = 7 if len(col_labels) > 24 else 8
    for row_idx in range(annotation_values.shape[0]):
        for col_idx in range(annotation_values.shape[1]):
            corr_value = annotation_values[row_idx, col_idx]
            text_color = "white" if abs(corr_value) >= 0.55 else "black"
            ax.text(
                col_idx,
                row_idx,
                f"{corr_value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=font_size,
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.02, pad=0.02)
    colorbar.set_label("绝对相关系数")
    colorbar.outline.set_edgecolor(MONO_EDGE_COLOR)
    fig.tight_layout()
    fig.savefig(output_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


# 把 pooled 缺失统计补入变量重要性表，形成“重要性 + 缺失率”的一张落地表。
def enrich_importance_report_with_missing(
    importance_df: pd.DataFrame,
    pooled_report_df: pd.DataFrame,
) -> pd.DataFrame:
    """把 pooled 缺失统计合并到变量重要性表。"""
    display_df = build_feature_display_frame(importance_df["feature_name"].tolist())
    base_df = importance_df.drop(
        columns=[
            "importance_rank",
            "feature_chinese_name",
            "feature_chinese_meaning",
            "overall_missing_ratio",
        ],
        errors="ignore",
    )
    base_df = base_df.merge(display_df, on="feature_name", how="left")
    enriched_df = base_df.merge(
        pooled_report_df.loc[:, ["feature_name", "overall_missing_ratio"]],
        on="feature_name",
        how="left",
    )

    unresolved_cols = enriched_df.loc[
        enriched_df["overall_missing_ratio"].isna(),
        "feature_name",
    ].tolist()
    if unresolved_cols:
        raise ValueError(f"缺失比例报告缺少这些重要性变量：{unresolved_cols}")

    return enriched_df.loc[
        :,
        [
            "feature_name",
            "feature_chinese_name",
            "feature_chinese_meaning",
            "importance",
            "overall_missing_ratio",
        ],
    ].sort_values(["importance", "feature_name"], ascending=[False, True]).reset_index(drop=True)


# 同时输出按最大绝对相关系数排序的条形图，便于快速挑出与交易流指标共线性偏高的候选变量。
def plot_trade_flow_corr_rank(corr_report_df: pd.DataFrame, output_file) -> None:
    """绘制候选变量与固定交易流指标的最大绝对相关系数排序图。"""
    configure_monochrome_matplotlib()
    plot_df = corr_report_df.sort_values("max_abs_corr_with_trade_flow", ascending=True)

    fig_height = max(8, len(plot_df) * 0.28)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.barh(
        plot_df["feature_chinese_name"],
        plot_df["max_abs_corr_with_trade_flow"],
        color=MONO_BAR_COLOR,
        edgecolor=MONO_EDGE_COLOR,
    )
    ax.set_xlabel("与三个交易流指标的最大绝对相关系数")
    ax.set_ylabel("统一筛选候选变量")
    ax.set_title("候选变量与交易流指标相关性排序图")
    ax.grid(axis="x", linestyle="--", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


# 缺失报告阶段固定要求显式传入资产列表和 horizon。
def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    return parser.parse_args()


# 当前主线约定缺失报告固定使用 1d 筛选期限。
# 这里同时输出缺失报告、重要性表补充信息和交易流相关性参考图，不做任何删列动作。
def main() -> None:
    """输出统一筛选前的缺失比例报告和交易流相关性报告。"""
    args = parse_args()
    ctx = get_horizon_context(horizon=args.horizon)
    screening_horizon = ctx.config.selected_feature_source_horizon()
    if args.horizon != screening_horizon:
        raise ValueError(
            f"缺失比例检查固定使用 {screening_horizon}d，请显式传入 --horizon {screening_horizon}"
        )

    asset_aliases = resolve_asset_aliases(args.assets)
    pooled_report_df, by_asset_report_df = build_missing_reports(asset_aliases)
    corr_report_df = build_trade_flow_corr_report(asset_aliases)

    ctx.paths.ensure_screening_output_dir()
    pooled_file = ctx.paths.screening_output_file("screening_missing_before_imputation.csv")
    by_asset_file = ctx.paths.screening_output_file("screening_missing_before_imputation_by_asset.csv")
    importance_file = ctx.paths.screening_output_file("screening_lgbm_importance.csv")
    corr_file = ctx.paths.screening_output_file("screening_trade_flow_corr.csv")
    corr_heatmap_file = ctx.paths.screening_output_file("screening_trade_flow_corr_heatmap.png")
    corr_rank_file = ctx.paths.screening_output_file("screening_trade_flow_corr_rank.png")
    legacy_file = ctx.paths.screening_output_file("screening_missing_before_imputation_by_future.csv")

    pooled_report_df.to_csv(pooled_file, index=False, encoding="utf-8-sig")
    by_asset_report_df.to_csv(by_asset_file, index=False, encoding="utf-8-sig")

    importance_df = load_importance_report_for_enrichment(ctx)
    enriched_importance_df = enrich_importance_report_with_missing(
        importance_df=importance_df,
        pooled_report_df=pooled_report_df,
    )
    enriched_importance_df.to_csv(importance_file, index=False, encoding="utf-8-sig")

    corr_report_df.to_csv(corr_file, index=False, encoding="utf-8-sig")
    plot_trade_flow_corr_heatmap(corr_report_df, corr_heatmap_file)
    plot_trade_flow_corr_rank(corr_report_df, corr_rank_file)

    if legacy_file.exists():
        legacy_file.unlink()

    print("筛选层分析结果已保存：")
    print(pooled_file)
    print(by_asset_file)
    print(importance_file)
    print(corr_file)
    print(corr_heatmap_file)
    print(corr_rank_file)


if __name__ == "__main__":
    main()

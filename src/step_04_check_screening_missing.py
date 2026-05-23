from __future__ import annotations

import os
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 第 04 步：生成筛选诊断
# 当前主线位置：统一筛选长表之前的质量诊断与辅助诊断图。
# 本文件负责：输出候选变量缺失率、交易流相关性表，以及 `screening_trade_flow_corr_heatmap.png`、`screening_trade_flow_corr_rank.png` 两张诊断图。
# 主函数：`run(assets, horizon)`。
# 命令行入口：`main()`，负责把命令行参数解析为 `run(...)` 调用。
# 直接输入：`data/processed/<asset>/feature_panel.csv`；`data/processed/<asset>/screening_ready_panel.csv`；`src/config.py` 中的 `FIXED_TRADEFLOW_COLUMNS`、`SCREENING_CANDIDATE_COLUMNS`。
# 直接输出：`outputs/screening/diagnostics/screening_missing_before_imputation.csv`；`screening_missing_before_imputation_by_asset.csv`；`screening_trade_flow_corr.csv`；`screening_trade_flow_corr_heatmap.png`；`screening_trade_flow_corr_rank.png`。
# 下游读取：第 07 步筛选参考总表；第 11 步章节结果。
# 关键修改位置：第 04 步脚本中的缺失率统计、相关性诊断逻辑和输出命名；`src/common/visual_style.py`。
# 变更后重跑起点：第 04 步。正式变量清单和章节结果都会读取这里的产物。
# 对应文档：`README.md` 的“主线结构总览”“变量筛选逻辑”和“结果目录与阅读顺序”。
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.visual_style import annotate_bars_horizontal, configure_matplotlib, gray_colors, save_figure, style_axis
from src.config import (
    FIXED_TRADEFLOW_COLUMNS,
    SCREENING_ASSETS,
    SCREENING_CANDIDATE_COLUMNS,
    SCREENING_HORIZON,
    SCREENING_OUTPUT_DIR,
    asset_processed_dir,
    parse_assets,
)
from src.common.paths import ensure_dir


def _safe_pearson_corr(left: pd.Series, right: pd.Series) -> float:
    """在任一变量近似常数时直接返回 0，避免相关系数计算产生运行警告。"""
    left_std = float(left.std())
    right_std = float(right.std())
    if np.isclose(left_std, 0.0) or np.isclose(right_std, 0.0):
        return 0.0
    corr_value = left.corr(right)
    if pd.isna(corr_value):
        return 0.0
    return float(corr_value)


def run(assets: list[str], horizon: int) -> None:
    """输出筛选阶段缺失统计、交易流相关性表和两张诊断图。"""
    # 统一中文字体与简洁坐标轴风格（诊断图也保持一致，避免多余网格线/边框线）。
    configure_matplotlib()
    # 诊断结果统一放在 screening/diagnostics，后续第 7 步和第 11 步都会读取这里。
    diagnostics_dir = ensure_dir(SCREENING_OUTPUT_DIR / "diagnostics")
    by_asset_records = []
    pooled_ready_parts = []

    for asset in assets:
        # feature_panel 保留填补前的变量，适合统计原始缺失情况。
        feature_panel = pd.read_csv(asset_processed_dir(asset) / "feature_panel.csv", parse_dates=["Date"])
        # screening_ready_panel 已按训练集统计量完成填补和标准化，适合做相关性诊断。
        # 这里故意不用 feature_panel 来算相关性，是因为相关性比较希望建立在已经完成统一预处理的面板上。
        ready_panel = pd.read_csv(asset_processed_dir(asset) / "screening_ready_panel.csv", parse_dates=["Date"])
        pooled_ready_parts.append(ready_panel)
        for feature_name in SCREENING_CANDIDATE_COLUMNS:
            # 每个候选控制变量按资产分别统计缺失数量和缺失率。
            missing_count = int(feature_panel[feature_name].isna().sum())
            by_asset_records.append(
                {
                    "asset_alias": asset,
                    "feature_name": feature_name,
                    "missing_count": missing_count,
                    "missing_rate": missing_count / len(feature_panel),
                }
            )

    by_asset_df = pd.DataFrame(by_asset_records)
    # 把单资产缺失统计合并为总体缺失率，用于判断变量质量。
    overall_df = by_asset_df.groupby("feature_name", as_index=False).agg(missing_count=("missing_count", "sum"))
    total_rows = sum(len(pd.read_csv(asset_processed_dir(asset) / "feature_panel.csv")) for asset in assets)
    overall_df["missing_rate_overall"] = overall_df["missing_count"] / total_rows
    overall_df = overall_df.sort_values(["missing_rate_overall", "feature_name"], ascending=[False, True]).reset_index(drop=True)
    # 这两张缺失表分别回答“总体缺失情况”和“分资产缺失情况”，第 07 步会读取总体表。
    overall_df.to_csv(diagnostics_dir / "screening_missing_before_imputation.csv", index=False)
    by_asset_df.to_csv(diagnostics_dir / "screening_missing_before_imputation_by_asset.csv", index=False)

    pooled_ready = pd.concat(pooled_ready_parts, ignore_index=True)
    # 相关性只用训练集，避免验证集和测试集信息进入筛选参考。
    pooled_train = pooled_ready.loc[pooled_ready["split"] == "train"].copy()
    corr_records = []
    corr_matrix = np.zeros((len(FIXED_TRADEFLOW_COLUMNS), len(SCREENING_CANDIDATE_COLUMNS)))
    for i, fixed_feature in enumerate(FIXED_TRADEFLOW_COLUMNS):
        for j, candidate in enumerate(SCREENING_CANDIDATE_COLUMNS):
            corr_value = _safe_pearson_corr(pooled_train[fixed_feature], pooled_train[candidate])
            corr_matrix[i, j] = corr_value
            corr_records.append(
                {
                    "tradeflow_feature": fixed_feature,
                    "candidate_feature": candidate,
                    "pearson_corr": corr_value,
                    "abs_pearson_corr": abs(corr_value),
                }
            )
    corr_df = pd.DataFrame(corr_records)
    # screening_trade_flow_corr.csv 会在第 07 步被合并进筛选参考总表，也会在第 11 步变成章节结果图表。
    corr_df.to_csv(diagnostics_dir / "screening_trade_flow_corr.csv", index=False)

    fig, ax = plt.subplots(figsize=(18, 5))
    heatmap = ax.imshow(corr_matrix, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(SCREENING_CANDIDATE_COLUMNS)), SCREENING_CANDIDATE_COLUMNS, rotation=90, ha="right", fontsize=9)
    ax.set_yticks(range(len(FIXED_TRADEFLOW_COLUMNS)), FIXED_TRADEFLOW_COLUMNS, fontsize=9)
    style_axis(ax, title="交易流变量与候选变量相关性热力图", enable_y_grid=False)
    cbar = fig.colorbar(heatmap, ax=ax, fraction=0.030, pad=0.02)
    cbar.ax.set_ylabel("皮尔逊相关系数", rotation=90, va="bottom")
    # 热力图用于直观看到“交易流变量和候选变量之间是否存在大面积高相关”。
    save_figure(fig, diagnostics_dir / "screening_trade_flow_corr_heatmap.png")

    rank_df = (
        corr_df.groupby("candidate_feature", as_index=False)
        .agg(max_abs_pearson_corr=("abs_pearson_corr", "max"))
        .sort_values(["max_abs_pearson_corr", "candidate_feature"], ascending=[False, True])
    )
    bar_count = int(rank_df.shape[0])
    fig_height = max(6.4, 0.30 * bar_count + 1.6)
    fig, ax = plt.subplots(figsize=(10.2, fig_height))
    bars = ax.barh(
        rank_df["candidate_feature"].tolist(),
        rank_df["max_abs_pearson_corr"].tolist(),
        color=gray_colors(bar_count),
        edgecolor="black",
        linewidth=0.8,
    )
    ax.invert_yaxis()
    style_axis(ax, title="候选变量相关性强度排序图", xlabel="最大绝对皮尔逊相关系数")
    annotate_bars_horizontal(ax, bars, fmt="{:.2f}")
    # 排序图把热力图中的矩阵信息压缩成“按变量排序”的结果，便于快速浏览。
    save_figure(fig, diagnostics_dir / "screening_trade_flow_corr_rank.png")


def main() -> None:
    """命令行入口；业务逻辑在 run(assets, horizon) 中。"""
    parser = argparse.ArgumentParser(description="第 04 步：生成筛选阶段的缺失率统计、交易流相关性表和诊断图。")
    parser.add_argument("--assets", default=None, help="逗号分隔的资产代码列表。通常与第 03 步保持一致；若不传，则按当前主线筛选资产范围执行。")
    parser.add_argument("--horizon", type=int, default=None, help="当前诊断对应的筛选记录口径。若不传，则按当前主线筛选期限执行。")
    args = parser.parse_args()
    run(parse_assets(args.assets) if args.assets else SCREENING_ASSETS.copy(), args.horizon if args.horizon is not None else SCREENING_HORIZON)


if __name__ == "__main__":
    main()


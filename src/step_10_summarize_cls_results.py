from __future__ import annotations

import os
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 第 10 步：汇总分类训练结果
# 当前主线位置：单 run 训练结果目录 -> 跨 run 汇总结果。
# 本文件负责：读取第 09 步每个 run 的 `metrics.csv`，生成方案内汇总表、最佳结果表、异常表和交易流增益对比表。
# 主函数：`run(assets, horizons, model_ids, scheme, experiment_tag, ...)`。
# 命令行入口：`main()`，负责把命令行参数解析为 `run(...)` 调用。
# 直接输入：`outputs/classification/<model>/<run_id>/metrics.csv`；`src/config.py` 中的汇总命名规则。
# 直接输出：`outputs/summary/classification_metrics__*.csv`；`classification_best_valid_auc__*.csv`；`classification_anomalies__*.csv`；`classification_ablation__*.csv`；`classification_ablation_overall__*.csv`；`classification_test_by_model__*.csv`。
# 下游读取：第 11 步 `src.step_11_build_visualizations`。
# 关键修改位置：汇总字段、特征数校验规则、跨方案对比口径和输出文件命名。
# 变更后重跑起点：第 10 步。如果第 07 步控制变量数量也变化，需要同步检查这里的特征数判断。
# 对应文档：`README.md` 的“主线结构总览”“文件流转与关键中间文件”和“结果目录与阅读顺序”。
import argparse
from pathlib import Path

import pandas as pd

from src.config import (
    CANONICAL_MAINLINE_CONTROL_COLUMNS,
    MAINLINE_CLASSIFICATION_MODELS,
    MAINLINE_SCHEMES,
    PAPER_ASSETS,
    PAPER_HORIZONS,
    classification_output_dir,
    classification_summary_dir,
    classification_summary_filename,
    experiment_tag_for_horizon,
    parse_assets,
    parse_horizons,
    parse_models,
    parse_scheme,
    tagged_run_name,
)
from src.common.paths import ensure_dir


TEST_METRIC_COLUMNS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "balanced_accuracy",
    "ks",
    "pred_unique_count",
    "pred_pos_rate",
]
SCHEME_DIFF_METRICS = ["accuracy", "precision", "recall", "f1", "auc", "balanced_accuracy"]


def expected_feature_count_for_scheme(scheme: str, expected_control_count: int | None = None) -> int:
    """按当前主线控制变量口径，计算指定方案应有的特征数。"""
    # 默认控制变量数量来自 config.py 的固定主线清单；测试可传入 expected_control_count 单独验证。
    if expected_control_count is None:
        expected_control_count = len(CANONICAL_MAINLINE_CONTROL_COLUMNS)
    # 加入交易流方案会比基准方案多 4 个固定交易流变量。
    return expected_control_count + 4 if scheme == "with_tradeflow4" else expected_control_count


def build_test_metric_comparison_table(merged: pd.DataFrame, model_id: str) -> pd.DataFrame:
    """把指定模型在两种方案下的测试指标并成一张对比表。"""
    # 只取当前模型，避免不同模型的指标混在同一张重点对比表里。
    model_df = merged.loc[merged["model_id"] == model_id].copy()
    if model_df.empty:
        return pd.DataFrame()
    out = model_df[
        [
            "asset_alias",
            "horizon",
            "scheme_name_with_tradeflow4",
            "scheme_name_no_tradeflow4",
            "feature_count_with_tradeflow4",
            "feature_count_no_tradeflow4",
        ]
        + [f"test_{metric}_with_tradeflow4" for metric in TEST_METRIC_COLUMNS]
        + [f"test_{metric}_no_tradeflow4" for metric in TEST_METRIC_COLUMNS]
    ].copy()
    for metric in TEST_METRIC_COLUMNS:
        # diff = 加入交易流 - 不加入交易流，用于判断交易流变量是否带来增益。
        out[f"test_{metric}_diff"] = out[f"test_{metric}_with_tradeflow4"] - out[f"test_{metric}_no_tradeflow4"]
    return out.sort_values(["test_auc_diff", "test_f1_diff", "asset_alias", "horizon"], ascending=[False, False, True, True]).reset_index(drop=True)


def export_cross_scheme_summaries(
    model_ids: list[str],
    experiment_tag: str | None,
    summary_dir: Path,
) -> None:
    """在 with/no 两种方案的汇总都存在时，补写跨方案对比结果。"""
    # 只有两种方案的 metrics 表都已经生成时，才输出消融对比表。
    with_metrics_path = summary_dir / classification_summary_filename("metrics", experiment_tag, "with_tradeflow4")
    no_metrics_path = summary_dir / classification_summary_filename("metrics", experiment_tag, "no_tradeflow4")
    if not with_metrics_path.exists() or not no_metrics_path.exists():
        return

    with_df = pd.read_csv(with_metrics_path)
    no_df = pd.read_csv(no_metrics_path)
    # 按模型、资产和期限一一配对，保证比较对象完全一致。
    merged = with_df.merge(
        no_df,
        on=["model_id", "asset_alias", "horizon"],
        suffixes=("_with_tradeflow4", "_no_tradeflow4"),
    )
    if merged.empty:
        return

    # 核心差值字段均定义为“加入交易流 - 不加入交易流”。
    for metric in SCHEME_DIFF_METRICS:
        merged[f"valid_{metric}_diff"] = merged[f"valid_{metric}_with_tradeflow4"] - merged[f"valid_{metric}_no_tradeflow4"]
        merged[f"test_{metric}_diff"] = merged[f"test_{metric}_with_tradeflow4"] - merged[f"test_{metric}_no_tradeflow4"]
    merged["experiment_tag"] = experiment_tag
    # ablation 表是第 11 步“交易流增益图”和“模型与基线评估指标对比表”的重要来源。
    merged = merged.sort_values(["test_auc_diff", "test_f1_diff", "asset_alias", "model_id"], ascending=[False, False, True, True])
    merged.to_csv(summary_dir / classification_summary_filename("ablation", experiment_tag), index=False)

    for model_id in model_ids:
        table = build_test_metric_comparison_table(merged, model_id)
        if table.empty:
            continue
        table.to_csv(
            summary_dir / classification_summary_filename("test_by_model", experiment_tag, model_id=model_id),
            index=False,
        )

    # ablation_overall 表把比较口径进一步压缩到“模型 × 期限”的平均增益层面。
    (
        merged.groupby(["model_id", "horizon"], as_index=False)
        .agg(
            avg_valid_auc_with_tradeflow4=("valid_auc_with_tradeflow4", "mean"),
            avg_valid_auc_no_tradeflow4=("valid_auc_no_tradeflow4", "mean"),
            avg_valid_auc_diff=("valid_auc_diff", "mean"),
            avg_test_auc_diff=("test_auc_diff", "mean"),
            avg_test_f1_diff=("test_f1_diff", "mean"),
            avg_test_accuracy_diff=("test_accuracy_diff", "mean"),
            avg_test_precision_diff=("test_precision_diff", "mean"),
            avg_test_recall_diff=("test_recall_diff", "mean"),
            avg_test_balanced_accuracy_diff=("test_balanced_accuracy_diff", "mean"),
        )
        .assign(experiment_tag=experiment_tag)
        .sort_values(["avg_test_auc_diff", "avg_test_f1_diff"], ascending=[False, False])
        .to_csv(summary_dir / classification_summary_filename("ablation_overall", experiment_tag), index=False)
    )


def run(
    assets: list[str],
    horizons: list[int],
    model_ids: list[str],
    scheme: str,
    feature_set_tag: str | None = None,
    experiment_tag: str | None = None,
    expected_control_count: int | None = None,
    classification_output_root: Path | None = None,
    summary_root: Path | None = None,
) -> None:
    """把单次分类训练结果汇总成指标表、最优表和异常标记表。"""
    del feature_set_tag
    summary_dir = ensure_dir(classification_summary_dir(summary_root))
    metrics_rows = []
    anomaly_rows = []
    for asset in assets:
        for horizon in horizons:
            # 先算出理论特征数，后面用来检查训练输出是否读错变量。
            # 这里的“理论特征数”来自当前方案应该包含多少个控制变量和交易流变量。
            expected_feature_count = expected_feature_count_for_scheme(scheme, expected_control_count)
            run_id = tagged_run_name(asset, horizon, scheme, experiment_tag)
            for model_id in model_ids:
                # 第 9 步每个 run 都会输出 metrics.csv；缺文件时不报错中断，而是写入异常清单。
                # 这样即使某个模型训练失败，也能从 anomalies 表里看到是哪一个组合缺文件。
                metrics_path = classification_output_dir(model_id, run_id, classification_output_root) / "metrics.csv"
                if not metrics_path.exists():
                    anomaly_rows.append(
                        {
                            "model_id": model_id,
                            "asset_alias": asset,
                            "horizon": horizon,
                            "scheme": scheme,
                            "scheme_name": run_id,
                            "experiment_tag": experiment_tag,
                            "missing_output": True,
                            "valid_auc_gt_train_auc": pd.NA,
                            "degenerate_valid_prediction": pd.NA,
                            "degenerate_test_prediction": pd.NA,
                            "feature_count_mismatch": pd.NA,
                        }
                    )
                    continue
                # metrics.csv 只有一行，包含 train/valid/test 的主要分类指标。
                row = pd.read_csv(metrics_path).iloc[0].to_dict()
                metrics_rows.append(row)
                anomaly_rows.append(
                    {
                        "model_id": model_id,
                        "asset_alias": asset,
                        "horizon": horizon,
                        "scheme": scheme,
                        "scheme_name": run_id,
                        "experiment_tag": experiment_tag,
                        "missing_output": False,
                        "valid_auc_gt_train_auc": row["valid_auc"] > row["train_auc"],
                        "degenerate_valid_prediction": row["valid_pred_unique_count"] <= 1,
                        "degenerate_test_prediction": row["test_pred_unique_count"] <= 1,
                        "feature_count_mismatch": row["feature_count"] != expected_feature_count,
                    }
                )

    # metrics 汇总所有正常 run；anomalies 记录缺失输出、预测退化和特征数不一致。
    metrics_df = pd.DataFrame(metrics_rows)
    anomalies_df = pd.DataFrame(anomaly_rows)
    # 每个资产和期限选验证集 AUC 最高的模型，作为“最佳模型”汇总。
    best_df = (
        metrics_df.sort_values(["asset_alias", "horizon", "valid_auc", "model_id"], ascending=[True, True, False, True])
        .groupby(["asset_alias", "horizon"], as_index=False)
        .head(1)
        .reset_index(drop=True)
        if not metrics_df.empty
        else pd.DataFrame()
    )
    # 这三张表分别服务不同用途：
    # 1）metrics：完整汇总所有 run 的训练/验证/测试指标；
    # 2）best_valid_auc：按资产和期限挑出验证集 AUC 最优模型；
    # 3）anomalies：记录缺输出、预测退化和特征数不一致等异常。
    metrics_df.to_csv(summary_dir / classification_summary_filename("metrics", experiment_tag, scheme), index=False)
    best_df.to_csv(summary_dir / classification_summary_filename("best_valid_auc", experiment_tag, scheme), index=False)
    anomalies_df.to_csv(summary_dir / classification_summary_filename("anomalies", experiment_tag, scheme), index=False)
    export_cross_scheme_summaries(model_ids, experiment_tag, summary_dir)


def main() -> None:
    """命令行入口；业务逻辑在 run(...) 中。"""
    parser = argparse.ArgumentParser(description="第 10 步：汇总第 09 步分类训练结果，并生成方案内汇总表与跨方案增益对比表。")
    parser.add_argument("--assets", default=None, help="逗号分隔的资产代码列表。若不传，则按当前主线正式预测资产范围全跑。")
    parser.add_argument("--horizons", default=None, help="逗号分隔的预测期限列表。若不传，则按当前主线正式预测期限范围全跑。")
    parser.add_argument("--models", default=None, help="逗号分隔的模型代码列表。若不传，则按当前主线分类模型范围全跑。")
    parser.add_argument("--scheme", default=None, help="当前先汇总哪一种方案。若不传，则按当前主线方案范围全跑。")
    parser.add_argument("--feature-set-tag", default=None, help="历史兼容字段，当前主线通常不用改。")
    parser.add_argument("--experiment-tag", default=None, help="实验标签。显式运行时可手动指定；不传则按当前主线期限标签映射自动选择。")
    parser.add_argument("--expected-control-count", type=int, default=None, help="预期控制变量数量。只有当第 07 步控制变量数发生变化时，才需要显式传入。")
    parser.add_argument("--classification-output-root", default=None, help="分类结果输入根目录。一般保持默认。")
    parser.add_argument("--summary-root", default=None, help="汇总结果输出根目录。一般保持默认。")
    args = parser.parse_args()
    classification_output_root = Path(args.classification_output_root) if args.classification_output_root else None
    summary_root = Path(args.summary_root) if args.summary_root else None
    explicit_requested = any(value is not None for value in [args.assets, args.horizons, args.models, args.scheme, args.experiment_tag])
    if not explicit_requested:
        for horizon in PAPER_HORIZONS:
            experiment_tag = experiment_tag_for_horizon(horizon)
            for scheme in MAINLINE_SCHEMES:
                run(PAPER_ASSETS.copy(), [horizon], MAINLINE_CLASSIFICATION_MODELS.copy(), scheme, args.feature_set_tag, experiment_tag, args.expected_control_count, classification_output_root, summary_root)
        return
    missing = [flag for flag, value in {"--assets": args.assets, "--horizons": args.horizons, "--models": args.models, "--scheme": args.scheme}.items() if value is None]
    if missing:
        parser.error("显式运行模式需要同时提供 --assets、--horizons、--models 和 --scheme。无参点击运行则会按主线配置全量执行。")
    assets = parse_assets(args.assets)
    horizons = parse_horizons(args.horizons)
    model_ids = parse_models(args.models)
    scheme = parse_scheme(args.scheme)
    if args.experiment_tag is not None:
        run(assets, horizons, model_ids, scheme, args.feature_set_tag, args.experiment_tag, args.expected_control_count, classification_output_root, summary_root)
        return
    for horizon in horizons:
        run(assets, [horizon], model_ids, scheme, args.feature_set_tag, experiment_tag_for_horizon(horizon), args.expected_control_count, classification_output_root, summary_root)


if __name__ == "__main__":
    main()

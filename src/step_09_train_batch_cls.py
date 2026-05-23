from __future__ import annotations

import os
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 第 09 步：批量训练分类模型
# 当前主线位置：建模输入目录 -> 单 run 训练结果目录。
# 本文件负责：按资产、期限、模型和方案批量训练分类器，并写出预测、指标、参数、重要性和模型文件。
# 主函数：`run(assets, horizons, model_ids, scheme, experiment_tag, ...)`。
# 命令行入口：`main()`，负责把命令行参数解析为 `run(...)` 调用。
# 直接输入：第 08 步生成的 `*_model_input.csv`；`src/common/search_specs.py`；`src/common/model_train_utils.py`。
# 直接输出：`outputs/classification/<model>/<run_id>/metrics.csv`；`best_params.csv`；`feature_importance.csv`；`coarse_search.csv`；`fine_search.csv`；`pred_train.csv`；`pred_valid.csv`；`pred_test.csv`；`models/classification/<model>/<run_id>/<model>.joblib`。
# 下游读取：第 10 步读取 `metrics.csv` 汇总；第 11 步读取单 run 结果目录。
# 关键修改位置：训练模型集合、并行组织方式、`src/common/search_specs.py` 和 `src/common/model_train_utils.py` 中的调参、指标与落盘逻辑。
# 变更后重跑起点：第 09 步。汇总结果、重点结果表和章节图表都依赖这里。
# 对应文档：`README.md` 的“主线结构总览”“文件流转与关键中间文件”和“结果目录与阅读顺序”。
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from src.config import MAINLINE_CLASSIFICATION_MODELS, MAINLINE_SCHEMES, PAPER_ASSETS, PAPER_HORIZONS, experiment_tag_for_horizon, parse_assets, parse_horizons, parse_models, parse_scheme
from src.common.model_train_utils import batch_parallel_jobs, run_classification_training


def _run_training_task(task: tuple[str, str, int, str, str | None, str | None, str | None, str, float, float, int, str]) -> None:
    """解包单个并行任务并调用统一分类训练入口。"""
    # 并行执行时只能传简单对象，所以这里把 tuple 解包后再转回 Path。
    (
        model_id,
        asset,
        horizon,
        scheme,
        experiment_tag,
        classification_output_root,
        classification_model_root,
        threshold_policy,
        min_pred_pos_rate,
        max_pred_pos_rate,
        min_class_count,
        search_primary_metric,
    ) = task
    run_classification_training(
        model_id,
        asset,
        horizon,
        scheme,
        experiment_tag,
        Path(classification_output_root) if classification_output_root else None,
        Path(classification_model_root) if classification_model_root else None,
        threshold_policy,
        min_pred_pos_rate,
        max_pred_pos_rate,
        min_class_count,
        search_primary_metric,
    )


def run(
    assets: list[str],
    horizons: list[int],
    model_ids: list[str],
    scheme: str,
    experiment_tag: str | None = None,
    classification_output_root: Path | None = None,
    classification_model_root: Path | None = None,
    threshold_policy: str = "mixed_f1_by_asset",
    min_pred_pos_rate: float = 0.05,
    max_pred_pos_rate: float = 0.95,
    min_class_count: int = 5,
    search_primary_metric: str = "valid_auc",
) -> None:
    """按资产、期限和模型批量训练分类任务。"""
    # 一个任务 = 一个模型 × 一个资产 × 一个预测期限 × 一种方案。
    # 这意味着 assets、horizons、models 三个参数每多写一个值，训练任务数都会成倍增加。
    tasks = [
        (
            model_id,
            asset,
            horizon,
            scheme,
            experiment_tag,
            str(classification_output_root) if classification_output_root else None,
            str(classification_model_root) if classification_model_root else None,
            threshold_policy,
            min_pred_pos_rate,
            max_pred_pos_rate,
            min_class_count,
            search_primary_metric,
        )
        for asset in assets
        for horizon in horizons
        for model_id in model_ids
    ]
    # 默认串行，只有设置 TRADEFLOW_BATCH_JOBS 时才会多进程并行。
    # 如果只是做单个任务复核，保持串行更容易看清楚报错来自哪一个模型和哪一个资产。
    jobs = batch_parallel_jobs()
    if jobs <= 1:
        # 串行模式更便于观察报错位置。
        for task in tasks:
            _run_training_task(task)
        return
    # 并行模式用于批量重训，输出目录仍然按 run 独立保存，不会互相覆盖。
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        list(executor.map(_run_training_task, tasks))


def main() -> None:
    """命令行入口；业务逻辑在 run(...) 中。"""
    parser = argparse.ArgumentParser(description="第 09 步：批量训练分类模型，并分别写出预测结果、指标表、重要性和模型文件。")
    parser.add_argument("--assets", default=None, help="逗号分隔的资产代码列表。若不传，则按当前主线正式预测资产范围全跑。")
    parser.add_argument("--horizons", default=None, help="逗号分隔的预测期限列表，例如 1 或 1,22。若不传，则按当前主线正式预测期限范围全跑。")
    parser.add_argument("--models", default=None, help="逗号分隔的模型代码列表，例如 dt_cls,rf_cls,svm_cls,xgb_cls,lgbm_cls。若不传，则按当前主线分类模型范围全跑。")
    parser.add_argument("--scheme", default=None, help="当前训练采用的特征方案。若不传，则按当前主线方案范围全跑。")
    parser.add_argument("--experiment-tag", default=None, help="实验标签。显式运行时可手动指定；不传则按当前主线期限标签映射自动选择。")
    parser.add_argument("--classification-output-root", default=None, help="分类结果输出根目录。一般保持默认，只有做临时试验时才改。")
    parser.add_argument("--classification-model-root", default=None, help="分类模型文件输出根目录。一般保持默认。")
    parser.add_argument(
        "--threshold-policy",
        default="mixed_f1_by_asset",
        choices=["mixed_f1_by_asset", "valid_f1_prev015", "valid_f1_cap075"],
        help="最终硬分类阈值策略：默认指数使用 valid_f1_prev015，非指数资产使用 valid_f1_cap075；阈值只在验证集上选择。",
    )
    parser.add_argument("--min-pred-pos-rate", type=float, default=0.05, help="验证集非退化约束：预测正类比例下界。")
    parser.add_argument("--max-pred-pos-rate", type=float, default=0.95, help="验证集非退化约束：预测正类比例上界。")
    parser.add_argument("--min-class-count", type=int, default=5, help="验证集非退化约束：每个预测类别的最小样本数。")
    parser.add_argument("--search-primary-metric", default="valid_auc", help="参数搜索主指标记录字段，默认 valid_auc。")
    args = parser.parse_args()
    classification_output_root = Path(args.classification_output_root) if args.classification_output_root else None
    classification_model_root = Path(args.classification_model_root) if args.classification_model_root else None
    explicit_requested = any(value is not None for value in [args.assets, args.horizons, args.models, args.scheme, args.experiment_tag])
    if not explicit_requested:
        for horizon in PAPER_HORIZONS:
            experiment_tag = experiment_tag_for_horizon(horizon)
            for scheme in MAINLINE_SCHEMES:
                run(
                    PAPER_ASSETS.copy(),
                    [horizon],
                    MAINLINE_CLASSIFICATION_MODELS.copy(),
                    scheme,
                    experiment_tag,
                    classification_output_root,
                    classification_model_root,
                    args.threshold_policy,
                    args.min_pred_pos_rate,
                    args.max_pred_pos_rate,
                    args.min_class_count,
                    args.search_primary_metric,
                )
        return
    missing = [flag for flag, value in {"--assets": args.assets, "--horizons": args.horizons, "--models": args.models, "--scheme": args.scheme}.items() if value is None]
    if missing:
        parser.error("显式运行模式需要同时提供 --assets、--horizons、--models 和 --scheme。无参点击运行则会按主线配置全量执行。")
    assets = parse_assets(args.assets)
    horizons = parse_horizons(args.horizons)
    model_ids = parse_models(args.models)
    scheme = parse_scheme(args.scheme)
    if args.experiment_tag is not None:
        run(
            assets,
            horizons,
            model_ids,
            scheme,
            args.experiment_tag,
            classification_output_root,
            classification_model_root,
            args.threshold_policy,
            args.min_pred_pos_rate,
            args.max_pred_pos_rate,
            args.min_class_count,
            args.search_primary_metric,
        )
        return
    for horizon in horizons:
        run(
            assets,
            [horizon],
            model_ids,
            scheme,
            experiment_tag_for_horizon(horizon),
            classification_output_root,
            classification_model_root,
            args.threshold_policy,
            args.min_pred_pos_rate,
            args.max_pred_pos_rate,
            args.min_class_count,
            args.search_primary_metric,
        )


if __name__ == "__main__":
    main()

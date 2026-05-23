from __future__ import annotations

import os
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 第 05 步：训练统一筛选模型
# 当前主线位置：统一筛选长表 -> 统一筛选重要性与筛选模型结果。
# 本文件负责：在 pooled screening 长面板上训练统一筛选模型，导出粗搜、细搜、重要性、指标、最优参数和模型文件。
# 主函数：`run(horizon)`。
# 命令行入口：`main()`，负责把命令行参数解析为 `run(...)` 调用。
# 直接输入：`data/processed/screening/screening_long_panel.csv`；`src/common/search_specs.py` 的筛选搜索空间；`src/config.py` 中的 `SCREENING_CANDIDATE_COLUMNS`。
# 直接输出：`outputs/screening/unified_lgbm/screening_lgbm_coarse_search.csv`；`screening_lgbm_fine_search.csv`；`screening_lgbm_importance.csv`；`screening_lgbm_metrics.csv`；`screening_lgbm_best_params.csv`；`models/screen_lgbm/screening_<horizon>d/screen_lgbm.joblib`。
# 下游读取：第 07 步正式变量清单；第 11 步章节结果。
# 关键修改位置：`src/common/search_specs.py`；本文件中的筛选指标、阈值、重要性导出和结果命名规则。
# 变更后重跑起点：第 05 步。通常还需要继续执行第 07 步和第 11 步，以同步正式变量清单和章节结果。
# 对应文档：`README.md` 的“主线结构总览”“变量筛选逻辑”和“结果目录与阅读顺序”。
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from src.config import MAINLINE_MODELS_DIR, SCREENING_CANDIDATE_COLUMNS, SCREENING_DIR, SCREENING_HORIZON, SCREENING_OUTPUT_DIR
from src.common.metrics_utils import extended_classification_metrics, select_validation_threshold
from src.common.paths import ensure_dir
from src.common.search_specs import SCREEN_LGBM_INITIAL_PARAMS, SCREEN_LGBM_PARAM_SPECS, build_fine_candidates_from_spec


def params_to_json(params: dict[str, object]) -> str:
    """把参数字典转成稳定的 JSON 字符串，便于写入搜索结果表。"""
    normalized = {}
    for key, value in params.items():
        if hasattr(value, "item"):
            value = value.item()
        normalized[key] = value
    return json.dumps(normalized, sort_keys=True, ensure_ascii=False)


def _fit_and_eval(params: dict[str, float], train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame):
    """在给定参数下训练 screening LGBM，并返回三段样本得分和指标。"""
    # 每次候选参数都重新实例化模型，避免前一次训练状态影响本次评估。
    model = LGBMClassifier(random_state=42, n_jobs=-1, objective="binary", verbosity=-1, **params)
    # 只使用候选控制变量做 pooled screening，不把 4 个固定交易流变量放入筛选候选池。
    model.fit(train_df[SCREENING_CANDIDATE_COLUMNS], train_df["target_label_1d"].astype(int))
    # 分别保存 train/valid/test 的上涨概率，用同一阈值计算指标。
    train_score = model.predict_proba(train_df[SCREENING_CANDIDATE_COLUMNS])[:, 1]
    valid_score = model.predict_proba(valid_df[SCREENING_CANDIDATE_COLUMNS])[:, 1]
    test_score = model.predict_proba(test_df[SCREENING_CANDIDATE_COLUMNS])[:, 1]
    threshold_choice = select_validation_threshold(valid_df["target_label_1d"].astype(int).to_numpy(), valid_score)
    threshold = float(threshold_choice["threshold"])
    train_metrics = extended_classification_metrics(train_df["target_label_1d"].astype(int).to_numpy(), train_score, threshold)
    valid_metrics = extended_classification_metrics(valid_df["target_label_1d"].astype(int).to_numpy(), valid_score, threshold)
    test_metrics = extended_classification_metrics(test_df["target_label_1d"].astype(int).to_numpy(), test_score, threshold)
    return model, train_score, valid_score, test_score, threshold, train_metrics, valid_metrics, test_metrics


def _score_unique_count(score) -> int:
    """返回预测概率唯一值数量，只用于剔除概率完全退化的候选。"""

    return int(pd.Series(score).round(15).nunique())


def run(horizon: int) -> None:
    """执行 pooled screening LightGBM 的粗搜、细搜、落盘和模型保存。"""
    # 第 3 步输出的统一长面板是本步骤唯一建模输入。
    df = pd.read_csv(SCREENING_DIR / "screening_long_panel.csv", parse_dates=["Date"])
    # 继续沿用第 2 步生成的时间切分，不能重新随机切分。
    train_df = df.loc[df["split"] == "train"].copy()
    valid_df = df.loc[df["split"] == "valid"].copy()
    test_df = df.loc[df["split"] == "test"].copy()

    current_params = dict(SCREEN_LGBM_INITIAL_PARAMS)
    coarse_rows = []
    search_id = 0
    # 第一阶段：逐个参数做粗搜索，每个参数选出当前验证表现较好的取值。
    # “粗搜”可以理解为先在较宽范围内找一个大致合理的参数区间。
    for param_name, spec in SCREEN_LGBM_PARAM_SPECS.items():
        step_rows = []
        for candidate in spec.coarse_candidates:
            params = dict(current_params)
            params[param_name] = candidate
            model, train_score, valid_score, test_score, threshold, train_metrics, valid_metrics, test_metrics = _fit_and_eval(params, train_df, valid_df, test_df)
            row = {
                "stage": "coarse",
                "param_name": param_name,
                "candidate_value": candidate,
                "search_id": search_id,
                "params_json": params_to_json(params),
                "train_auc": train_metrics["auc"],
                "valid_auc": valid_metrics["auc"],
                "valid_search_f1": valid_metrics["f1"],
                "valid_ks": valid_metrics["ks"],
                "train_score_unique_count": _score_unique_count(train_score),
                "valid_score_unique_count": _score_unique_count(valid_score),
                "test_score_unique_count": _score_unique_count(test_score),
                "score_non_degenerate": _score_unique_count(valid_score) > 1,
                "search_threshold_valid_balanced_accuracy": threshold,
            }
            search_id += 1
            coarse_rows.append(row)
            step_rows.append(row)
        # 只过滤验证集预测概率完全退化的候选，不再要求验证集 AUC 低于训练集 AUC。
        eligible = [row for row in step_rows if row["score_non_degenerate"]]
        chosen = pd.DataFrame(eligible or step_rows).sort_values(
            by=["valid_auc", "valid_search_f1", "valid_ks", "search_id"],
            ascending=[False, False, False, True],
        ).iloc[0]
        current_params[param_name] = chosen["candidate_value"]

    coarse_df = pd.DataFrame(coarse_rows)
    coarse_pool = coarse_df.loc[coarse_df["score_non_degenerate"]].copy()
    if coarse_pool.empty:
        coarse_pool = coarse_df.copy()
    coarse_best = coarse_pool.sort_values(
        by=["valid_auc", "valid_search_f1", "valid_ks", "search_id"],
        ascending=[False, False, False, True],
    ).iloc[0]
    coarse_best_params = json.loads(coarse_best["params_json"])
    coarse_best_valid_auc = float(coarse_best["valid_auc"])

    fine_rows = []
    # 第二阶段：围绕粗搜索最优值生成细搜索候选，只有验证表现不退步才采纳。
    # “细搜”可以理解为在粗搜找到的大致最优位置附近再做更精细的调整。
    for param_name, spec in SCREEN_LGBM_PARAM_SPECS.items():
        fine_candidates = build_fine_candidates_from_spec(spec, coarse_best_params[param_name])
        if not fine_candidates:
            continue
        for candidate in fine_candidates:
            params = dict(coarse_best_params)
            params[param_name] = candidate
            model, train_score, valid_score, test_score, threshold, train_metrics, valid_metrics, test_metrics = _fit_and_eval(params, train_df, valid_df, test_df)
            row = {
                "stage": "fine",
                "param_name": param_name,
                "candidate_value": candidate,
                "search_id": search_id,
                "params_json": params_to_json(params),
                "train_auc": train_metrics["auc"],
                "valid_auc": valid_metrics["auc"],
                "valid_search_f1": valid_metrics["f1"],
                "valid_ks": valid_metrics["ks"],
                "train_score_unique_count": _score_unique_count(train_score),
                "valid_score_unique_count": _score_unique_count(valid_score),
                "test_score_unique_count": _score_unique_count(test_score),
                "score_non_degenerate": _score_unique_count(valid_score) > 1,
                "qualified": valid_metrics["auc"] >= coarse_best_valid_auc and _score_unique_count(valid_score) > 1,
                "search_threshold_valid_balanced_accuracy": threshold,
            }
            search_id += 1
            fine_rows.append(row)

    fine_df = pd.DataFrame(fine_rows)
    if not fine_df.empty and fine_df["qualified"].any():
        final_params = json.loads(
            fine_df.loc[fine_df["qualified"]].sort_values(
                by=["valid_auc", "valid_search_f1", "valid_ks", "search_id"],
                ascending=[False, False, False, True],
            ).iloc[0]["params_json"]
        )
    else:
        final_params = coarse_best_params

    # 用最终参数重训一遍模型，并统一采用验证集 balanced accuracy 最大阈值。
    model, train_score, valid_score, test_score, decision_threshold, train_metrics, valid_metrics, test_metrics = _fit_and_eval(final_params, train_df, valid_df, test_df)

    # 模型文件和结果表分别进入 models/ 与 outputs/，方便代码和结果分开查看。
    model_dir = ensure_dir(MAINLINE_MODELS_DIR / "screen_lgbm" / f"screening_{horizon}d")
    output_dir = ensure_dir(SCREENING_OUTPUT_DIR / "unified_lgbm")
    # 这两张搜索表分别保留粗搜和细搜全记录，可以直接查看每个候选参数对应的验证表现。
    coarse_df.to_csv(output_dir / "screening_lgbm_coarse_search.csv", index=False)
    fine_df.to_csv(output_dir / "screening_lgbm_fine_search.csv", index=False)
    # 重要性表是第 7 步筛选参考总表的重要来源。
    importance_df = pd.DataFrame(
        {
            "feature_name": SCREENING_CANDIDATE_COLUMNS,
            "importance_gain": model.feature_importances_,
        }
    )
    importance_df["importance_rank"] = importance_df["importance_gain"].rank(method="dense", ascending=False).astype(int)
    importance_df = importance_df.sort_values(["importance_rank", "feature_name"]).reset_index(drop=True)
    importance_df.to_csv(output_dir / "screening_lgbm_importance.csv", index=False)
    # metrics 记录最终筛选模型在三段样本上的表现。
    metrics_df = pd.DataFrame(
        [
            {
                "horizon": horizon,
                "decision_threshold": decision_threshold,
                "train_auc": train_metrics["auc"],
                "valid_auc": valid_metrics["auc"],
                "test_auc": test_metrics["auc"],
                "train_f1": train_metrics["f1"],
                "valid_f1": valid_metrics["f1"],
                "test_f1": test_metrics["f1"],
            }
        ]
    )
    metrics_df.to_csv(output_dir / "screening_lgbm_metrics.csv", index=False)
    # best_params 表只保留最后采用的参数与阈值，便于第 11 步整理训练过程说明。
    pd.DataFrame(
        [{"horizon": horizon, "params_json": params_to_json(final_params), "decision_threshold": decision_threshold}]
    ).to_csv(output_dir / "screening_lgbm_best_params.csv", index=False)
    # 模型文件保存后，就可以直接继续查看筛选结果，不需要重复训练。
    joblib.dump({"model": model, "decision_threshold": decision_threshold, "feature_columns": SCREENING_CANDIDATE_COLUMNS}, model_dir / "screen_lgbm.joblib")


def main() -> None:
    """命令行入口；业务逻辑在 run(horizon) 中。"""
    parser = argparse.ArgumentParser(description="第 05 步：在 screening_long_panel.csv 上训练统一筛选 LightGBM，并导出重要性与搜索结果。")
    parser.add_argument("--horizon", type=int, default=None, help="当前筛选结果的期限标记。若不传，则按当前主线筛选期限执行。")
    args = parser.parse_args()
    run(args.horizon if args.horizon is not None else SCREENING_HORIZON)


if __name__ == "__main__":
    main()


from __future__ import annotations

# 本模块负责：集中维护筛选、辅助诊断和分类训练反复复用的评价指标函数。
# 直接服务的步骤：第 05、06、09、11 步。
# 关键修改位置：AUC、balanced accuracy 阈值选择和分类/回归指标的统一口径。
# 变更后重跑起点：第 05 步或第 09 步，取决于修改影响的是筛选阶段还是正式训练阶段。

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, roc_curve


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """在标签只有单一取值时返回 NaN，避免 AUC 计算直接报错。"""

    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """计算二分类得分的 KS 统计量。"""

    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(np.abs(tpr - fpr)))


def apply_threshold(y_score: np.ndarray, threshold: float) -> np.ndarray:
    """按统一规则把连续得分转为硬分类标签。"""

    return (np.asarray(y_score, dtype=float) >= threshold).astype(int)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    """返回 TN、FP、FN、TP。"""

    true = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    tn = int(np.sum((true == 0) & (pred == 0)))
    fp = int(np.sum((true == 0) & (pred == 1)))
    fn = int(np.sum((true == 1) & (pred == 0)))
    tp = int(np.sum((true == 1) & (pred == 1)))
    return tn, fp, fn, tp


def extended_classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, Any]:
    """在给定阈值下返回正式分类训练落盘使用的扩展指标。"""

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = apply_threshold(y_score, threshold)
    n = int(len(y_true))
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)

    accuracy = (tp + tn) / n if n else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    balanced_accuracy = (recall + specificity) / 2
    mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 and len(np.unique(y_true)) > 1 else 0.0
    pred_unique_count = int(pd.Series(y_pred).nunique()) if n else 0
    pred_pos_rate = float(y_pred.mean()) if n else float("nan")

    return {
        "auc": safe_auc(y_true, y_score),
        "ks": ks_statistic(y_true, y_score),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_accuracy": float(balanced_accuracy),
        "mcc": float(mcc),
        "pred_unique_count": pred_unique_count,
        "pred_pos_rate": pred_pos_rate,
        "degenerate_prediction": bool(pred_unique_count <= 1),
        "all_one_prediction": bool(pred_unique_count == 1 and pred_pos_rate == 1.0),
        "all_zero_prediction": bool(pred_unique_count == 1 and pred_pos_rate == 0.0),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def candidate_thresholds_from_valid_scores(y_score: np.ndarray) -> np.ndarray:
    """只基于验证集得分生成候选阈值。"""

    scores = np.asarray(y_score, dtype=float)
    if len(scores) == 0:
        return np.asarray([0.5], dtype=float)
    values = np.quantile(scores, np.linspace(0.001, 0.999, 999)).astype(float).tolist()
    unique_scores = np.unique(scores)
    values.extend(unique_scores.astype(float).tolist())
    values.append(0.5)
    return np.asarray(sorted({round(float(value), 15) for value in values if not pd.isna(value)}), dtype=float)


def valid_threshold_is_non_degenerate(
    metrics: dict[str, Any],
    min_pred_pos_rate: float = 0.05,
    max_pred_pos_rate: float = 0.95,
    min_class_count: int = 5,
) -> bool:
    """判断验证集阈值候选是否满足非退化约束。"""

    predicted_positive = int(metrics["tp"] + metrics["fp"])
    predicted_negative = int(metrics["tn"] + metrics["fn"])
    return (
        metrics["pred_pos_rate"] >= min_pred_pos_rate
        and metrics["pred_pos_rate"] <= max_pred_pos_rate
        and predicted_positive >= min_class_count
        and predicted_negative >= min_class_count
    )


def select_validation_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    policy: str = "valid_f1_cap075",
    min_pred_pos_rate: float = 0.05,
    max_pred_pos_rate: float = 0.95,
    min_class_count: int = 5,
) -> dict[str, Any]:
    """只在验证集上选择最终硬分类阈值，测试集不得参与阈值选择。"""

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    policy = str(policy)

    candidates = candidate_thresholds_from_valid_scores(y_score)
    rows: list[dict[str, Any]] = []
    true_pos_rate = float(np.mean(y_true)) if len(y_true) else float("nan")
    for threshold in candidates:
        metrics = extended_classification_metrics(y_true, y_score, float(threshold))
        pred_pos_rate = float(metrics["pred_pos_rate"])
        pred_pos_gap = abs(pred_pos_rate - true_pos_rate) if not pd.isna(pred_pos_rate) else float("inf")
        target_040_gap = abs(pred_pos_rate - 0.40) if not pd.isna(pred_pos_rate) else float("inf")
        predicted_positive = int(metrics["tp"] + metrics["fp"])
        predicted_negative = int(metrics["tn"] + metrics["fn"])
        passes_guard = valid_threshold_is_non_degenerate(metrics, min_pred_pos_rate, max_pred_pos_rate, min_class_count)
        rows.append(
            {
                "threshold": float(threshold),
                "balanced_accuracy": metrics["balanced_accuracy"],
                "f1": metrics["f1"],
                "mcc": metrics["mcc"],
                "pred_pos_rate": pred_pos_rate,
                "pred_pos_rate_abs_gap": pred_pos_gap,
                "target_040_abs_gap": target_040_gap,
                "predicted_positive": predicted_positive,
                "predicted_negative": predicted_negative,
                "passes_guard": passes_guard,
                "valid_f1_prev015_candidate": (
                    pred_pos_rate >= 0.10
                    and pred_pos_rate <= 0.90
                    and predicted_positive >= min_class_count
                    and predicted_negative >= min_class_count
                    and pred_pos_gap <= 0.15
                ),
                "valid_f1_cap075_candidate": (
                    pred_pos_rate >= 0.25
                    and pred_pos_rate <= 0.75
                    and predicted_positive >= min_class_count
                    and predicted_negative >= min_class_count
                ),
                "valid_target_pos_040_candidate": (
                    pred_pos_rate >= 0.20
                    and pred_pos_rate <= 0.80
                    and predicted_positive >= min_class_count
                    and predicted_negative >= min_class_count
                ),
                "metrics": metrics,
            }
        )

    candidates_df = pd.DataFrame(rows)
    if candidates_df.empty:
        fallback_metrics = extended_classification_metrics(y_true, y_score, 0.5)
        return {
            "threshold": 0.5,
            "policy": policy,
            "status": "fallback_0p5_no_valid_score",
            "objective": float(fallback_metrics["f1"]),
            "metrics": fallback_metrics,
            "candidate_count": 0,
            "eligible_count": 0,
            "search": candidates_df,
        }

    if policy == "valid_f1_prev015":
        eligible_df = candidates_df.loc[candidates_df["valid_f1_prev015_candidate"]].copy()
        fallback_df = candidates_df.loc[candidates_df["passes_guard"]].copy()
        selection_df = eligible_df if not eligible_df.empty else (fallback_df if not fallback_df.empty else candidates_df)
        status = "selected_valid_f1_prev015" if not eligible_df.empty else "fallback_valid_f1_prev015_no_candidate"
        sort_cols = ["f1", "balanced_accuracy", "mcc", "pred_pos_rate_abs_gap", "threshold"]
        ascending = [False, False, False, True, True]
        objective_col = "f1"
    elif policy == "valid_f1_cap075":
        eligible_df = candidates_df.loc[candidates_df["valid_f1_cap075_candidate"]].copy()
        fallback_df = candidates_df.loc[candidates_df["passes_guard"]].copy()
        selection_df = eligible_df if not eligible_df.empty else (fallback_df if not fallback_df.empty else candidates_df)
        status = "selected_valid_f1_cap075" if not eligible_df.empty else "fallback_valid_f1_cap075_no_candidate"
        sort_cols = ["f1", "balanced_accuracy", "mcc", "pred_pos_rate_abs_gap", "threshold"]
        ascending = [False, False, False, True, True]
        objective_col = "f1"
    elif policy == "valid_target_pos_040_nondegenerate":
        eligible_df = candidates_df.loc[candidates_df["valid_target_pos_040_candidate"]].copy()
        fallback_df = candidates_df.loc[candidates_df["passes_guard"]].copy()
        selection_df = eligible_df if not eligible_df.empty else (fallback_df if not fallback_df.empty else candidates_df)
        status = (
            "selected_valid_target_pos_040_nondegenerate"
            if not eligible_df.empty
            else "fallback_valid_target_pos_040_no_candidate"
        )
        sort_cols = ["target_040_abs_gap", "balanced_accuracy", "f1", "threshold"]
        ascending = [True, False, False, True]
        objective_col = "target_040_abs_gap"
    else:
        raise ValueError(f"unknown threshold policy: {policy}")

    chosen = selection_df.sort_values(sort_cols, ascending=ascending).iloc[0]
    return {
        "threshold": float(chosen["threshold"]),
        "policy": policy,
        "status": status,
        "objective": float(chosen[objective_col]),
        "metrics": chosen["metrics"],
        "candidate_count": int(len(rows)),
        "eligible_count": int(len(eligible_df)),
        "search": candidates_df.drop(columns=["metrics"]).copy(),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """返回回归诊断使用的 RMSE、MAE 和 R2。"""

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }

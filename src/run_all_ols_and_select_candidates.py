from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from src.common.metrics_utils import apply_threshold, extended_classification_metrics, select_validation_threshold
from src.common.model_train_utils import score_diagnostics
from src.common.paths import ensure_dir
from src.config import (
    MAINLINE_SCHEMES,
    OUTPUTS_DIR,
    PAPER_ASSETS,
    PAPER_HORIZONS,
    asset_horizon_dir,
    experiment_tag_for_horizon,
)

OLS_MODEL_ID = "ols_lpm"
OLS_MODEL_NAME = "OLS"
OLS_THRESHOLD_POLICY = "valid_target_pos_040_nondegenerate"
OLS_OUTPUT_DIR = OUTPUTS_DIR / "summary" / "ols"
OLS_RUNS_DIR = OLS_OUTPUT_DIR / "runs"


def read_model_input(asset: str, horizon: int, scheme: str, split: str) -> pd.DataFrame:
    path = asset_horizon_dir(asset, horizon, scheme, experiment_tag_for_horizon(horizon)) / f"{split}_model_input.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, parse_dates=["Date"])


def _score_diag_prefixed(prefix: str, score: np.ndarray) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in score_diagnostics(score).items()}


def _prediction_frame(
    df: pd.DataFrame,
    target_col: str,
    score: np.ndarray,
    threshold: float,
    threshold_status: str,
    split: str,
    probability_degenerate: bool,
) -> pd.DataFrame:
    pred = apply_threshold(score, threshold)
    diag = score_diagnostics(score)
    return pd.DataFrame(
        {
            "Date": df["Date"],
            "split": split,
            "y_true": df[target_col].astype(int),
            "y_score": score,
            "y_pred_valid_adj": pred,
            "y_pred": pred,
            "threshold_valid_adj": threshold,
            "threshold_policy": OLS_THRESHOLD_POLICY,
            "threshold_status": threshold_status,
            "probability_degenerate": probability_degenerate,
            "score_unique_count": diag["score_unique_count"],
            "score_range": diag["score_range"],
            "score_std": diag["score_std"],
        }
    )


def _fit_linear_probability_model(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return beta


def fit_ols_run(asset: str, horizon: int, scheme: str) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    target_col = f"target_label_{horizon}d"
    train = read_model_input(asset, horizon, scheme, "train")
    valid = read_model_input(asset, horizon, scheme, "valid")
    test = read_model_input(asset, horizon, scheme, "test")
    feature_cols = [col for col in train.columns if col not in {"Date", target_col}]

    x_train = np.column_stack([np.ones(len(train)), train[feature_cols].to_numpy(dtype=float)])
    y_train = train[target_col].to_numpy(dtype=int)
    beta = _fit_linear_probability_model(x_train, y_train)

    split_data: dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {}
    for split, df in [("train", train), ("valid", valid), ("test", test)]:
        x = np.column_stack([np.ones(len(df)), df[feature_cols].to_numpy(dtype=float)])
        y = df[target_col].to_numpy(dtype=int)
        split_data[split] = (df, y, x @ beta)

    _, y_valid, valid_score = split_data["valid"]
    threshold_choice = select_validation_threshold(
        y_valid,
        valid_score,
        policy=OLS_THRESHOLD_POLICY,
        min_pred_pos_rate=0.20,
        max_pred_pos_rate=0.80,
        min_class_count=5,
    )
    threshold = float(threshold_choice["threshold"])
    threshold_status = str(threshold_choice["status"])

    score_degenerate_by_split = {
        split: score_diagnostics(score)["score_unique_count"] <= 1
        for split, (_, _, score) in split_data.items()
    }
    probability_degenerate = bool(any(score_degenerate_by_split.values()))

    run_id = f"{OLS_MODEL_ID}_{asset}_{horizon}d__{scheme}"
    run_dir = ensure_dir(OLS_RUNS_DIR / run_id)
    prediction_frames = []

    metrics_row: dict[str, Any] = {
        "model_id": OLS_MODEL_ID,
        "model_name": OLS_MODEL_NAME,
        "asset_alias": asset,
        "horizon": horizon,
        "scheme": scheme,
        "run_id": run_id,
        "feature_count": len(feature_cols),
        "decision_threshold": threshold,
        "threshold_valid_adj": threshold,
        "threshold_policy": OLS_THRESHOLD_POLICY,
        "threshold_status": threshold_status,
        "threshold_valid_objective": threshold_choice["objective"],
        "threshold_candidate_count": threshold_choice["candidate_count"],
        "threshold_eligible_count": threshold_choice["eligible_count"],
        "probability_degenerate": probability_degenerate,
        "features": ",".join(feature_cols),
        "source_run_dir": str(run_dir),
    }

    hard_label_degenerate = False
    near_degenerate = False
    for split, (df, y_true, score) in split_data.items():
        split_metrics = extended_classification_metrics(y_true, score, threshold)
        for metric_name in [
            "auc",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "balanced_accuracy",
            "pred_pos_rate",
            "pred_unique_count",
            "degenerate_prediction",
            "all_one_prediction",
            "all_zero_prediction",
            "ks",
        ]:
            metrics_row[f"{split}_{metric_name}"] = split_metrics[metric_name]
        metrics_row.update(_score_diag_prefixed(split, score))
        hard_label_degenerate = hard_label_degenerate or bool(split_metrics["degenerate_prediction"])
        pred_pos_rate = float(split_metrics["pred_pos_rate"])
        near_degenerate = near_degenerate or bool(pred_pos_rate <= 0.05 or pred_pos_rate >= 0.95)
        pred_frame = _prediction_frame(df, target_col, score, threshold, threshold_status, split, probability_degenerate)
        pred_frame.insert(2, "asset_alias", asset)
        pred_frame.insert(3, "horizon", horizon)
        pred_frame.insert(4, "scheme", scheme)
        pred_frame.insert(5, "model_id", OLS_MODEL_ID)
        prediction_frames.append(pred_frame)
        pred_frame.to_csv(run_dir / f"pred_{split}.csv", index=False, encoding="utf-8-sig")

    metrics_row["hard_label_degenerate"] = hard_label_degenerate
    metrics_row["near_degenerate"] = near_degenerate

    metrics_df = pd.DataFrame([metrics_row])
    metrics_df.to_csv(run_dir / "metrics.csv", index=False, encoding="utf-8-sig")

    coefficients = pd.DataFrame({"feature_name": ["const"] + feature_cols, "coefficient": beta})
    coefficients.to_csv(run_dir / "coefficients.csv", index=False, encoding="utf-8-sig")

    threshold_audit = threshold_choice["search"].copy()
    threshold_audit.insert(0, "model_id", OLS_MODEL_ID)
    threshold_audit.insert(1, "asset_alias", asset)
    threshold_audit.insert(2, "horizon", horizon)
    threshold_audit.insert(3, "scheme", scheme)
    threshold_audit.insert(4, "run_id", run_id)
    threshold_audit["threshold_policy"] = OLS_THRESHOLD_POLICY
    threshold_audit["threshold_status"] = threshold_status
    threshold_audit["selected"] = np.isclose(threshold_audit["threshold"], threshold)
    threshold_audit.to_csv(run_dir / "threshold_audit.csv", index=False, encoding="utf-8-sig")

    return pd.concat(prediction_frames, ignore_index=True), metrics_row, threshold_audit


def build_pairwise(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (asset, horizon), group in metrics.groupby(["asset_alias", "horizon"], sort=True):
        if not set(MAINLINE_SCHEMES).issubset(set(group["scheme"])):
            continue
        base = group.loc[group["scheme"] == "no_tradeflow4"].iloc[0]
        expanded = group.loc[group["scheme"] == "with_tradeflow4"].iloc[0]
        row = {
            "model_id": OLS_MODEL_ID,
            "model_name": OLS_MODEL_NAME,
            "asset_alias": asset,
            "horizon": int(horizon),
            "base_accuracy": base["test_accuracy"],
            "with_accuracy": expanded["test_accuracy"],
            "accuracy_gain": expanded["test_accuracy"] - base["test_accuracy"],
            "base_precision": base["test_precision"],
            "with_precision": expanded["test_precision"],
            "precision_gain": expanded["test_precision"] - base["test_precision"],
            "base_recall": base["test_recall"],
            "with_recall": expanded["test_recall"],
            "recall_gain": expanded["test_recall"] - base["test_recall"],
            "base_f1": base["test_f1"],
            "with_f1": expanded["test_f1"],
            "f1_gain": expanded["test_f1"] - base["test_f1"],
            "base_auc": base["test_auc"],
            "with_auc": expanded["test_auc"],
            "auc_gain": expanded["test_auc"] - base["test_auc"],
            "base_balanced_accuracy": base["test_balanced_accuracy"],
            "with_balanced_accuracy": expanded["test_balanced_accuracy"],
            "balanced_accuracy_gain": expanded["test_balanced_accuracy"] - base["test_balanced_accuracy"],
            "base_probability_degenerate": base["probability_degenerate"],
            "with_probability_degenerate": expanded["probability_degenerate"],
            "base_near_degenerate": base["near_degenerate"],
            "with_near_degenerate": expanded["near_degenerate"],
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["auc_gain", "f1_gain"], ascending=[False, False]).reset_index(drop=True)


def run_all_ols() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_dir(OLS_OUTPUT_DIR)
    ensure_dir(OLS_RUNS_DIR)
    pred_frames = []
    metrics_rows = []
    audit_frames = []
    for asset in PAPER_ASSETS:
        for horizon in PAPER_HORIZONS:
            for scheme in MAINLINE_SCHEMES:
                predictions, metrics, audit = fit_ols_run(asset, horizon, scheme)
                pred_frames.append(predictions)
                metrics_rows.append(metrics)
                audit_frames.append(audit)
    predictions_df = pd.concat(pred_frames, ignore_index=True)
    metrics_df = pd.DataFrame(metrics_rows).sort_values(["asset_alias", "horizon", "scheme"]).reset_index(drop=True)
    audit_df = pd.concat(audit_frames, ignore_index=True)
    pairwise_df = build_pairwise(metrics_df)

    predictions_df.to_csv(OLS_OUTPUT_DIR / "ols_all_predictions.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(OLS_OUTPUT_DIR / "ols_all_metrics.csv", index=False, encoding="utf-8-sig")
    audit_df.to_csv(OLS_OUTPUT_DIR / "ols_all_threshold_audit.csv", index=False, encoding="utf-8-sig")
    pairwise_df.to_csv(OLS_OUTPUT_DIR / "ols_tradeflow_pairwise.csv", index=False, encoding="utf-8-sig")
    return predictions_df, metrics_df, audit_df, pairwise_df


def main() -> None:
    _, metrics, _, pairwise = run_all_ols()
    degenerate = metrics.loc[
        metrics["probability_degenerate"]
        | metrics["test_all_one_prediction"]
        | metrics["test_all_zero_prediction"]
        | metrics["near_degenerate"]
    ]
    print(f"ols_runs={len(metrics)}")
    print(f"ols_pairwise_rows={len(pairwise)}")
    print(f"ols_problem_rows={len(degenerate)}")
    if not degenerate.empty:
        print(degenerate[["asset_alias", "horizon", "scheme", "test_auc", "test_f1", "test_pred_pos_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()

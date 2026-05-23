from __future__ import annotations

# 本模块负责：封装第 09 步分类训练的公共流程，包括读入建模输入、两阶段调参、最终拟合、阈值选择、结果落盘和模型保存。
# 直接服务的步骤：第 09 步直接调用；第 10、11 步继续读取这里写出的训练结果。
# 关键修改位置：模型构造分支、两阶段搜索逻辑、结果文件结构、GPU/并行配置读取方式。
# 变更后重跑起点：第 09 步。训练流程、指标或落盘结构变化会影响汇总结果和章节图表。

import json
import math
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import (
    FIXED_TRADEFLOW_COLUMNS,
    asset_horizon_dir,
    classification_model_dir,
    classification_output_dir,
    tagged_run_name,
)
from src.common.metrics_utils import apply_threshold, extended_classification_metrics, select_validation_threshold
from src.common.paths import ensure_dir
from src.common.search_specs import (
    build_fine_candidates_from_spec,
    get_fallback_candidates,
    get_initial_params,
    get_param_specs,
)

RANDOM_STATE = 42
SCORE_RANGE_MIN = 1e-6
SCORE_STD_MIN = 1e-8


def resolve_classification_threshold_policy(asset: str, requested_policy: str) -> str:
    """把主线混合阈值口径解析为具体的验证集阈值策略。"""

    if requested_policy == "mixed_f1_by_asset":
        return "valid_f1_prev015" if asset == "index" else "valid_f1_cap075"
    return requested_policy

SEARCH_LOG_COLUMNS = [
    "stage",
    "search_id",
    "param_name",
    "candidate_value",
    "params_json",
    "train_auc",
    "valid_auc",
    "valid_balanced_accuracy",
    "valid_f1",
    "valid_ks",
    "train_valid_auc_gap",
    "train_score_unique_count",
    "valid_score_unique_count",
    "valid_score_range",
    "valid_score_std",
    "test_score_unique_count",
    "test_score_range",
    "test_score_std",
    "valid_pred_pos_rate",
    "valid_pred_unique_count",
    "valid_pred_class_0_count",
    "valid_pred_class_1_count",
    "decision_threshold",
    "threshold_status",
    "candidate_invalid",
    "candidate_invalid_reason",
    "selected_in_step",
]


@dataclass
class DatasetBundle:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    feature_cols: list[str]
    target_col: str


def gpu_acceleration_enabled() -> bool:
    """读取环境变量，判断是否启用 GPU 训练。"""

    raw = os.getenv("TRADEFLOW_USE_GPU", "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def batch_parallel_jobs() -> int:
    """读取批量训练并行进程数配置。"""

    raw = os.getenv("TRADEFLOW_BATCH_JOBS", "1").strip()
    try:
        jobs = int(raw)
    except ValueError:
        return 1
    return max(1, jobs)


def params_to_json(params: dict[str, Any]) -> str:
    """把参数字典转成稳定的 JSON 字符串。"""

    normalized = {}
    for key, value in params.items():
        if hasattr(value, "item"):
            value = value.item()
        normalized[key] = value
    return json.dumps(normalized, sort_keys=True, ensure_ascii=False)


def load_model_input_bundle(asset: str, horizon: int, scheme: str, experiment_tag: str | None = None) -> DatasetBundle:
    """读取单个 run 的 train/valid/test 建模输入。"""

    horizon_dir = asset_horizon_dir(asset, horizon, scheme, experiment_tag)
    datasets = {}
    for split in ["train", "valid", "test"]:
        path = horizon_dir / f"{split}_model_input.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        datasets[split] = pd.read_csv(path, parse_dates=["Date"])
    target_col = f"target_label_{horizon}d"
    feature_cols = [col for col in datasets["train"].columns if col not in {"Date", target_col}]
    return DatasetBundle(
        train=datasets["train"],
        valid=datasets["valid"],
        test=datasets["test"],
        feature_cols=feature_cols,
        target_col=target_col,
    )


def _positive_class_ratio(bundle: DatasetBundle) -> tuple[float, float]:
    y_train = bundle.train[bundle.target_col].astype(int)
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    ratio = neg / pos if pos > 0 else 1.0
    return float(ratio), float(math.sqrt(ratio))


def resolve_dynamic_params(params: dict[str, Any], bundle: DatasetBundle) -> dict[str, Any]:
    """把搜索空间里的动态 token 转成当前训练集对应的实际数值。"""

    resolved = deepcopy(params)
    neg_pos_ratio, sqrt_neg_pos_ratio = _positive_class_ratio(bundle)
    for key, value in list(resolved.items()):
        if value == "neg_pos_ratio":
            resolved[key] = neg_pos_ratio
        elif value == "sqrt_neg_pos_ratio":
            resolved[key] = sqrt_neg_pos_ratio
    return resolved


def make_classifier(model_id: str, params: dict[str, Any]):
    """按模型编号构造分类器实例。"""

    threaded_jobs = 1 if batch_parallel_jobs() > 1 else -1
    if model_id == "dt_cls":
        return DecisionTreeClassifier(random_state=RANDOM_STATE, **params)
    if model_id == "rf_cls":
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=threaded_jobs, **params)
    if model_id == "svm_cls":
        svm_params = deepcopy(params)
        return SVC(probability=True, cache_size=1000, random_state=RANDOM_STATE, **svm_params)
    if model_id == "xgb_cls":
        common = dict(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            objective="binary:logistic",
            tree_method="hist",
            verbosity=0,
        )
        if gpu_acceleration_enabled():
            return XGBClassifier(n_jobs=1, device="cuda", **common, **params)
        return XGBClassifier(n_jobs=threaded_jobs, **common, **params)
    if model_id == "lgbm_cls":
        common = dict(random_state=RANDOM_STATE, objective="binary", verbosity=-1)
        if gpu_acceleration_enabled():
            return LGBMClassifier(n_jobs=1, device="gpu", **common, **params)
        return LGBMClassifier(n_jobs=threaded_jobs, **common, **params)
    raise ValueError(f"unknown model_id {model_id}")


def score_diagnostics(score: np.ndarray) -> dict[str, Any]:
    """返回预测概率唯一值、取值范围和标准差诊断。"""

    values = np.asarray(score, dtype=float)
    if values.size == 0:
        return {"score_unique_count": 0, "score_range": float("nan"), "score_std": float("nan")}
    return {
        "score_unique_count": int(pd.Series(values).round(15).nunique()),
        "score_range": float(np.nanmax(values) - np.nanmin(values)),
        "score_std": float(np.nanstd(values)),
    }


def _score_diag_with_prefix(prefix: str, score: np.ndarray) -> dict[str, Any]:
    diag = score_diagnostics(score)
    return {f"{prefix}_{key}": value for key, value in diag.items()}


def lgbm_depth_constraint_invalid(model_id: str, params: dict[str, Any]) -> bool:
    """LightGBM 约束：max_depth > 0 时 num_leaves 不应超过 2 ** max_depth。"""

    if model_id != "lgbm_cls":
        return False
    max_depth = params.get("max_depth")
    num_leaves = params.get("num_leaves")
    if max_depth is None or num_leaves is None:
        return False
    try:
        max_depth_int = int(max_depth)
        num_leaves_int = int(num_leaves)
    except (TypeError, ValueError):
        return False
    return max_depth_int > 0 and num_leaves_int > 2**max_depth_int


def native_feature_importance_sum(model: Any) -> float:
    """返回模型原生重要性和系数的总强度；不可用时返回 NaN。"""

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        return float(np.nansum(np.abs(values)))
    if hasattr(model, "coef_"):
        values = np.asarray(model.coef_, dtype=float)
        return float(np.nansum(np.abs(values)))
    return float("nan")


def model_structure_diagnostics(model_id: str, model: Any) -> dict[str, Any]:
    """返回树结构和重要性诊断。"""

    out = {
        "node_count": np.nan,
        "n_leaves": np.nan,
        "feature_importance_sum": native_feature_importance_sum(model),
    }
    if hasattr(model, "tree_"):
        out["node_count"] = int(model.tree_.node_count)
        out["n_leaves"] = int(model.get_n_leaves()) if hasattr(model, "get_n_leaves") else np.nan
    return out


def _threshold_candidate_valid_metrics(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result["threshold_selection"]["metrics"]
    return {
        "valid_pred_pos_rate": metrics["pred_pos_rate"],
        "valid_pred_unique_count": metrics["pred_unique_count"],
        "valid_pred_class_0_count": int(metrics["tn"] + metrics["fn"]),
        "valid_pred_class_1_count": int(metrics["tp"] + metrics["fp"]),
    }


def candidate_invalid_reason(
    model_id: str,
    params: dict[str, Any],
    result: dict[str, Any],
    min_pred_pos_rate: float,
    max_pred_pos_rate: float,
    min_class_count: int,
) -> str:
    """按 v2 hard filter 返回候选无效原因；空字符串表示候选有效。"""

    train_diag = score_diagnostics(result["train_score"])
    valid_diag = score_diagnostics(result["valid_score"])
    pred_metrics = _threshold_candidate_valid_metrics(result)
    structure = model_structure_diagnostics(model_id, result["model"])

    if train_diag["score_unique_count"] < 2:
        return "constant_train_score"
    if valid_diag["score_unique_count"] < 2:
        return "constant_valid_score"
    if valid_diag["score_range"] <= SCORE_RANGE_MIN:
        return "valid_score_range_too_small"
    if valid_diag["score_std"] <= SCORE_STD_MIN:
        return "valid_score_std_too_small"
    if pred_metrics["valid_pred_unique_count"] != 2:
        return "valid_pred_single_class"
    if pred_metrics["valid_pred_pos_rate"] < min_pred_pos_rate:
        return "valid_pred_pos_rate_too_low"
    if pred_metrics["valid_pred_pos_rate"] > max_pred_pos_rate:
        return "valid_pred_pos_rate_too_high"
    if (
        pred_metrics["valid_pred_class_0_count"] < min_class_count
        or pred_metrics["valid_pred_class_1_count"] < min_class_count
    ):
        return "valid_pred_class_count_too_small"
    if model_id == "dt_cls":
        if structure["node_count"] <= 1 or structure["n_leaves"] <= 1:
            return "tree_single_leaf"
    if model_id in {"dt_cls", "rf_cls", "xgb_cls", "lgbm_cls"}:
        importance_sum = structure["feature_importance_sum"]
        if pd.isna(importance_sum) or float(importance_sum) <= 0:
            return "tree_zero_importance"
    if lgbm_depth_constraint_invalid(model_id, params):
        return "num_leaves_exceeds_depth_limit"
    return ""


def fit_and_score(
    model_id: str,
    params: dict[str, Any],
    bundle: DatasetBundle,
    *,
    threshold_policy: str = "valid_f1_cap075",
    min_pred_pos_rate: float = 0.05,
    max_pred_pos_rate: float = 0.95,
    min_class_count: int = 5,
) -> dict[str, Any]:
    """训练单个模型，并返回三段样本得分与阈值后指标。"""

    resolved_params = resolve_dynamic_params(params, bundle)
    if lgbm_depth_constraint_invalid(model_id, resolved_params):
        raise ValueError("num_leaves_exceeds_depth_limit")
    model = make_classifier(model_id, resolved_params)
    X_train = bundle.train[bundle.feature_cols]
    y_train = bundle.train[bundle.target_col].astype(int)
    X_valid = bundle.valid[bundle.feature_cols]
    y_valid = bundle.valid[bundle.target_col].astype(int)
    X_test = bundle.test[bundle.feature_cols]
    y_test = bundle.test[bundle.target_col].astype(int)

    model.fit(X_train, y_train)
    train_score = model.predict_proba(X_train)[:, 1]
    valid_score = model.predict_proba(X_valid)[:, 1]
    test_score = model.predict_proba(X_test)[:, 1]
    threshold_selection = select_validation_threshold(
        y_valid.to_numpy(),
        valid_score,
        policy=threshold_policy,
        min_pred_pos_rate=min_pred_pos_rate,
        max_pred_pos_rate=max_pred_pos_rate,
        min_class_count=min_class_count,
    )
    threshold = float(threshold_selection["threshold"])
    train_metrics = extended_classification_metrics(y_train.to_numpy(), train_score, threshold)
    valid_metrics = extended_classification_metrics(y_valid.to_numpy(), valid_score, threshold)
    test_metrics = extended_classification_metrics(y_test.to_numpy(), test_score, threshold)
    return {
        "model": model,
        "params": resolved_params,
        "train_score": train_score,
        "valid_score": valid_score,
        "test_score": test_score,
        "threshold_selection": threshold_selection,
        "decision_threshold": threshold,
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
    }


def _sort_key(row: dict[str, Any]) -> tuple:
    return (
        -float(np.nan_to_num(row.get("valid_auc"), nan=-1e9)),
        -float(np.nan_to_num(row.get("valid_balanced_accuracy"), nan=-1e9)),
        -float(np.nan_to_num(row.get("valid_ks"), nan=-1e9)),
        -float(np.nan_to_num(row.get("valid_f1"), nan=-1e9)),
        float(np.nan_to_num(row.get("train_valid_auc_gap"), nan=1e9)),
        int(row.get("search_id", 10**9)),
    )


def _evaluate_candidate(
    model_id: str,
    params: dict[str, Any],
    bundle: DatasetBundle,
    stage: str,
    param_name: str,
    candidate_value: Any,
    search_id: int,
    *,
    threshold_policy: str,
    min_pred_pos_rate: float,
    max_pred_pos_rate: float,
    min_class_count: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """训练并评价一个候选，返回搜索日志行和完整结果。"""

    resolved_params = resolve_dynamic_params(params, bundle)
    row: dict[str, Any] = {
        "stage": stage,
        "search_id": search_id,
        "param_name": param_name,
        "candidate_value": candidate_value,
        "params_json": params_to_json(resolved_params),
        "candidate_invalid": False,
        "candidate_invalid_reason": "",
        "selected_in_step": False,
    }
    if lgbm_depth_constraint_invalid(model_id, resolved_params):
        row.update({"candidate_invalid": True, "candidate_invalid_reason": "num_leaves_exceeds_depth_limit"})
        return _complete_search_row(row), None
    try:
        result = fit_and_score(
            model_id,
            resolved_params,
            bundle,
            threshold_policy=threshold_policy,
            min_pred_pos_rate=min_pred_pos_rate,
            max_pred_pos_rate=max_pred_pos_rate,
            min_class_count=min_class_count,
        )
    except Exception as exc:  # noqa: BLE001 - 搜索日志需要记录失败候选而不是直接吞掉上下文。
        row.update({"candidate_invalid": True, "candidate_invalid_reason": f"fit_error:{type(exc).__name__}"})
        return _complete_search_row(row), None

    pred_metrics = _threshold_candidate_valid_metrics(result)
    row.update(
        {
            "train_auc": result["train_metrics"]["auc"],
            "valid_auc": result["valid_metrics"]["auc"],
            "valid_balanced_accuracy": result["valid_metrics"]["balanced_accuracy"],
            "valid_f1": result["valid_metrics"]["f1"],
            "valid_ks": result["valid_metrics"]["ks"],
            "train_valid_auc_gap": abs(result["train_metrics"]["auc"] - result["valid_metrics"]["auc"]),
            **_score_diag_with_prefix("train", result["train_score"]),
            **_score_diag_with_prefix("valid", result["valid_score"]),
            **_score_diag_with_prefix("test", result["test_score"]),
            **pred_metrics,
            "decision_threshold": result["decision_threshold"],
            "threshold_status": result["threshold_selection"]["status"],
        }
    )
    invalid_reason = candidate_invalid_reason(
        model_id,
        resolved_params,
        result,
        min_pred_pos_rate,
        max_pred_pos_rate,
        min_class_count,
    )
    row["candidate_invalid"] = bool(invalid_reason)
    row["candidate_invalid_reason"] = invalid_reason
    return _complete_search_row(row), result


def _complete_search_row(row: dict[str, Any]) -> dict[str, Any]:
    """保证搜索日志列稳定存在。"""

    for col in SEARCH_LOG_COLUMNS:
        row.setdefault(col, np.nan)
    return row


def _choose_best_valid(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid_rows = [row for row in rows if not bool(row.get("candidate_invalid"))]
    if not valid_rows:
        return None
    return sorted(valid_rows, key=_sort_key)[0]


def _choose_best_any(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    scored = [row for row in rows if not pd.isna(row.get("valid_auc", np.nan))]
    return sorted(scored or rows, key=_sort_key)[0]


def _mark_selected(rows: list[dict[str, Any]], chosen: dict[str, Any] | None) -> None:
    if chosen is None:
        return
    for row in rows:
        if row["search_id"] == chosen["search_id"]:
            row["selected_in_step"] = True
            return


def run_two_stage_search(
    model_id: str,
    bundle: DatasetBundle,
    *,
    threshold_policy: str = "valid_f1_cap075",
    min_pred_pos_rate: float = 0.05,
    max_pred_pos_rate: float = 0.95,
    min_class_count: int = 5,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """先粗搜再细搜；若无有效候选，再进入 fallback 搜索。"""

    current_params = get_initial_params(model_id)
    param_specs = get_param_specs(model_id)
    coarse_rows: list[dict[str, Any]] = []
    fine_rows: list[dict[str, Any]] = []
    fallback_rows: list[dict[str, Any]] = []
    search_id = 0

    for param_name, spec in param_specs.items():
        step_rows = []
        for candidate in spec.coarse_candidates:
            params = deepcopy(current_params)
            params[param_name] = candidate
            row, _ = _evaluate_candidate(
                model_id,
                params,
                bundle,
                "coarse",
                param_name,
                candidate,
                search_id,
                threshold_policy=threshold_policy,
                min_pred_pos_rate=min_pred_pos_rate,
                max_pred_pos_rate=max_pred_pos_rate,
                min_class_count=min_class_count,
            )
            search_id += 1
            coarse_rows.append(row)
            step_rows.append(row)
        chosen = _choose_best_valid(step_rows) or _choose_best_any(step_rows)
        _mark_selected(step_rows, chosen)
        if chosen is not None:
            current_params = json.loads(chosen["params_json"])

    coarse_best = _choose_best_valid(coarse_rows) or _choose_best_any(coarse_rows)
    coarse_best_params = json.loads(coarse_best["params_json"]) if coarse_best is not None else current_params

    for param_name, spec in param_specs.items():
        if param_name not in coarse_best_params:
            continue
        fine_candidates = build_fine_candidates_from_spec(spec, coarse_best_params[param_name])
        if not fine_candidates:
            continue
        step_rows = []
        for candidate in fine_candidates:
            params = deepcopy(coarse_best_params)
            params[param_name] = candidate
            row, _ = _evaluate_candidate(
                model_id,
                params,
                bundle,
                "fine",
                param_name,
                candidate,
                search_id,
                threshold_policy=threshold_policy,
                min_pred_pos_rate=min_pred_pos_rate,
                max_pred_pos_rate=max_pred_pos_rate,
                min_class_count=min_class_count,
            )
            search_id += 1
            fine_rows.append(row)
            step_rows.append(row)
        chosen = _choose_best_valid(step_rows) or _choose_best_any(step_rows)
        _mark_selected(step_rows, chosen)

    fine_best = _choose_best_valid(fine_rows)
    selected_stage = "fine" if fine_best is not None else "coarse"
    selected_row = fine_best or _choose_best_valid(coarse_rows)

    if selected_row is None:
        fallback_candidates = get_fallback_candidates(model_id)
        fallback_params = deepcopy(get_initial_params(model_id))
        for param_name, candidates in fallback_candidates.items():
            step_rows = []
            for candidate in candidates:
                params = deepcopy(fallback_params)
                params[param_name] = candidate
                row, _ = _evaluate_candidate(
                    model_id,
                    params,
                    bundle,
                    "fallback",
                    param_name,
                    candidate,
                    search_id,
                    threshold_policy=threshold_policy,
                    min_pred_pos_rate=min_pred_pos_rate,
                    max_pred_pos_rate=max_pred_pos_rate,
                    min_class_count=min_class_count,
                )
                search_id += 1
                fallback_rows.append(row)
                step_rows.append(row)
            chosen = _choose_best_valid(step_rows) or _choose_best_any(step_rows)
            _mark_selected(step_rows, chosen)
            if chosen is not None:
                fallback_params = json.loads(chosen["params_json"])
        selected_row = _choose_best_valid(fallback_rows)
        selected_stage = "fallback" if selected_row is not None else "invalid_best_auc"

    if selected_row is None:
        selected_row = _choose_best_any(fallback_rows + fine_rows + coarse_rows)
        selected_stage = "invalid_best_auc"
    if selected_row is None:
        raise RuntimeError(f"no search candidate was evaluated for {model_id}")

    final_params = json.loads(selected_row["params_json"])
    search_summary = {
        "selected_stage": selected_stage,
        "selected_search_id": int(selected_row["search_id"]),
        "selected_candidate_invalid": bool(selected_row.get("candidate_invalid")),
        "selected_candidate_invalid_reason": selected_row.get("candidate_invalid_reason", ""),
    }
    return (
        final_params,
        pd.DataFrame(coarse_rows, columns=SEARCH_LOG_COLUMNS),
        pd.DataFrame(fine_rows, columns=SEARCH_LOG_COLUMNS),
        pd.DataFrame(fallback_rows, columns=SEARCH_LOG_COLUMNS),
        search_summary,
    )


def extract_feature_importance(
    model,
    feature_cols: list[str],
    X_reference: pd.DataFrame | None = None,
    y_reference: pd.Series | None = None,
) -> pd.DataFrame:
    """优先使用模型原生重要性，必要时退回到验证集置换重要性。"""

    def has_nonzero_signal(values: np.ndarray) -> bool:
        return bool(values.size) and not np.allclose(np.nan_to_num(values, nan=0.0), 0.0)

    def valid_permutation_inputs() -> bool:
        return X_reference is not None and y_reference is not None and y_reference.nunique() > 1

    importance_method = "unavailable"
    importance_std: np.ndarray | None = None
    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_, dtype=float)
        importance_method = "native_feature_importances"
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        importance = np.abs(coef).ravel()
        importance_method = "abs_linear_coef"
    elif valid_permutation_inputs():
        permutation = permutation_importance(
            model,
            X_reference,
            y_reference,
            scoring="roc_auc",
            n_repeats=10,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        importance = np.asarray(permutation.importances_mean, dtype=float)
        importance_std = np.asarray(permutation.importances_std, dtype=float)
        importance_method = "permutation_valid_auc"
    else:
        importance = np.full(len(feature_cols), np.nan, dtype=float)

    if not has_nonzero_signal(importance):
        if valid_permutation_inputs():
            permutation = permutation_importance(
                model,
                X_reference,
                y_reference,
                scoring="roc_auc",
                n_repeats=10,
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
            perm_mean = np.asarray(permutation.importances_mean, dtype=float)
            perm_std = np.asarray(permutation.importances_std, dtype=float)
            if has_nonzero_signal(perm_mean):
                importance = perm_mean
                importance_std = perm_std
                importance_method = "permutation_valid_auc"
            else:
                importance = np.full(len(feature_cols), np.nan, dtype=float)
                importance_std = None
                importance_method = "degenerate_zero_importance"
        else:
            importance = np.full(len(feature_cols), np.nan, dtype=float)
            importance_std = None
            importance_method = "degenerate_zero_importance"
    out = pd.DataFrame({"feature_name": feature_cols, "importance_gain": importance})
    if importance_std is not None:
        out["importance_std"] = importance_std
    out["importance_method"] = importance_method
    if out["importance_gain"].notna().any():
        out["importance_rank"] = out["importance_gain"].rank(method="dense", ascending=False).astype("Int64")
    else:
        out["importance_rank"] = pd.Series([pd.NA] * len(out), dtype="Int64")
    return out.sort_values(["importance_rank", "feature_name"]).reset_index(drop=True)


def tradeflow_importance_diagnostics(importance_df: pd.DataFrame, scheme: str) -> dict[str, Any]:
    """计算 with_tradeflow4 中交易流变量的重要性诊断。"""

    keys = {
        "tradeflow_importance_sum": np.nan,
        "tradeflow_importance_max": np.nan,
        "tradeflow_importance_rank_min": np.nan,
        "tradeflow_top3_count": np.nan,
        "tradeflow_top5_count": np.nan,
    }
    if scheme != "with_tradeflow4" or importance_df.empty:
        return keys
    subset = importance_df.loc[importance_df["feature_name"].isin(FIXED_TRADEFLOW_COLUMNS)].copy()
    if subset.empty or subset["importance_gain"].isna().all():
        return keys
    keys["tradeflow_importance_sum"] = float(subset["importance_gain"].fillna(0.0).sum())
    keys["tradeflow_importance_max"] = float(subset["importance_gain"].fillna(0.0).max())
    ranks = pd.to_numeric(subset["importance_rank"], errors="coerce")
    keys["tradeflow_importance_rank_min"] = float(ranks.min()) if ranks.notna().any() else np.nan
    keys["tradeflow_top3_count"] = int((ranks <= 3).sum()) if ranks.notna().any() else np.nan
    keys["tradeflow_top5_count"] = int((ranks <= 5).sum()) if ranks.notna().any() else np.nan
    return keys


def split_probability_degenerate(score: np.ndarray) -> bool:
    return score_diagnostics(score)["score_unique_count"] <= 1


def save_predictions(
    df: pd.DataFrame,
    target_col: str,
    scores: np.ndarray,
    threshold_valid_adj: float,
    threshold_policy: str,
    threshold_status: str,
    split: str,
    path: Path,
    probability_degenerate: bool,
) -> None:
    """把真实标签、预测标签、得分和概率诊断写成预测结果文件。"""

    preds_valid_adj = apply_threshold(scores, threshold_valid_adj)
    diag = score_diagnostics(scores)
    out = pd.DataFrame(
        {
            "Date": df["Date"],
            "split": split,
            "y_true": df[target_col].astype(int),
            "y_score": scores,
            "y_pred_valid_adj": preds_valid_adj,
            "y_pred": preds_valid_adj,
            "threshold_valid_adj": threshold_valid_adj,
            "threshold_policy": threshold_policy,
            "threshold_status": threshold_status,
            "probability_degenerate": probability_degenerate,
            "score_unique_count": diag["score_unique_count"],
            "score_range": diag["score_range"],
            "score_std": diag["score_std"],
        }
    )
    out.to_csv(path, index=False)


def _final_probability_degenerate(fit_result: dict[str, Any], selected_invalid: bool) -> bool:
    if selected_invalid:
        return True
    return any(
        split_probability_degenerate(fit_result[f"{split}_score"])
        for split in ["train", "valid", "test"]
    )


def run_classification_training(
    model_id: str,
    asset: str,
    horizon: int,
    scheme: str,
    experiment_tag: str | None = None,
    classification_output_root: Path | None = None,
    classification_model_root: Path | None = None,
    threshold_policy: str = "mixed_f1_by_asset",
    min_pred_pos_rate: float = 0.05,
    max_pred_pos_rate: float = 0.95,
    min_class_count: int = 5,
    search_primary_metric: str = "valid_auc",
) -> dict[str, Any]:
    """执行单个分类 run 的完整训练、搜索、落盘和模型保存。"""

    bundle = load_model_input_bundle(asset, horizon, scheme, experiment_tag)
    resolved_threshold_policy = resolve_classification_threshold_policy(asset, threshold_policy)
    final_params, coarse_df, fine_df, fallback_df, search_summary = run_two_stage_search(
        model_id,
        bundle,
        threshold_policy=resolved_threshold_policy,
        min_pred_pos_rate=min_pred_pos_rate,
        max_pred_pos_rate=max_pred_pos_rate,
        min_class_count=min_class_count,
    )
    fit_result = fit_and_score(
        model_id,
        final_params,
        bundle,
        threshold_policy=resolved_threshold_policy,
        min_pred_pos_rate=min_pred_pos_rate,
        max_pred_pos_rate=max_pred_pos_rate,
        min_class_count=min_class_count,
    )
    final_model = fit_result["model"]
    final_structure = model_structure_diagnostics(model_id, final_model)
    final_invalid_reason = candidate_invalid_reason(
        model_id,
        final_params,
        fit_result,
        min_pred_pos_rate,
        max_pred_pos_rate,
        min_class_count,
    )
    probability_degenerate = _final_probability_degenerate(fit_result, bool(final_invalid_reason))

    threshold_selection = fit_result["threshold_selection"]
    decision_threshold = float(threshold_selection["threshold"])
    threshold_policy_name = str(threshold_selection["policy"])
    threshold_status = (
        "probability_degenerate_threshold_not_effective"
        if probability_degenerate and final_invalid_reason in {"constant_train_score", "constant_valid_score", "tree_single_leaf", "tree_zero_importance"}
        else str(threshold_selection["status"])
    )

    run_id = tagged_run_name(asset, horizon, scheme, experiment_tag)
    model_dir = ensure_dir(classification_model_dir(model_id, run_id, classification_model_root))
    output_dir = ensure_dir(classification_output_dir(model_id, run_id, classification_output_root))

    coarse_df.to_csv(output_dir / "coarse_search.csv", index=False)
    fine_df.to_csv(output_dir / "fine_search.csv", index=False)
    fallback_df.to_csv(output_dir / "fallback_search.csv", index=False)

    importance_df = extract_feature_importance(
        final_model,
        bundle.feature_cols,
        bundle.valid[bundle.feature_cols],
        bundle.valid[bundle.target_col].astype(int),
    )
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
    tradeflow_diag = tradeflow_importance_diagnostics(importance_df, scheme)

    metrics_row = {
        "model_id": model_id,
        "asset_alias": asset,
        "horizon": horizon,
        "scheme": scheme,
        "run_id": run_id,
        "scheme_name": run_id,
        "experiment_tag": experiment_tag,
        "decision_threshold": decision_threshold,
        "requested_threshold_policy": threshold_policy,
        "threshold_policy": threshold_policy_name,
        "threshold_status": threshold_status,
        "threshold_valid_adj": decision_threshold,
        "threshold_valid_objective": threshold_selection["objective"],
        "threshold_candidate_count": threshold_selection["candidate_count"],
        "threshold_eligible_count": threshold_selection["eligible_count"],
        "min_pred_pos_rate": min_pred_pos_rate,
        "max_pred_pos_rate": max_pred_pos_rate,
        "min_class_count": min_class_count,
        "search_primary_metric": search_primary_metric,
        "selected_search_stage": search_summary["selected_stage"],
        "selected_search_id": search_summary["selected_search_id"],
        "selected_candidate_invalid": search_summary["selected_candidate_invalid"],
        "selected_candidate_invalid_reason": search_summary["selected_candidate_invalid_reason"],
        "final_candidate_invalid_reason": final_invalid_reason,
        "feature_count": len(bundle.feature_cols),
        "target_col": bundle.target_col,
        "probability_degenerate": probability_degenerate,
        "node_count": final_structure["node_count"],
        "n_leaves": final_structure["n_leaves"],
        "feature_importance_sum": final_structure["feature_importance_sum"],
        **tradeflow_diag,
    }
    hard_label_degenerate = False
    near_degenerate = False
    for split_name, df, score in [
        ("train", bundle.train, fit_result["train_score"]),
        ("valid", bundle.valid, fit_result["valid_score"]),
        ("test", bundle.test, fit_result["test_score"]),
    ]:
        y_true = df[bundle.target_col].astype(int).to_numpy()
        split_metrics = extended_classification_metrics(y_true, score, decision_threshold)
        for metric_name in ["auc", "accuracy", "precision", "recall", "f1", "balanced_accuracy", "pred_pos_rate", "pred_unique_count", "degenerate_prediction", "all_one_prediction", "all_zero_prediction", "ks"]:
            metrics_row[f"{split_name}_{metric_name}"] = split_metrics[metric_name]
        diag = score_diagnostics(score)
        for key, value in diag.items():
            metrics_row[f"{split_name}_{key}"] = value
        hard_label_degenerate = hard_label_degenerate or bool(split_metrics["degenerate_prediction"])
        pred_pos_rate = split_metrics["pred_pos_rate"]
        near_degenerate = near_degenerate or bool(pred_pos_rate <= 0.05 or pred_pos_rate >= 0.95)
    metrics_row["hard_label_degenerate"] = hard_label_degenerate
    metrics_row["near_degenerate"] = near_degenerate
    pd.DataFrame([metrics_row]).to_csv(output_dir / "metrics.csv", index=False)

    best_params_row = {
        "model_id": model_id,
        "asset_alias": asset,
        "horizon": horizon,
        "scheme": scheme,
        "run_id": run_id,
        "scheme_name": run_id,
        "experiment_tag": experiment_tag,
        "decision_threshold": decision_threshold,
        "requested_threshold_policy": threshold_policy,
        "threshold_policy": threshold_policy_name,
        "threshold_valid_adj": decision_threshold,
        "threshold_status": threshold_status,
        "min_pred_pos_rate": min_pred_pos_rate,
        "max_pred_pos_rate": max_pred_pos_rate,
        "min_class_count": min_class_count,
        "search_primary_metric": search_primary_metric,
        **search_summary,
        "params_json": params_to_json(final_params),
    }
    pd.DataFrame([best_params_row]).to_csv(output_dir / "best_params.csv", index=False)

    save_predictions(bundle.train, bundle.target_col, fit_result["train_score"], decision_threshold, threshold_policy_name, threshold_status, "train", output_dir / "pred_train.csv", probability_degenerate)
    save_predictions(bundle.valid, bundle.target_col, fit_result["valid_score"], decision_threshold, threshold_policy_name, threshold_status, "valid", output_dir / "pred_valid.csv", probability_degenerate)
    save_predictions(bundle.test, bundle.target_col, fit_result["test_score"], decision_threshold, threshold_policy_name, threshold_status, "test", output_dir / "pred_test.csv", probability_degenerate)

    joblib.dump(
        {
            "model": final_model,
            "decision_threshold": decision_threshold,
            "requested_threshold_policy": threshold_policy,
            "threshold_policy": threshold_policy_name,
            "threshold_status": threshold_status,
            "threshold_valid_adj": decision_threshold,
            "feature_columns": bundle.feature_cols,
            "target_col": bundle.target_col,
            "scheme": scheme,
            "experiment_tag": experiment_tag,
            "params": final_params,
            "probability_degenerate": probability_degenerate,
        },
        model_dir / f"{model_id}.joblib",
    )
    return metrics_row

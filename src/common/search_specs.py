from __future__ import annotations

# 本模块负责：统一维护筛选模型、交易流辅助诊断模型和主线分类模型的初始参数、粗搜索空间和细搜索空间。
# 直接服务的步骤：第 05、06、09 步。
# 关键修改位置：`MODEL_INITIAL_PARAMS`、`MODEL_PARAM_SPECS`、`MODEL_FALLBACK_PARAM_SPECS`。
# 变更后重跑起点：第 05 步或第 09 步，取决于修改影响的是筛选训练还是正式分类训练。

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParamSpec:
    """描述一个参数在粗搜索、细搜索和 fallback 搜索中的取值规则。"""

    coarse_candidates: tuple[Any, ...]
    fine_enabled: bool
    fine_min_value: float | None = None
    fine_max_value: float | None = None
    legal_values: tuple[Any, ...] | None = None
    is_integer: bool = False
    fine_candidates: tuple[Any, ...] | None = None
    fallback_candidates: tuple[Any, ...] | None = None


def _spec(
    coarse: tuple[Any, ...],
    fine: tuple[Any, ...] | None = None,
    fallback: tuple[Any, ...] | None = None,
    *,
    fine_enabled: bool | None = None,
    is_integer: bool = False,
) -> ParamSpec:
    """简化 v2 搜索空间定义。"""

    enabled = fine_enabled if fine_enabled is not None else fine is not None
    return ParamSpec(
        coarse_candidates=coarse,
        fine_enabled=enabled,
        is_integer=is_integer,
        fine_candidates=fine,
        fallback_candidates=fallback,
    )


# 这一组是每个模型真正开始粗搜索前的默认参数。
MODEL_INITIAL_PARAMS = {
    "dt_cls": {
        "criterion": "gini",
        "max_depth": 4,
        "min_samples_leaf": 20,
        "min_samples_split": 40,
        "max_features": None,
        "ccp_alpha": 0.0,
        "min_impurity_decrease": 0.0,
        "class_weight": "balanced",
    },
    "rf_cls": {
        "n_estimators": 300,
        "criterion": "gini",
        "max_depth": 8,
        "min_samples_leaf": 5,
        "min_samples_split": 10,
        "max_features": "sqrt",
        "bootstrap": True,
        "class_weight": "balanced_subsample",
        "ccp_alpha": 0.0,
    },
    "svm_cls": {
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
        "class_weight": "balanced",
    },
    "xgb_cls": {
        "n_estimators": 300,
        "learning_rate": 0.03,
        "max_depth": 3,
        "min_child_weight": 1,
        "gamma": 0.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "scale_pos_weight": 1.0,
    },
    "lgbm_cls": {
        "n_estimators": 300,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 5,
        "min_child_samples": 10,
        "min_child_weight": 0.001,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "min_split_gain": 0.0,
        "class_weight": "balanced",
        "importance_type": "gain",
    },
}


MODEL_PARAM_SPECS = {
    "dt_cls": {
        "criterion": _spec(("gini", "entropy", "log_loss"), fallback=("gini", "entropy"), fine_enabled=False),
        "max_depth": _spec(
            (2, 3, 4, 5, 6, 8, None),
            (1, 2, 3, 4, 5, 6, 8, 10, None),
            (4, 5, 6, 8),
            is_integer=True,
        ),
        "min_samples_leaf": _spec((1, 2, 5, 10, 20, 40, 60), (1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 60, 80), (1, 2, 5, 10, 20), is_integer=True),
        "min_samples_split": _spec((2, 5, 10, 20, 40, 80, 120), (2, 5, 8, 10, 15, 20, 30, 40, 60, 80, 120, 160), (2, 5, 10, 20, 40), is_integer=True),
        "max_features": _spec((None, "sqrt", "log2", 0.6, 0.8, 1.0), fallback=(None, 0.8, 1.0), fine_enabled=False),
        "ccp_alpha": _spec((0.0, 0.00001, 0.0001, 0.0005, 0.001, 0.003, 0.005), (0.0, 0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.003, 0.005, 0.01), (0.0, 0.000001, 0.00001, 0.0001)),
        "min_impurity_decrease": _spec((0.0, 0.000001, 0.00001, 0.0001, 0.0005, 0.001), (0.0, 0.0000001, 0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001), (0.0, 0.000001, 0.00001)),
        "class_weight": _spec((None, "balanced"), fallback=("balanced", None), fine_enabled=False),
    },
    "rf_cls": {
        "n_estimators": _spec((100, 200, 300, 500, 800), (100, 150, 200, 300, 400, 500, 650, 800), (300, 500, 800), is_integer=True),
        "criterion": _spec(("gini", "entropy", "log_loss"), fine_enabled=False),
        "max_depth": _spec((3, 5, 8, 12, None), (2, 3, 5, 8, 10, 12, 16, None), (8, 12, None), is_integer=True),
        "min_samples_leaf": _spec((1, 2, 5, 10, 20, 40), (1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 60), (1, 2, 5), is_integer=True),
        "min_samples_split": _spec((2, 5, 10, 20, 40, 80), (2, 3, 5, 8, 10, 15, 20, 30, 40, 60, 80, 120), (2, 5, 10), is_integer=True),
        "max_features": _spec(("sqrt", "log2", 0.5, 0.8, None), fine_enabled=False),
        "class_weight": _spec((None, "balanced", "balanced_subsample"), fallback=("balanced_subsample",), fine_enabled=False),
        "ccp_alpha": _spec((0.0, 0.00001, 0.0001, 0.001), (0.0, 0.000001, 0.00001, 0.0001, 0.0005, 0.001)),
    },
    "svm_cls": {
        "kernel": _spec(("linear", "rbf"), fine_enabled=False),
        "C": _spec((0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0), (0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 50.0)),
        "gamma": _spec(("scale", "auto", 0.001, 0.003, 0.01, 0.03, 0.1, 0.3), ("scale", "auto", 0.0003, 0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5)),
        "class_weight": _spec((None, "balanced"), fine_enabled=False),
    },
    "xgb_cls": {
        "n_estimators": _spec((100, 200, 300, 500, 800), (80, 100, 150, 200, 250, 300, 400, 500, 650, 800), (300, 500, 800), is_integer=True),
        "learning_rate": _spec((0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1), (0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15), (0.02, 0.03, 0.05)),
        "max_depth": _spec((1, 2, 3, 4, 5, 6), (1, 2, 3, 4, 5, 6, 8), (2, 3, 4), is_integer=True),
        "min_child_weight": _spec((1, 2, 3, 5, 10, 20), (1, 2, 3, 5, 7, 10, 15, 20), (1, 2, 3)),
        "gamma": _spec((0.0, 0.01, 0.05, 0.1, 0.3, 0.5), (0.0, 0.001, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5), (0.0, 0.01, 0.05)),
        "subsample": _spec((0.6, 0.7, 0.8, 0.9, 1.0), (0.5, 0.6, 0.7, 0.8, 0.9, 1.0), (0.8, 0.9, 1.0)),
        "colsample_bytree": _spec((0.6, 0.7, 0.8, 0.9, 1.0), (0.5, 0.6, 0.7, 0.8, 0.9, 1.0), (0.8, 0.9, 1.0)),
        "reg_lambda": _spec((0.1, 0.3, 1.0, 3.0, 5.0, 10.0), (0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 5.0, 10.0, 20.0), (0.1, 1.0, 3.0)),
        "reg_alpha": _spec((0.0, 0.001, 0.01, 0.1, 0.3, 1.0), (0.0, 0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0), (0.0, 0.001, 0.01)),
        "scale_pos_weight": _spec((1.0, "neg_pos_ratio", "sqrt_neg_pos_ratio"), fine_enabled=False),
    },
    "lgbm_cls": {
        "n_estimators": _spec((100, 200, 300, 500, 800), (80, 100, 150, 200, 250, 300, 400, 500, 650, 800), (300, 500), is_integer=True),
        "learning_rate": _spec((0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1), (0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15), (0.02, 0.03, 0.05)),
        "num_leaves": _spec((7, 15, 31, 63, 127), (3, 5, 7, 10, 15, 20, 31, 45, 63, 90, 127), (15, 31, 63), is_integer=True),
        "max_depth": _spec((2, 3, 5, 7, 10, -1), (2, 3, 4, 5, 6, 7, 8, 10, -1), (5, 7, -1), is_integer=True),
        "min_child_samples": _spec((5, 10, 20, 40, 80), (2, 5, 8, 10, 15, 20, 30, 40, 60, 80, 120), (5, 10, 20), is_integer=True),
        "min_child_weight": _spec((0.0001, 0.001, 0.01, 0.1, 1.0), (0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0), (0.0001, 0.001, 0.01)),
        "subsample": _spec((0.6, 0.7, 0.8, 0.9, 1.0), (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)),
        "colsample_bytree": _spec((0.6, 0.7, 0.8, 0.9, 1.0), (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)),
        "reg_lambda": _spec((0.0, 0.1, 0.3, 1.0, 3.0, 5.0, 10.0), (0.0, 0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 5.0, 10.0, 20.0), (0.1, 1.0, 3.0)),
        "reg_alpha": _spec((0.0, 0.001, 0.01, 0.1, 0.3, 1.0), (0.0, 0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0), (0.0, 0.001, 0.01)),
        "min_split_gain": _spec((0.0, 0.001, 0.01, 0.05, 0.1, 0.3), (0.0, 0.0001, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3), (0.0, 0.001)),
        "class_weight": _spec((None, "balanced"), fallback=("balanced",), fine_enabled=False),
    },
}


SCREEN_LGBM_INITIAL_PARAMS = deepcopy(MODEL_INITIAL_PARAMS["lgbm_cls"])
SCREEN_LGBM_PARAM_SPECS = deepcopy(MODEL_PARAM_SPECS["lgbm_cls"])
TRADEFLOW_LGBM_INITIAL_PARAMS = deepcopy(MODEL_INITIAL_PARAMS["lgbm_cls"])
TRADEFLOW_LGBM_PARAM_SPECS = deepcopy(MODEL_PARAM_SPECS["lgbm_cls"])


def get_initial_params(model_id: str) -> dict[str, Any]:
    """返回指定模型的主线初始参数。"""

    return deepcopy(MODEL_INITIAL_PARAMS[model_id])


def get_param_specs(model_id: str) -> dict[str, ParamSpec]:
    """返回指定模型的两阶段调参搜索空间。"""

    return deepcopy(MODEL_PARAM_SPECS[model_id])


def get_fallback_candidates(model_id: str) -> dict[str, tuple[Any, ...]]:
    """返回指定模型的 fallback 搜索空间。"""

    specs = MODEL_PARAM_SPECS[model_id]
    return {
        param_name: tuple(spec.fallback_candidates)
        for param_name, spec in specs.items()
        if spec.fallback_candidates is not None and len(spec.fallback_candidates) > 0
    }


def build_fine_candidates_from_spec(spec: ParamSpec, center_value: Any) -> list[Any]:
    """围绕粗搜索最优点生成细搜索候选值；v2 优先使用显式细搜主列表。"""

    if not spec.fine_enabled:
        return []
    if spec.fine_candidates is not None:
        return list(dict.fromkeys(spec.fine_candidates))
    if spec.legal_values is not None:
        ordered = sorted(spec.legal_values)
        center_index = min(range(len(ordered)), key=lambda i: abs(ordered[i] - center_value))
        start = max(0, center_index - 2)
        end = min(len(ordered), center_index + 3)
        return list(dict.fromkeys(ordered[start:end]))

    coarse_values = list(spec.coarse_candidates)
    if len(coarse_values) < 2:
        return []
    step = (coarse_values[1] - coarse_values[0]) / 3.0
    raw_candidates = [
        center_value - 2 * step,
        center_value - step,
        center_value,
        center_value + step,
        center_value + 2 * step,
    ]
    bounded = []
    for value in raw_candidates:
        if spec.fine_min_value is not None:
            value = max(spec.fine_min_value, value)
        if spec.fine_max_value is not None:
            value = min(spec.fine_max_value, value)
        if spec.is_integer:
            value = int(round(value))
        bounded.append(value)
    unique = list(dict.fromkeys(bounded))
    return unique if len(unique) >= 3 else []

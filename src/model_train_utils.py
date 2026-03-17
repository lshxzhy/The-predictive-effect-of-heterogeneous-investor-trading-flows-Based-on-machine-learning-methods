import math
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from config import RuntimeContext
from training_utils import (
    align_features,
    build_prediction_frame,
    evaluate_model,
    load_model_bundle,
    require_feature_names,
    save_model_bundle,
)


METRIC_COLS = ["accuracy", "precision", "recall", "f1", "auc", "ks"]
SEARCH_LOG_HELPER_COLS = [
    "meet_valid_auc_le_train_auc",
    "meet_coarse_valid_auc_floor",
    "meet_fine_selection_requirements",
    "coarse_best_valid_auc",
]
SELECTION_RULE_TEXT = (
    "第一轮逐个参数只接受 valid_auc 不高于 train_auc 的候选，再取 valid_auc 最高；"
    "第二轮只接受 valid_auc 不低于第一轮最优且 valid_auc 不高于 train_auc 的候选，"
    "再从中选 train_valid_auc_gap 最小的一组"
)


@dataclass(frozen=True)
class SearchParameterSpec:
    """单个超参数的搜索配置。"""

    name: str
    meaning: str
    trend_note: str
    coarse_candidates: tuple[object, ...]
    fine_builder: Callable[[object], list[object]]


@dataclass(frozen=True)
class ModelTrainingSpec:
    """单个模型的训练与搜索配置。"""

    model_id: str
    display_name: str
    initial_params: dict[str, object]
    search_params: tuple[SearchParameterSpec, ...]
    build_model_fn: Callable[[dict[str, object]], object]


# 用统一文件名保存单模型结果，避免不同模型脚本各写一套命名约定。
def cleanup_legacy_output_files(ctx: RuntimeContext, model_id: str) -> None:
    """清理模型目录下已经废弃的旧结果文件。"""
    legacy_files = [
        ctx.paths.output_file(model_id, ctx.scheme_name, "cv_results.csv"),
        ctx.paths.output_file(model_id, ctx.scheme_name, "search_results.csv"),
    ]
    for file_path in legacy_files:
        if file_path.exists():
            file_path.unlink()


# 读取单个 split 的模型输入文件，后续统一按 Date + y + X 拆开。
def load_split_frame(ctx: RuntimeContext, split_name: str) -> pd.DataFrame:
    """读取单个数据分段。"""
    return pd.read_csv(ctx.paths.model_input_file(split_name), parse_dates=["Date"])


# 单期限建模层的结构固定是 Date、标签列和特征列，
# 这里把日期、X、y 统一拆开给搜索和训练复用。
def split_to_xy(
    df: pd.DataFrame,
    ctx: RuntimeContext,
) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """拆分日期、特征和标签。"""
    target_label_col = ctx.config.target_label_col(ctx.horizon)
    required_cols = ["Date", target_label_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{ctx.scheme_name} 缺少建模列：{missing_cols}")

    dates = df["Date"].copy()
    X = df.drop(columns=["Date", target_label_col])
    y = df[target_label_col].astype(int)
    return dates, X, y


# 训练、验证、测试三段都从已经准备好的 model_input.csv 读取，
# 并强制用训练集的列顺序对齐 valid/test。
def load_datasets(
    ctx: RuntimeContext,
) -> tuple[dict[str, dict[str, object]], list[str]]:
    """读取训练集、验证集和测试集。"""
    datasets: dict[str, dict[str, object]] = {}

    for split_name in ["train", "valid", "test"]:
        df = load_split_frame(ctx, split_name)
        dates, X, y = split_to_xy(df, ctx)
        datasets[split_name] = {"dates": dates, "X": X, "y": y}

    feature_names = datasets["train"]["X"].columns.tolist()
    for split_name in ["valid", "test"]:
        datasets[split_name]["X"] = align_features(
            datasets[split_name]["X"],
            feature_names,
        )

    return datasets, feature_names


# 统一把单次候选参数在 train/valid 上拟合并评估成两行指标，
# 便于后面构造粗搜和细搜日志。
def fit_and_evaluate_on_holdout(
    spec: ModelTrainingSpec,
    params: dict[str, object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> tuple[object, pd.Series, pd.Series]:
    """拟合单组参数并返回训练集与验证集评估。"""
    model = spec.build_model_fn(params)
    model.fit(X_train, y_train)
    train_metrics = evaluate_model(model, X_train, y_train, "train").iloc[0]
    valid_metrics = evaluate_model(model, X_valid, y_valid, "valid").iloc[0]
    return model, train_metrics, valid_metrics


# 搜索日志一律保留完整参数快照、train/valid 指标和 AUC gap，
# 方便后续解释是“验证集更高”还是“过拟合更轻”。
def build_search_row(
    spec: ModelTrainingSpec,
    stage: str,
    search_id: int,
    tuned_parameter: str,
    params: dict[str, object],
    train_metrics: pd.Series,
    valid_metrics: pd.Series,
) -> dict[str, object]:
    """整理单次搜索结果。"""
    row: dict[str, object] = {
        "stage": stage,
        "search_id": search_id,
        "tuned_parameter": tuned_parameter,
    }
    row.update({param_spec.name: params[param_spec.name] for param_spec in spec.search_params})

    for metric_col in METRIC_COLS:
        row[f"train_{metric_col}"] = train_metrics[metric_col]
        row[f"valid_{metric_col}"] = valid_metrics[metric_col]

    train_auc = train_metrics["auc"]
    valid_auc = valid_metrics["auc"]
    row["meet_valid_auc_le_train_auc"] = (
        pd.notna(train_auc)
        and pd.notna(valid_auc)
        and valid_auc <= train_auc
    )
    row["auc_gap"] = (
        abs(train_auc - valid_auc)
        if pd.notna(train_auc) and pd.notna(valid_auc)
        else float("inf")
    )
    return row


# 第一轮先筛掉 valid_auc 高于 train_auc 的候选，再在合格候选里按 valid_auc 选优。
def rank_coarse_step(step_df: pd.DataFrame) -> pd.DataFrame:
    """按第一轮规则排序单参数粗搜结果。"""
    if not step_df["meet_valid_auc_le_train_auc"].any():
        raise ValueError(
            "当前粗搜参数步不存在满足 valid_auc <= train_auc 的候选，"
            "请检查搜索网格或上游数据。"
        )
    ranked_df = step_df.sort_values(
        ["meet_valid_auc_le_train_auc", "valid_auc", "auc_gap", "train_auc", "search_id"],
        ascending=[False, False, True, False, True],
    ).reset_index(drop=True)
    ranked_df["selection_rank"] = ranked_df.index + 1
    return ranked_df


# 第二轮只接受同时满足“valid_auc 不低于粗搜最优”且“valid_auc 不高于 train_auc”的候选，
# 再在达标候选里优先压缩 train-valid gap。
def rank_fine_step(step_df: pd.DataFrame, coarse_best_valid_auc: float) -> pd.DataFrame:
    """按第二轮规则排序单参数细搜结果。"""
    ranked_df = step_df.copy()
    ranked_df["meet_coarse_valid_auc_floor"] = ranked_df["valid_auc"] >= coarse_best_valid_auc
    ranked_df["meet_fine_selection_requirements"] = (
        ranked_df["meet_coarse_valid_auc_floor"]
        & ranked_df["meet_valid_auc_le_train_auc"]
    )
    ranked_df = ranked_df.sort_values(
        [
            "meet_fine_selection_requirements",
            "auc_gap",
            "valid_auc",
            "train_auc",
            "search_id",
        ],
        ascending=[False, True, False, False, True],
    ).reset_index(drop=True)
    ranked_df["selection_rank"] = ranked_df.index + 1
    return ranked_df


# 整数型参数统一围绕当前中心值做对称微调，并强制截到合法区间。
def build_int_candidates(
    center: int,
    step: int,
    min_value: int,
    max_value: int | None = None,
) -> list[int]:
    """围绕整数中心值构造细搜候选。"""
    candidates = [
        center - 2 * step,
        center - step,
        center,
        center + step,
        center + 2 * step,
    ]
    normalized_candidates = []
    for candidate in candidates:
        clipped = max(min_value, int(candidate))
        if max_value is not None:
            clipped = min(clipped, max_value)
        normalized_candidates.append(clipped)
    return sorted(set(normalized_candidates))


# 浮点型参数统一围绕当前中心值做对称微调，并保留四位小数。
def build_float_candidates(
    center: float,
    step: float,
    min_value: float,
    max_value: float,
) -> list[float]:
    """围绕浮点中心值构造细搜候选。"""
    candidates = [
        center - 2 * step,
        center - step,
        center,
        center + step,
        center + 2 * step,
    ]
    normalized_candidates = []
    for candidate in candidates:
        clipped = min(max(float(candidate), min_value), max_value)
        normalized_candidates.append(round(clipped, 4))
    return sorted(set(normalized_candidates))


# 对正则或阈值类参数，小值附近用更小步长，大值附近适当放宽步长。
def build_regularization_candidates(
    center: float,
    max_value: float = 10.0,
) -> list[float]:
    """构造正则项细搜候选。"""
    step = 0.0005 if center <= 0.005 else (0.01 if center <= 0.1 else 0.25 if center > 1.0 else 0.1)
    return build_float_candidates(center=center, step=step, min_value=0.0, max_value=max_value)


# 某些权重类参数更适合按倍数而不是固定步长微调。
def build_scaled_float_candidates(
    center: float,
    factors: tuple[float, ...],
    min_value: float,
    max_value: float,
) -> list[float]:
    """按倍数围绕中心值构造细搜候选。"""
    normalized_candidates = []
    for factor in factors:
        candidate = min(max(float(center) * factor, min_value), max_value)
        normalized_candidates.append(round(candidate, 4))
    return sorted(set(normalized_candidates))


# 枚举型参数细搜不做插值，只在当前值和可选集合里切换。
def build_choice_candidates(center: object, candidates: tuple[object, ...]) -> list[object]:
    """构造离散候选参数的细搜集合。"""
    ordered_candidates = [center, *candidates]
    deduped_candidates: list[object] = []
    for candidate in ordered_candidates:
        if candidate not in deduped_candidates:
            deduped_candidates.append(candidate)
    return deduped_candidates


# 第一轮粗搜沿当前参数状态逐个锁定，
# 每一步都只修改一个参数，其余参数保持上一步最优值。
def run_coarse_search(
    spec: ModelTrainingSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> tuple[pd.DataFrame, dict[str, object], float]:
    """执行第一轮粗搜。"""
    current_params = spec.initial_params.copy()
    all_rows: list[dict[str, object]] = []
    selected_search_ids: list[int] = []
    search_id = 1

    for param_spec in spec.search_params:
        step_rows: list[dict[str, object]] = []
        for candidate in param_spec.coarse_candidates:
            candidate_params = current_params.copy()
            candidate_params[param_spec.name] = candidate
            _, train_metrics, valid_metrics = fit_and_evaluate_on_holdout(
                spec=spec,
                params=candidate_params,
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                y_valid=y_valid,
            )
            row = build_search_row(
                spec=spec,
                stage="coarse",
                search_id=search_id,
                tuned_parameter=param_spec.name,
                params=candidate_params,
                train_metrics=train_metrics,
                valid_metrics=valid_metrics,
            )
            all_rows.append(row)
            step_rows.append(row)
            search_id += 1

        ranked_step_df = rank_coarse_step(pd.DataFrame(step_rows))
        best_step_row = ranked_step_df.iloc[0]
        current_params = {candidate_spec.name: best_step_row[candidate_spec.name] for candidate_spec in spec.search_params}
        selected_search_ids.append(int(best_step_row["search_id"]))

    coarse_df = pd.DataFrame(all_rows)
    coarse_df["step_selected"] = coarse_df["search_id"].isin(selected_search_ids)
    coarse_best_valid_auc = float(
        coarse_df.loc[coarse_df["search_id"].isin(selected_search_ids), "valid_auc"].iloc[-1]
    )
    return coarse_df, current_params, coarse_best_valid_auc


# 第二轮细搜围绕粗搜最优值做小步长调整，
# 只有验证集 AUC 不低于粗搜最优时才允许更新当前参数。
def run_fine_search(
    spec: ModelTrainingSpec,
    coarse_params: dict[str, object],
    coarse_best_valid_auc: float,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> tuple[pd.DataFrame, dict[str, object], str]:
    """执行第二轮细搜。"""
    current_params = coarse_params.copy()
    all_rows: list[dict[str, object]] = []
    selected_search_ids: list[int] = []
    search_id = 1
    has_any_update = False

    for param_spec in spec.search_params:
        step_rows: list[dict[str, object]] = []
        for candidate in param_spec.fine_builder(current_params[param_spec.name]):
            candidate_params = current_params.copy()
            candidate_params[param_spec.name] = candidate
            _, train_metrics, valid_metrics = fit_and_evaluate_on_holdout(
                spec=spec,
                params=candidate_params,
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                y_valid=y_valid,
            )
            row = build_search_row(
                spec=spec,
                stage="fine",
                search_id=search_id,
                tuned_parameter=param_spec.name,
                params=candidate_params,
                train_metrics=train_metrics,
                valid_metrics=valid_metrics,
            )
            all_rows.append(row)
            step_rows.append(row)
            search_id += 1

        ranked_step_df = rank_fine_step(pd.DataFrame(step_rows), coarse_best_valid_auc)
        qualified_step_df = ranked_step_df.loc[
            ranked_step_df["meet_fine_selection_requirements"]
        ].reset_index(drop=True)
        if qualified_step_df.empty:
            continue

        best_step_row = qualified_step_df.iloc[0]
        current_params = {candidate_spec.name: best_step_row[candidate_spec.name] for candidate_spec in spec.search_params}
        selected_search_ids.append(int(best_step_row["search_id"]))
        has_any_update = True

    fine_df = pd.DataFrame(all_rows)
    fine_df["meet_coarse_valid_auc_floor"] = fine_df["valid_auc"] >= coarse_best_valid_auc
    fine_df["meet_fine_selection_requirements"] = (
        fine_df["meet_coarse_valid_auc_floor"]
        & fine_df["meet_valid_auc_le_train_auc"]
    )
    fine_df["step_selected"] = fine_df["search_id"].isin(selected_search_ids)
    final_stage = "fine" if has_any_update else "coarse"
    return fine_df, current_params if has_any_update else coarse_params, final_stage


# 最终模型仍然严格只在训练集拟合，
# 再分别在 train/valid/test 三段输出最终结果。
def fit_final_model(
    spec: ModelTrainingSpec,
    final_params: dict[str, object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[object, pd.DataFrame]:
    """用最终参数拟合模型并输出三段评估。"""
    model = spec.build_model_fn(final_params)
    model.fit(X_train, y_train)
    metrics_df = pd.concat(
        [
            evaluate_model(model, X_train, y_train, "train"),
            evaluate_model(model, X_valid, y_valid, "valid"),
            evaluate_model(model, X_test, y_test, "test"),
        ],
        ignore_index=True,
    )
    return model, metrics_df


# 对支持 feature_importances_ 的模型统一输出重要性，
# 便于后续比较不同模型对同一批变量的偏好。
def build_feature_importance_frame(model, feature_names: list[str]) -> pd.DataFrame:
    """把模型特征重要性整理成表。"""
    if not hasattr(model, "feature_importances_"):
        raise AttributeError("当前模型不支持 feature_importances_。")

    importance_df = pd.DataFrame(
        {
            "feature_name": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values(["importance", "feature_name"], ascending=[False, True]).reset_index(drop=True)
    importance_df["importance_rank"] = importance_df.index + 1
    total_importance = float(importance_df["importance"].sum())
    if total_importance > 0:
        importance_df["importance_share"] = importance_df["importance"] / total_importance
    else:
        importance_df["importance_share"] = 0.0
    return importance_df


# 最优参数表保留参数值、含义、粗搜范围和最终三段指标，
# 方便统一核对模型复杂度是否过于激进。
def build_best_params_frame(
    spec: ModelTrainingSpec,
    final_params: dict[str, object],
    metrics_df: pd.DataFrame,
    coarse_best_valid_auc: float,
    final_stage: str,
) -> pd.DataFrame:
    """整理最终参数和选择依据。"""
    train_auc = metrics_df.loc[metrics_df["dataset"] == "train", "auc"].iloc[0]
    valid_auc = metrics_df.loc[metrics_df["dataset"] == "valid", "auc"].iloc[0]
    test_auc = metrics_df.loc[metrics_df["dataset"] == "test", "auc"].iloc[0]
    auc_gap = abs(train_auc - valid_auc)

    rows = []
    for param_spec in spec.search_params:
        rows.append(
            {
                "parameter": param_spec.name,
                "value": final_params[param_spec.name],
                "meaning": param_spec.meaning,
                "search_range": str(list(param_spec.coarse_candidates)),
                "trend_note": param_spec.trend_note,
            }
        )
    rows.extend(
        [
            {
                "parameter": "coarse_best_valid_auc",
                "value": coarse_best_valid_auc,
                "meaning": "第一轮粗搜锁定参数后的最高 valid AUC。",
                "search_range": "",
                "trend_note": "",
            },
            {
                "parameter": "final_selection_stage",
                "value": final_stage,
                "meaning": "最终参数来自 coarse 还是 fine。",
                "search_range": "",
                "trend_note": "",
            },
            {
                "parameter": "selection_rule",
                "value": SELECTION_RULE_TEXT,
                "meaning": "当前训练模型沿用的两轮逐参数搜索规则。",
                "search_range": "",
                "trend_note": "",
            },
            {
                "parameter": "final_train_auc",
                "value": train_auc,
                "meaning": "最终参数在训练集上的 AUC。",
                "search_range": "",
                "trend_note": "",
            },
            {
                "parameter": "final_valid_auc",
                "value": valid_auc,
                "meaning": "最终参数在验证集上的 AUC。",
                "search_range": "",
                "trend_note": "",
            },
            {
                "parameter": "final_test_auc",
                "value": test_auc,
                "meaning": "最终参数在测试集上的 AUC。",
                "search_range": "",
                "trend_note": "",
            },
            {
                "parameter": "final_train_valid_auc_gap",
                "value": auc_gap,
                "meaning": "最终训练集与验证集 AUC 的绝对差。",
                "search_range": "",
                "trend_note": "",
            },
        ]
    )
    return pd.DataFrame(rows)


# 把 coarse、fine、best_params、feature_importance 统一保存到模型结果目录。
def save_search_outputs(
    ctx: RuntimeContext,
    spec: ModelTrainingSpec,
    coarse_df: pd.DataFrame,
    fine_df: pd.DataFrame,
    best_params_df: pd.DataFrame,
    feature_importance_df: pd.DataFrame,
) -> None:
    """保存超参数搜索结果。"""
    coarse_save_df = coarse_df.drop(columns=SEARCH_LOG_HELPER_COLS, errors="ignore")
    fine_save_df = fine_df.drop(columns=SEARCH_LOG_HELPER_COLS, errors="ignore")
    coarse_save_df.to_csv(
        ctx.paths.output_file(spec.model_id, ctx.scheme_name, "coarse_search.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    fine_save_df.to_csv(
        ctx.paths.output_file(spec.model_id, ctx.scheme_name, "fine_search.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    best_params_df.to_csv(
        ctx.paths.output_file(spec.model_id, ctx.scheme_name, "best_params.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    feature_importance_df.to_csv(
        ctx.paths.output_file(spec.model_id, ctx.scheme_name, "feature_importance.csv"),
        index=False,
        encoding="utf-8-sig",
    )


# 三段评估和预测明细统一复用这一套输出逻辑，
# 确保四个模型的目录结构完全一致。
def evaluate_and_save(
    model,
    datasets: dict[str, dict[str, object]],
    feature_names: list[str],
    ctx: RuntimeContext,
    model_id: str,
) -> pd.DataFrame:
    """计算并保存三段评估结果与预测明细。"""
    metric_frames = []

    for split_name in ["train", "valid", "test"]:
        split_data = datasets[split_name]
        X = align_features(split_data["X"], feature_names)
        y = split_data["y"]
        dates = split_data["dates"]

        metric_frames.append(evaluate_model(model, X, y, split_name))
        prediction_df = build_prediction_frame(model, dates, X, y, split_name)
        prediction_df.to_csv(
            ctx.paths.output_file(model_id, ctx.scheme_name, f"pred_{split_name}.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    metrics_df = pd.concat(metric_frames, ignore_index=True)
    metrics_df.to_csv(
        ctx.paths.output_file(model_id, ctx.scheme_name, "metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    return metrics_df


# 统一的训练入口：读取建模数据、执行两轮搜索、保存模型、指标和重要性。
def train_model_pipeline(
    ctx: RuntimeContext,
    spec: ModelTrainingSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """执行训练、搜索和结果保存。"""
    cleanup_legacy_output_files(ctx, spec.model_id)
    datasets, feature_names = load_datasets(ctx)

    coarse_df, coarse_params, coarse_best_valid_auc = run_coarse_search(
        spec=spec,
        X_train=datasets["train"]["X"],
        y_train=datasets["train"]["y"],
        X_valid=datasets["valid"]["X"],
        y_valid=datasets["valid"]["y"],
    )
    fine_df, final_params, final_stage = run_fine_search(
        spec=spec,
        coarse_params=coarse_params,
        coarse_best_valid_auc=coarse_best_valid_auc,
        X_train=datasets["train"]["X"],
        y_train=datasets["train"]["y"],
        X_valid=datasets["valid"]["X"],
        y_valid=datasets["valid"]["y"],
    )
    final_model, metrics_df = fit_final_model(
        spec=spec,
        final_params=final_params,
        X_train=datasets["train"]["X"],
        y_train=datasets["train"]["y"],
        X_valid=datasets["valid"]["X"],
        y_valid=datasets["valid"]["y"],
        X_test=datasets["test"]["X"],
        y_test=datasets["test"]["y"],
    )
    best_params_df = build_best_params_frame(
        spec=spec,
        final_params=final_params,
        metrics_df=metrics_df,
        coarse_best_valid_auc=coarse_best_valid_auc,
        final_stage=final_stage,
    )
    feature_importance_df = build_feature_importance_frame(final_model, feature_names)

    ctx.paths.ensure_model_dirs(spec.model_id, ctx.scheme_name)
    save_model_bundle(
        model=final_model,
        model_file=ctx.paths.model_file(spec.model_id, ctx.scheme_name),
        model_id=spec.model_id,
        feature_names=feature_names,
    )
    save_search_outputs(
        ctx=ctx,
        spec=spec,
        coarse_df=coarse_df,
        fine_df=fine_df,
        best_params_df=best_params_df,
        feature_importance_df=feature_importance_df,
    )
    metrics_df = evaluate_and_save(
        model=final_model,
        datasets=datasets,
        feature_names=feature_names,
        ctx=ctx,
        model_id=spec.model_id,
    )
    return best_params_df, metrics_df


# 统一的重评估入口：只读本地模型，并重新输出 metrics 与 pred 明细。
def evaluate_saved_model_pipeline(
    ctx: RuntimeContext,
    spec: ModelTrainingSpec,
) -> pd.DataFrame:
    """读取本地模型并保存三段评估结果。"""
    datasets, _ = load_datasets(ctx)
    bundle = load_model_bundle(ctx.paths.model_file(spec.model_id, ctx.scheme_name))
    model = bundle["model"]
    feature_names = require_feature_names(bundle)
    ctx.paths.ensure_model_dirs(spec.model_id, ctx.scheme_name)
    return evaluate_and_save(
        model=model,
        datasets=datasets,
        feature_names=feature_names,
        ctx=ctx,
        model_id=spec.model_id,
    )

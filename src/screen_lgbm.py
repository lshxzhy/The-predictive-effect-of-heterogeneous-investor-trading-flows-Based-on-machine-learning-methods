import argparse

import pandas as pd
from lightgbm import LGBMClassifier

from config import get_horizon_context
from training_utils import evaluate_model


RANDOM_STATE = 42
N_JOBS = -1
METRIC_COLS = ["accuracy", "precision", "recall", "f1", "auc", "ks"]
SEARCH_PARAM_NAMES = [
    "n_estimators",
    "learning_rate",
    "num_leaves",
    "max_depth",
    "min_child_samples",
    "min_child_weight",
    "min_split_gain",
    "subsample",
    "subsample_freq",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
    "max_bin",
    "path_smooth",
    "extra_trees",
    "scale_pos_weight",
]
SEARCH_PARAM_ORDER = [
    "learning_rate",
    "n_estimators",
    "num_leaves",
    "max_depth",
    "min_child_samples",
    "min_child_weight",
    "min_split_gain",
    "subsample",
    "subsample_freq",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
    "max_bin",
    "path_smooth",
    "extra_trees",
    "scale_pos_weight",
]
INITIAL_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 80,
    "min_child_weight": 1.0,
    "min_split_gain": 0.1,
    "subsample": 0.7,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "max_bin": 255,
    "path_smooth": 0.5,
    "extra_trees": False,
    "scale_pos_weight": 1.0,
}
COARSE_CANDIDATES = {
    "n_estimators": [100, 200, 300, 500, 700, 900],
    "learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05, 0.08],
    "num_leaves": [4, 8, 12, 16, 24, 32, 48, 64],
    "max_depth": [2, 3, 4, 5, 6, 8, 10, -1],
    "min_child_samples": [10, 20, 40, 60, 80, 100, 150, 200],
    "min_child_weight": [0.001, 0.01, 0.1, 1.0, 5.0, 10.0],
    "min_split_gain": [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
    "subsample": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "subsample_freq": [0, 1, 2, 3, 5, 7],
    "colsample_bytree": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "reg_alpha": [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0],
    "reg_lambda": [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0],
    "max_bin": [31, 63, 127, 255, 511],
    "path_smooth": [0.0, 0.1, 0.5, 1.0, 2.0, 5.0],
    "extra_trees": [False, True],
    "scale_pos_weight": [0.8, 0.9, 1.0, 1.1, 1.2],
}


def cleanup_legacy_output_files(ctx) -> None:
    """清理已经废弃的旧筛选输出文件。"""
    legacy_files = [
        ctx.paths.screening_output_file("screening_rf_importance.csv"),
    ]
    for file_path in legacy_files:
        if file_path.exists():
            file_path.unlink()


def load_screening_long_panel(horizon: int) -> tuple[pd.DataFrame, list[str], str]:
    """读取统一筛选长面板并拆分特征列。"""
    ctx = get_horizon_context(horizon=horizon)
    screening_horizon = ctx.config.selected_feature_source_horizon()
    if horizon != screening_horizon:
        raise ValueError(
            f"统一变量筛选固定使用 {screening_horizon}d，请显式传入 --horizon {screening_horizon}"
        )

    df = pd.read_csv(ctx.paths.screening_long_panel_file(), parse_dates=["Date"])
    feature_cols = ctx.config.all_screening_supplementary_cols()
    target_col = ctx.config.target_label_col(screening_horizon)
    return (
        df.loc[:, [*ctx.config.screening_id_cols, "split", target_col, *feature_cols]].copy(),
        feature_cols,
        target_col,
    )


def split_to_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """拆分筛选模型的训练集、验证集和测试集。"""
    train_df = df.loc[df["split"] == "train"].copy()
    valid_df = df.loc[df["split"] == "valid"].copy()
    test_df = df.loc[df["split"] == "test"].copy()
    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError("统一筛选长面板缺少 train、valid 或 test 样本。")

    missing_counts = df.loc[:, feature_cols].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if not missing_counts.empty:
        raise ValueError(f"统一筛选长面板仍存在特征缺失：{missing_counts.to_dict()}")

    return (
        train_df.loc[:, feature_cols].copy(),
        train_df[target_col].astype(int).copy(),
        valid_df.loc[:, feature_cols].copy(),
        valid_df[target_col].astype(int).copy(),
        test_df.loc[:, feature_cols].copy(),
        test_df[target_col].astype(int).copy(),
    )


def build_model(params: dict) -> LGBMClassifier:
    """根据参数构造 LightGBM 筛选模型。"""
    return LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        importance_type="gain",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbosity=-1,
        **params,
    )


def fit_and_evaluate_on_holdout(
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> tuple[LGBMClassifier, pd.Series, pd.Series]:
    """拟合单组参数并返回训练集与验证集评估。"""
    model = build_model(params)
    model.fit(X_train, y_train)
    train_metrics = evaluate_model(model, X_train, y_train, "train").iloc[0]
    valid_metrics = evaluate_model(model, X_valid, y_valid, "valid").iloc[0]
    return model, train_metrics, valid_metrics


def build_search_row(
    stage: str,
    search_id: int,
    tuned_parameter: str,
    params: dict,
    train_metrics: pd.Series,
    valid_metrics: pd.Series,
) -> dict[str, object]:
    """整理单次搜索结果。"""
    row: dict[str, object] = {
        "stage": stage,
        "search_id": search_id,
        "tuned_parameter": tuned_parameter,
    }
    row.update({name: params[name] for name in SEARCH_PARAM_NAMES})

    for metric_col in METRIC_COLS:
        row[f"train_{metric_col}"] = train_metrics[metric_col]
        row[f"valid_{metric_col}"] = valid_metrics[metric_col]

    train_auc = train_metrics["auc"]
    valid_auc = valid_metrics["auc"]
    row["auc_gap"] = abs(train_auc - valid_auc) if pd.notna(train_auc) and pd.notna(valid_auc) else float("inf")
    return row


def rank_coarse_step(step_df: pd.DataFrame) -> pd.DataFrame:
    """按第一轮规则排序单参数粗搜结果。"""
    ranked_df = step_df.sort_values(
        ["valid_auc", "auc_gap", "train_auc", "search_id"],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)
    ranked_df["selection_rank"] = ranked_df.index + 1
    return ranked_df


def rank_fine_step(step_df: pd.DataFrame, coarse_best_valid_auc: float) -> pd.DataFrame:
    """按第二轮规则排序单参数细搜结果。"""
    ranked_df = step_df.copy()
    ranked_df["beat_coarse_valid_auc"] = ranked_df["valid_auc"] > coarse_best_valid_auc
    if ranked_df["beat_coarse_valid_auc"].any():
        ranked_df = ranked_df.sort_values(
            ["beat_coarse_valid_auc", "auc_gap", "valid_auc", "train_auc", "search_id"],
            ascending=[False, True, False, False, True],
        ).reset_index(drop=True)
    else:
        ranked_df = ranked_df.sort_values(
            ["valid_auc", "auc_gap", "train_auc", "search_id"],
            ascending=[False, True, False, True],
        ).reset_index(drop=True)
    ranked_df["selection_rank"] = ranked_df.index + 1
    return ranked_df


def build_int_candidates(center: int, step: int, min_value: int) -> list[int]:
    """围绕整数中心值构造细搜候选。"""
    candidates = [
        center - 3 * step,
        center - 2 * step,
        center - step,
        center,
        center + step,
        center + 2 * step,
        center + 3 * step,
    ]
    return sorted({max(min_value, int(candidate)) for candidate in candidates})


def build_float_candidates(
    center: float,
    step: float,
    min_value: float,
    max_value: float,
) -> list[float]:
    """围绕浮点中心值构造细搜候选。"""
    candidates = [
        center - 3 * step,
        center - 2 * step,
        center - step,
        center,
        center + step,
        center + 2 * step,
        center + 3 * step,
    ]
    rounded_candidates = []
    for candidate in candidates:
        clipped = min(max(candidate, min_value), max_value)
        rounded_candidates.append(round(float(clipped), 4))
    return sorted(set(rounded_candidates))


def build_max_depth_candidates(center: int) -> list[int]:
    """构造 max_depth 细搜候选。"""
    if center == -1:
        return [-1, 8, 10, 12, 14]
    return build_int_candidates(center=center, step=1, min_value=2)


def build_min_child_weight_candidates(center: float) -> list[float]:
    """构造 min_child_weight 细搜候选。"""
    factors = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
    candidates = []
    for factor in factors:
        candidate = max(0.0001, round(float(center) * factor, 4))
        candidates.append(candidate)
    return sorted(set(candidates))


def build_regularization_candidates(center: float) -> list[float]:
    """构造正则项细搜候选。"""
    step = 0.1 if center <= 1.0 else 0.25
    return build_float_candidates(center=center, step=step, min_value=0.0, max_value=10.0)


def build_fine_candidates(parameter_name: str, center_value: object) -> list[object]:
    """围绕粗搜最优值生成超小步长细搜候选。"""
    if parameter_name == "n_estimators":
        step = max(20, int(round(float(center_value) * 0.08)))
        return build_int_candidates(int(center_value), step=step, min_value=50)
    if parameter_name == "learning_rate":
        step = max(0.0025, float(center_value) * 0.08)
        return build_float_candidates(float(center_value), step=step, min_value=0.0025, max_value=0.2)
    if parameter_name == "num_leaves":
        step = max(2, int(round(float(center_value) * 0.12)))
        return build_int_candidates(int(center_value), step=step, min_value=2)
    if parameter_name == "max_depth":
        return build_max_depth_candidates(int(center_value))
    if parameter_name == "min_child_samples":
        step = max(5, int(round(float(center_value) * 0.08)))
        return build_int_candidates(int(center_value), step=step, min_value=5)
    if parameter_name == "min_child_weight":
        return build_min_child_weight_candidates(float(center_value))
    if parameter_name == "min_split_gain":
        step = max(0.02, float(center_value) * 0.2)
        return build_float_candidates(float(center_value), step=step, min_value=0.0, max_value=3.0)
    if parameter_name == "subsample":
        return build_float_candidates(float(center_value), step=0.03, min_value=0.4, max_value=1.0)
    if parameter_name == "subsample_freq":
        return build_int_candidates(int(center_value), step=1, min_value=0)
    if parameter_name == "colsample_bytree":
        return build_float_candidates(float(center_value), step=0.03, min_value=0.4, max_value=1.0)
    if parameter_name in {"reg_alpha", "reg_lambda"}:
        return build_regularization_candidates(float(center_value))
    if parameter_name == "max_bin":
        return build_int_candidates(int(center_value), step=32, min_value=31)
    if parameter_name == "path_smooth":
        step = max(0.1, float(center_value) * 0.25)
        return build_float_candidates(float(center_value), step=step, min_value=0.0, max_value=10.0)
    if parameter_name == "extra_trees":
        return [False, True]
    if parameter_name == "scale_pos_weight":
        return build_float_candidates(float(center_value), step=0.05, min_value=0.5, max_value=1.5)
    raise KeyError(f"未配置细搜参数：{parameter_name}")


def run_coarse_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> tuple[pd.DataFrame, dict[str, object], float]:
    """执行第一轮粗搜，并逐个参数锁定验证集 AUC 更高的值。"""
    current_params = INITIAL_PARAMS.copy()
    all_rows: list[dict[str, object]] = []
    selected_search_ids: list[int] = []
    search_id = 1

    for parameter_name in SEARCH_PARAM_ORDER:
        step_rows: list[dict[str, object]] = []
        for candidate in COARSE_CANDIDATES[parameter_name]:
            candidate_params = current_params.copy()
            candidate_params[parameter_name] = candidate
            _, train_metrics, valid_metrics = fit_and_evaluate_on_holdout(
                candidate_params,
                X_train,
                y_train,
                X_valid,
                y_valid,
            )
            row = build_search_row(
                stage="coarse",
                search_id=search_id,
                tuned_parameter=parameter_name,
                params=candidate_params,
                train_metrics=train_metrics,
                valid_metrics=valid_metrics,
            )
            all_rows.append(row)
            step_rows.append(row)
            search_id += 1

        ranked_step_df = rank_coarse_step(pd.DataFrame(step_rows))
        best_step_row = ranked_step_df.iloc[0]
        current_params = {name: best_step_row[name] for name in SEARCH_PARAM_NAMES}
        selected_search_ids.append(int(best_step_row["search_id"]))

    coarse_df = pd.DataFrame(all_rows)
    coarse_df["step_selected"] = coarse_df["search_id"].isin(selected_search_ids)
    coarse_best_valid_auc = float(coarse_df.loc[coarse_df["search_id"].isin(selected_search_ids), "valid_auc"].iloc[-1])
    return coarse_df, current_params, coarse_best_valid_auc


def run_fine_search(
    coarse_params: dict[str, object],
    coarse_best_valid_auc: float,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> tuple[pd.DataFrame, dict[str, object], str]:
    """执行第二轮细搜，并只接受验证集 AUC 超过第一轮最优的参数。"""
    current_params = coarse_params.copy()
    all_rows: list[dict[str, object]] = []
    selected_search_ids: list[int] = []
    search_id = 1
    has_any_update = False

    for parameter_name in SEARCH_PARAM_ORDER:
        step_rows: list[dict[str, object]] = []
        for candidate in build_fine_candidates(parameter_name, current_params[parameter_name]):
            candidate_params = current_params.copy()
            candidate_params[parameter_name] = candidate
            _, train_metrics, valid_metrics = fit_and_evaluate_on_holdout(
                candidate_params,
                X_train,
                y_train,
                X_valid,
                y_valid,
            )
            row = build_search_row(
                stage="fine",
                search_id=search_id,
                tuned_parameter=parameter_name,
                params=candidate_params,
                train_metrics=train_metrics,
                valid_metrics=valid_metrics,
            )
            row["coarse_best_valid_auc"] = coarse_best_valid_auc
            all_rows.append(row)
            step_rows.append(row)
            search_id += 1

        ranked_step_df = rank_fine_step(pd.DataFrame(step_rows), coarse_best_valid_auc)
        qualified_step_df = ranked_step_df.loc[ranked_step_df["beat_coarse_valid_auc"]].reset_index(drop=True)
        if qualified_step_df.empty:
            continue

        best_step_row = qualified_step_df.iloc[0]
        current_params = {name: best_step_row[name] for name in SEARCH_PARAM_NAMES}
        selected_search_ids.append(int(best_step_row["search_id"]))
        has_any_update = True

    fine_df = pd.DataFrame(all_rows)
    fine_df["beat_coarse_valid_auc"] = fine_df["valid_auc"] > coarse_best_valid_auc
    fine_df["step_selected"] = fine_df["search_id"].isin(selected_search_ids)
    final_stage = "fine" if has_any_update else "coarse"
    return fine_df, current_params if has_any_update else coarse_params, final_stage


def fit_final_model(
    final_params: dict[str, object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[LGBMClassifier, pd.DataFrame]:
    """用最终参数拟合模型并输出三段评估。"""
    model = build_model(final_params)
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


def build_importance_frame(model, feature_cols: list[str]) -> pd.DataFrame:
    """把模型特征重要性整理成表。"""
    if not hasattr(model, "feature_importances_"):
        raise AttributeError("当前筛选模型不支持 feature_importances_。")

    importance_df = pd.DataFrame(
        {
            "feature_name": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    importance_df["importance_rank"] = importance_df.index + 1
    return importance_df.loc[:, ["feature_name", "importance_rank", "importance"]]


def build_best_params_frame(
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

    rows = [{"parameter": name, "value": final_params[name]} for name in SEARCH_PARAM_NAMES]
    rows.extend(
        [
            {"parameter": "coarse_best_valid_auc", "value": coarse_best_valid_auc},
            {"parameter": "final_selection_stage", "value": final_stage},
            {
                "parameter": "selection_rule",
                "value": "第一轮逐个参数取 valid_auc 最高；第二轮只接受 valid_auc 超过第一轮最优的候选，再从中选 train_valid_auc_gap 最小的一组",
            },
            {"parameter": "final_train_auc", "value": train_auc},
            {"parameter": "final_valid_auc", "value": valid_auc},
            {"parameter": "final_test_auc", "value": test_auc},
            {"parameter": "final_train_valid_auc_gap", "value": auc_gap},
        ]
    )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    """执行 LightGBM 两轮筛选搜索并输出重要性与评估结果。"""
    args = parse_args()
    ctx = get_horizon_context(horizon=args.horizon)
    cleanup_legacy_output_files(ctx)

    panel_df, feature_cols, target_col = load_screening_long_panel(args.horizon)
    X_train, y_train, X_valid, y_valid, X_test, y_test = split_to_xy(
        panel_df,
        feature_cols,
        target_col,
    )

    coarse_df, coarse_params, coarse_best_valid_auc = run_coarse_search(
        X_train,
        y_train,
        X_valid,
        y_valid,
    )
    fine_df, final_params, final_stage = run_fine_search(
        coarse_params,
        coarse_best_valid_auc,
        X_train,
        y_train,
        X_valid,
        y_valid,
    )
    final_model, metrics_df = fit_final_model(
        final_params,
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
    )
    importance_df = build_importance_frame(final_model, feature_cols)
    best_params_df = build_best_params_frame(
        final_params,
        metrics_df,
        coarse_best_valid_auc=coarse_best_valid_auc,
        final_stage=final_stage,
    )

    ctx.paths.ensure_screening_output_dir()
    coarse_file = ctx.paths.screening_output_file("screening_lgbm_coarse_search.csv")
    fine_file = ctx.paths.screening_output_file("screening_lgbm_fine_search.csv")
    importance_file = ctx.paths.screening_output_file("screening_lgbm_importance.csv")
    metrics_file = ctx.paths.screening_output_file("screening_lgbm_metrics.csv")
    best_params_file = ctx.paths.screening_output_file("screening_lgbm_best_params.csv")

    coarse_df.to_csv(coarse_file, index=False, encoding="utf-8-sig")
    fine_df.to_csv(fine_file, index=False, encoding="utf-8-sig")
    importance_df.to_csv(importance_file, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(metrics_file, index=False, encoding="utf-8-sig")
    best_params_df.to_csv(best_params_file, index=False, encoding="utf-8-sig")

    print("统一筛选 LightGBM 输出已保存：")
    print(coarse_file)
    print(fine_file)
    print(importance_file)
    print(metrics_file)
    print(best_params_file)
    print("\n最终参数：")
    print(best_params_df)
    print("\n最终三段评估：")
    print(metrics_df)


if __name__ == "__main__":
    main()

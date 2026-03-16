import argparse

import pandas as pd
from xgboost import XGBClassifier

from config import RuntimeContext, get_runtime_context
from training_utils import (
    align_features,
    build_prediction_frame,
    evaluate_model,
    load_model_bundle,
    require_feature_names,
    run_holdout_search,
    save_model_bundle,
)


MODEL_ID = "xgb_cls"
RANDOM_STATE = 42
N_JOBS = -1


def cleanup_legacy_output_files(ctx: RuntimeContext) -> None:
    """清理模型目录下已经废弃的旧结果文件。"""
    legacy_files = [
        ctx.paths.output_file(MODEL_ID, ctx.scheme_name, "cv_results.csv"),
    ]
    for file_path in legacy_files:
        if file_path.exists():
            file_path.unlink()


def load_split_frame(ctx: RuntimeContext, split_name: str) -> pd.DataFrame:
    """读取单个数据分段。"""
    return pd.read_csv(ctx.paths.model_input_file(split_name), parse_dates=["Date"])


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


def build_param_grid() -> dict:
    """返回 XGBoost 参数网格。"""
    return {
        "n_estimators": [600],
        "learning_rate": [0.02],
        "max_depth": [5],
        "min_child_weight": [50],
        "subsample": [0.7],
        "colsample_bytree": [0.7],
        "gamma": [0.7],
        "reg_alpha": [1.0],
        "reg_lambda": [1.0],
    }


def build_model(params: dict) -> XGBClassifier:
    """根据参数构造 XGBoost 模型。"""
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        tree_method="hist",
        verbosity=0,
        **params,
    )


def save_search_outputs(
    ctx: RuntimeContext,
    best_params_df: pd.DataFrame,
    search_df: pd.DataFrame,
) -> None:
    """保存超参数搜索结果。"""
    best_params_df.to_csv(
        ctx.paths.output_file(MODEL_ID, ctx.scheme_name, "best_params.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    search_df.to_csv(
        ctx.paths.output_file(MODEL_ID, ctx.scheme_name, "search_results.csv"),
        index=False,
        encoding="utf-8-sig",
    )


def evaluate_and_save(
    model,
    datasets: dict[str, dict[str, object]],
    feature_names: list[str],
    ctx: RuntimeContext,
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
            ctx.paths.output_file(MODEL_ID, ctx.scheme_name, f"pred_{split_name}.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    metrics_df = pd.concat(metric_frames, ignore_index=True)
    metrics_df.to_csv(
        ctx.paths.output_file(MODEL_ID, ctx.scheme_name, "metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    return metrics_df


def train_model(ctx: RuntimeContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    """执行训练、搜索和结果保存。"""
    cleanup_legacy_output_files(ctx)
    datasets, feature_names = load_datasets(ctx)

    model, best_params_df, search_df = run_holdout_search(
        build_model_fn=build_model,
        param_grid=build_param_grid(),
        X_train=datasets["train"]["X"],
        y_train=datasets["train"]["y"],
        X_valid=datasets["valid"]["X"],
        y_valid=datasets["valid"]["y"],
        metric_name=ctx.config.search_metric,
    )

    ctx.paths.ensure_model_dirs(MODEL_ID, ctx.scheme_name)
    save_model_bundle(
        model=model,
        model_file=ctx.paths.model_file(MODEL_ID, ctx.scheme_name),
        model_id=MODEL_ID,
        feature_names=feature_names,
    )
    save_search_outputs(ctx, best_params_df, search_df)
    metrics_df = evaluate_and_save(model, datasets, feature_names, ctx)
    return best_params_df, metrics_df


def evaluate_saved_model(ctx: RuntimeContext) -> pd.DataFrame:
    """读取本地模型并保存三段评估结果。"""
    datasets, _ = load_datasets(ctx)
    bundle = load_model_bundle(ctx.paths.model_file(MODEL_ID, ctx.scheme_name))
    model = bundle["model"]
    feature_names = require_feature_names(bundle)
    ctx.paths.ensure_model_dirs(MODEL_ID, ctx.scheme_name)
    return evaluate_and_save(model, datasets, feature_names, ctx)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--train", action="store_true")
    mode_group.add_argument("--evaluate", action="store_true")
    return parser.parse_args()


def main() -> None:
    """执行训练、评估或训练后评估。"""
    args = parse_args()
    ctx = get_runtime_context(asset_alias=args.asset, horizon=args.horizon)

    if args.train:
        best_params_df, metrics_df = train_model(ctx)
        print("训练完成，已保存：")
        print(ctx.paths.model_file(MODEL_ID, ctx.scheme_name))
        print(ctx.paths.output_file(MODEL_ID, ctx.scheme_name, "best_params.csv"))
        print(ctx.paths.output_file(MODEL_ID, ctx.scheme_name, "search_results.csv"))
        print(ctx.paths.output_file(MODEL_ID, ctx.scheme_name, "metrics.csv"))
        print("\n最优超参数：")
        print(best_params_df)
        print("\n评估结果：")
        print(metrics_df)
    else:
        metrics_df = evaluate_saved_model(ctx)
        print("已重新计算评估结果：")
        print(ctx.paths.output_file(MODEL_ID, ctx.scheme_name, "metrics.csv"))
        print(metrics_df)


if __name__ == "__main__":
    main()

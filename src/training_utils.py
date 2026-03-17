from itertools import count
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import ParameterGrid


# KS 只依赖真实标签和正类概率，用于补充 AUC 之外的排序区分能力诊断。
def calc_ks(y_true: pd.Series, y_score: np.ndarray) -> float:
    """计算 KS 指标。"""
    if pd.Series(y_true).nunique() < 2:
        return np.nan
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float((tpr - fpr).max())


# 统一把分类模型在单个数据集上的核心指标收口成一行表，
# 这样筛选搜索、正式训练和可视化都能复用同一评估结构。
def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    dataset_name: str,
) -> pd.DataFrame:
    """在指定数据集上评估分类模型。"""
    y_score = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    auc_value = np.nan
    if pd.Series(y).nunique() >= 2:
        auc_value = roc_auc_score(y, y_score)

    metrics = {
        "dataset": dataset_name,
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "auc": auc_value,
        "ks": calc_ks(y, y_score),
    }
    return pd.DataFrame([metrics])


# 预测明细保留 Date、数据集分段、真实值、预测值和概率，
# 主要给后续可视化或误判诊断使用。
def build_prediction_frame(
    model,
    dates: pd.Series,
    X: pd.DataFrame,
    y: pd.Series,
    dataset_name: str,
) -> pd.DataFrame:
    """保存预测明细。"""
    y_score = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)
    return pd.DataFrame(
        {
            "Date": dates,
            "dataset": dataset_name,
            "y_true": y,
            "y_pred": y_pred,
            "y_score": y_score,
        }
    )


# 把搜索指标名映射到评估表里的实际列名，
# 并统一处理 NaN 指标，避免超参搜索时出现隐式比较错误。
def get_metric_score(metric_name: str, metrics_row: pd.Series) -> float:
    """从评估结果中提取搜索分数。"""
    metric_map = {
        "roc_auc": "auc",
        "auc": "auc",
        "f1": "f1",
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "ks": "ks",
    }
    if metric_name not in metric_map:
        raise ValueError(f"未支持的搜索指标：{metric_name}")

    score_col = metric_map[metric_name]
    if score_col not in metrics_row.index:
        raise KeyError(f"评估结果缺少指标列：{score_col}")

    score_value = metrics_row[score_col]
    if pd.isna(score_value):
        return float("-inf")
    return float(score_value)


# 这是通用 holdout 搜索器：在 train 上拟合、在 valid 上打分，
# 最终返回最佳模型、最佳参数表和完整搜索日志。
def run_holdout_search(
    build_model_fn,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    metric_name: str,
):
    """使用验证集执行超参数搜索。"""
    search_rows: list[dict[str, object]] = []
    best_model = None
    best_params = None
    best_score = float("-inf")

    for search_id, params in zip(count(1), ParameterGrid(param_grid)):
        model = build_model_fn(params)
        model.fit(X_train, y_train)

        train_metrics = evaluate_model(model, X_train, y_train, "train").iloc[0]
        valid_metrics = evaluate_model(model, X_valid, y_valid, "valid").iloc[0]

        row = {"search_id": search_id, **params}
        for metric_col in ["accuracy", "precision", "recall", "f1", "auc", "ks"]:
            row[f"train_{metric_col}"] = train_metrics[metric_col]
            row[f"valid_{metric_col}"] = valid_metrics[metric_col]
        search_rows.append(row)

        current_score = get_metric_score(metric_name, valid_metrics)
        if current_score > best_score:
            best_score = current_score
            best_params = params
            best_model = model

    search_df = pd.DataFrame(search_rows)
    sort_col = "valid_auc" if metric_name in {"roc_auc", "auc"} else f"valid_{metric_name}"
    if sort_col in search_df.columns:
        search_df = search_df.sort_values(sort_col, ascending=False).reset_index(drop=True)

    best_params_df = pd.DataFrame(
        {
            "parameter": list(best_params.keys()) + ["best_valid_score", "search_metric"],
            "value": list(best_params.values()) + [best_score, metric_name],
        }
    )
    return best_model, best_params_df, search_df


# 模型包必须显式带上训练时特征顺序，否则推理阶段无法保证列对齐。
def require_feature_names(bundle: dict) -> list[str]:
    """读取模型中保存的特征名。"""
    if "feature_names" not in bundle:
        raise KeyError("模型文件缺少 feature_names。")

    feature_names = bundle["feature_names"]
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError("模型文件中的 feature_names 为空或格式不正确。")

    return feature_names


# 统一把模型对象、模型标识和特征顺序一起打包保存，
# 避免后续只剩裸模型而不知道训练时的输入列。
def save_model_bundle(
    model,
    model_file: Path,
    model_id: str,
    feature_names: list[str],
) -> None:
    """保存模型对象和特征名。"""
    model_file.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model_id": model_id,
        "model": model,
        "feature_names": feature_names,
    }
    joblib.dump(bundle, model_file)


# 读取模型包时不做额外推断，缺字段由下游显式校验。
def load_model_bundle(model_file: Path):
    """读取模型对象。"""
    return joblib.load(model_file)


# 推理输入必须按训练时保存的特征顺序重排，
# 只要缺任何一列就直接报错，避免静默错列。
def align_features(X: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """按训练时的特征顺序对齐输入矩阵。"""
    missing_cols = [col for col in feature_names if col not in X.columns]
    if missing_cols:
        raise ValueError(f"缺少训练时使用的特征列：{missing_cols}")
    return X.loc[:, feature_names]

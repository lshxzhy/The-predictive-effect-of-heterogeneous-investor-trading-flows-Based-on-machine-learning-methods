from pathlib import Path
import argparse

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
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier


MODEL_ID = "dt_cls"
RANDOM_STATE = 42
CV_SPLITS = 5
GRID_SEARCH_SCORING = "roc_auc"
N_JOBS = -1


def load_data(processed_dir):
    """读取训练集和测试集，并拆分特征与标签。"""
    train_df = pd.read_csv(processed_dir / "train_model_input.csv", parse_dates=["Date"])
    test_df = pd.read_csv(processed_dir / "test_model_input.csv", parse_dates=["Date"])

    X_train = train_df.drop(columns=["Date", "y"])
    y_train = train_df["y"]
    X_test = test_df.drop(columns=["Date", "y"])
    y_test = test_df["y"]

    return X_train, y_train, X_test, y_test


def build_param_grid():
    """返回决策树参数网格。"""
    param_grid = {
        # criterion：节点分裂的评价标准。
        "criterion": ["entropy"],

        # max_depth：树的最大深度。
        "max_depth": [4, 5, 6],

        # min_samples_split：一个节点继续分裂所需的最小样本数。
        "min_samples_split": [15, 20, 30],

        # min_samples_leaf：每个叶节点至少保留的样本数。
        "min_samples_leaf": [15, 20, 25, 30],

        # max_features：每次分裂时可参与候选的特征数。
        "max_features": ["sqrt"],

        # class_weight：类别权重。
        "class_weight": [None],

        # ccp_alpha：后剪枝强度。
        "ccp_alpha": [0.003, 0.004, 0.005, 0.006, 0.008],

        # min_impurity_decrease：允许分裂所需达到的最小纯度提升。
        "min_impurity_decrease": [0.0005, 0.001, 0.002],
    }
    return param_grid


def get_paths(project_root):
    """构造模型文件和结果文件路径。"""
    model_dir = project_root / "models"
    out_dir = project_root / "outputs" / MODEL_ID

    model_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "model": model_dir / f"{MODEL_ID}.joblib",
        "best_params": out_dir / "best_params.csv",
        "cv_results": out_dir / "cv_results.csv",
        "metrics": out_dir / "metrics.csv",
    }
    return paths


def run_grid_search(X_train, y_train, param_grid):
    """使用时间序列交叉验证执行网格搜索。"""
    model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    cv = TimeSeriesSplit(n_splits=CV_SPLITS)

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=GRID_SEARCH_SCORING,
        cv=cv,
        n_jobs=N_JOBS,
        refit=True,
        verbose=1,
    )
    search.fit(X_train, y_train)

    best_params_df = pd.DataFrame(
        {
            "parameter": list(search.best_params_.keys()) + ["best_cv_score"],
            "value": list(search.best_params_.values()) + [search.best_score_],
        }
    )

    cv_results_df = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")

    return search.best_estimator_, best_params_df, cv_results_df


def calc_ks(y_true, y_score):
    """根据预测概率计算 KS 值。"""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float((tpr - fpr).max())


def find_best_threshold(y_true, y_score):
    """在训练集上按 KS 最大值选择分类阈值。"""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    ks_values = tpr - fpr
    best_idx = int(np.argmax(ks_values))

    threshold = float(thresholds[best_idx])
    ks_value = float(ks_values[best_idx])

    if not np.isfinite(threshold):
        threshold = 0.5

    return threshold, ks_value


def append_threshold_info(best_params_df, threshold, train_ks):
    """将阈值信息追加到最优参数表。"""
    extra_df = pd.DataFrame(
        {
            "parameter": ["decision_threshold", "train_ks_at_threshold"],
            "value": [threshold, train_ks],
        }
    )
    return pd.concat([best_params_df, extra_df], ignore_index=True)


def save_model_bundle(model, threshold, model_file):
    """保存模型和阈值。"""
    bundle = {
        "model_id": MODEL_ID,
        "threshold": threshold,
        "model": model,
    }
    joblib.dump(bundle, model_file)


def load_model_bundle(model_file):
    """读取模型和阈值。"""
    if not model_file.exists():
        raise FileNotFoundError(f"未找到模型文件：{model_file}")
    return joblib.load(model_file)


def save_training_outputs(best_params_df, cv_results_df, paths):
    """保存最优参数表和交叉验证结果。"""
    best_params_df.to_csv(paths["best_params"], index=False, encoding="utf-8-sig")
    cv_results_df.to_csv(paths["cv_results"], index=False, encoding="utf-8-sig")


def evaluate(model, threshold, X, y, dataset_name):
    """计算指定数据集上的评估指标。"""
    y_score = model.predict_proba(X)[:, 1]
    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "dataset": dataset_name,
        "threshold": threshold,
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "auc": roc_auc_score(y, y_score),
        "ks": calc_ks(y, y_score),
    }

    return pd.DataFrame([metrics])


def evaluate_saved_model(model_file, X_train, y_train, X_test, y_test):
    """读取本地模型并评估训练集和测试集。"""
    bundle = load_model_bundle(model_file)
    model = bundle["model"]
    threshold = bundle["threshold"]

    train_metrics = evaluate(model, threshold, X_train, y_train, "train")
    test_metrics = evaluate(model, threshold, X_test, y_test, "test")

    return pd.concat([train_metrics, test_metrics], ignore_index=True)


def save_metrics(metrics_df, metrics_file):
    """保存评估结果表。"""
    metrics_df.to_csv(metrics_file, index=False, encoding="utf-8-sig")


def train(processed_dir, project_root):
    """执行训练、搜索、阈值选择和模型保存。"""
    X_train, y_train, _, _ = load_data(processed_dir)
    paths = get_paths(project_root)
    param_grid = build_param_grid()

    model, best_params_df, cv_results_df = run_grid_search(X_train, y_train, param_grid)

    train_score = model.predict_proba(X_train)[:, 1]
    threshold, train_ks = find_best_threshold(y_train, train_score)
    best_params_df = append_threshold_info(best_params_df, threshold, train_ks)

    save_model_bundle(model, threshold, paths["model"])
    save_training_outputs(best_params_df, cv_results_df, paths)

    return paths, best_params_df


def evaluate_model(processed_dir, project_root):
    """读取本地模型并保存训练集和测试集评估结果。"""
    X_train, y_train, X_test, y_test = load_data(processed_dir)
    paths = get_paths(project_root)

    metrics_df = evaluate_saved_model(paths["model"], X_train, y_train, X_test, y_test)
    save_metrics(metrics_df, paths["metrics"])

    return metrics_df, paths


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    return parser.parse_args()


def main():
    """执行训练、评估或训练后评估。"""
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"

    run_train = args.train or (not args.train and not args.evaluate)
    run_evaluate = args.evaluate or (not args.train and not args.evaluate)

    if run_train:
        paths, best_params_df = train(processed_dir, project_root)
        print("训练完成，文件已保存：")
        print(paths["model"])
        print(paths["best_params"])
        print(paths["cv_results"])
        print("\n最优超参数：")
        print(best_params_df)

    if run_evaluate:
        metrics_df, paths = evaluate_model(processed_dir, project_root)
        print("\n训练集和测试集评估结果：")
        print(metrics_df)
        print("\n评估结果文件：")
        print(paths["metrics"])


if __name__ == "__main__":
    main()

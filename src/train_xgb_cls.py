from pathlib import Path
import argparse

import joblib
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
from xgboost import XGBClassifier


MODEL_ID = "xgb_cls"
RANDOM_STATE = 42
CV_SPLITS = 5
GRID_SEARCH_SCORING = "roc_auc"
N_JOBS = -1


def load_data(processed_dir):
    """读取训练集和测试集，并拆分特征与标签。"""
    train_df = pd.read_csv(processed_dir / "train_model_input.csv", parse_dates=["Date"])
    test_df = pd.read_csv(processed_dir / "test_model_input.csv", parse_dates=["Date"])

    X_train = train_df.drop(columns=["Date", "y", "qzenergy_garch_lag1"])
    y_train = train_df["y"].astype(int)
    X_test = test_df.drop(columns=["Date", "y", "qzenergy_garch_lag1"])
    y_test = test_df["y"].astype(int)

    return X_train, y_train, X_test, y_test


def build_param_grid():
    """返回 XGBoost 参数网格。"""
    param_grid = {
        # n_estimators：树的数量。
        # 取值范围：正整数，常见 100 到 1000。
        # 变化趋势：越大通常越强，但训练更慢；如果学习率不变，过大也更容易过拟合。
        "n_estimators": [600],

        # learning_rate：每棵树的学习步长。
        # 取值范围：0 到 1 之间，常见 0.01 到 0.3。
        # 变化趋势：越小学习越慢、更稳；越大收敛更快，但更容易学得过头。
        "learning_rate": [0.02],

        # max_depth：每棵树的最大深度。
        # 取值范围：正整数，常见 2 到 10。
        # 变化趋势：越大越容易拟合复杂关系，但过拟合风险更高。
        "max_depth": [5],

        # min_child_weight：子节点所需的最小样本权重和。
        # 取值范围：大于等于 0，常见 1 到 20。
        # 变化趋势：越大越保守，越不容易继续分裂，可抑制过拟合。
        "min_child_weight": [50],

        # subsample：每棵树使用的样本比例。
        # 取值范围：0 到 1 之间，常见 0.5 到 1。
        # 变化趋势：越小随机性越强，通常更能抑制过拟合；过小可能欠拟合。
        "subsample": [0.7],

        # colsample_bytree：每棵树建树时抽取的特征比例。
        # 取值范围：0 到 1 之间，常见 0.5 到 1。
        # 变化趋势：越小越保守，有助于降低过拟合；过小可能损失信息。
        "colsample_bytree": [0.7],

        # gamma：节点继续分裂所需的最小损失下降值。
        # 取值范围：大于等于 0，常见 0 到 5。
        # 变化趋势：越大越难分裂，模型越简单，越能限制过拟合。
        "gamma": [0.7],

        # reg_alpha：L1 正则化强度。
        # 取值范围：大于等于 0，常见 0 到 10。
        # 变化趋势：越大越容易压缩不重要特征的作用，模型更稀疏、更保守。
        "reg_alpha": [1],

        # reg_lambda：L2 正则化强度。
        # 取值范围：大于等于 0，常见 1 到 10。
        # 变化趋势：越大约束越强，通常更稳，但过大可能欠拟合。
        "reg_lambda": [1],
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
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        tree_method="hist",
        verbosity=0,
    )
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


def save_model_bundle(model, model_file):
    """保存模型对象。"""
    bundle = {
        "model_id": MODEL_ID,
        "model": model,
    }
    joblib.dump(bundle, model_file)


def load_model_bundle(model_file):
    """读取模型对象。"""
    if not model_file.exists():
        raise FileNotFoundError(f"未找到模型文件：{model_file}")
    return joblib.load(model_file)


def save_training_outputs(best_params_df, cv_results_df, paths):
    """保存最优参数表和交叉验证结果。"""
    best_params_df.to_csv(paths["best_params"], index=False, encoding="utf-8-sig")
    cv_results_df.to_csv(paths["cv_results"], index=False, encoding="utf-8-sig")


def evaluate(model, X, y, dataset_name):
    """计算指定数据集上的评估指标。"""
    y_score = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    metrics = {
        "dataset": dataset_name,
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

    train_metrics = evaluate(model, X_train, y_train, "train")
    test_metrics = evaluate(model, X_test, y_test, "test")

    return pd.concat([train_metrics, test_metrics], ignore_index=True)


def save_metrics(metrics_df, metrics_file):
    """保存评估结果表。"""
    metrics_df.to_csv(metrics_file, index=False, encoding="utf-8-sig")


def train(processed_dir, project_root):
    """执行训练、搜索和模型保存。"""
    X_train, y_train, _, _ = load_data(processed_dir)
    paths = get_paths(project_root)
    param_grid = build_param_grid()

    model, best_params_df, cv_results_df = run_grid_search(X_train, y_train, param_grid)

    save_model_bundle(model, paths["model"])
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

from pathlib import Path
import argparse

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


MODEL_ID = "rf_cls"
RANDOM_STATE = 42
CV_SPLITS = 5
GRID_SEARCH_SCORING = "roc_auc"
N_JOBS = -1


def load_data(processed_dir):
    """读取训练集和测试集，并拆分特征与标签。"""
    train_df = pd.read_csv(processed_dir / "train_model_input.csv", parse_dates=["Date"])
    test_df = pd.read_csv(processed_dir / "test_model_input.csv", parse_dates=["Date"])

    X_train = train_df.drop(columns=["Date", "y", "qzenergy_garch_lag1"])
    y_train = train_df["y"]
    X_test = test_df.drop(columns=["Date", "y", "qzenergy_garch_lag1"])
    y_test = test_df["y"]

    return X_train, y_train, X_test, y_test


def build_param_grid():
    """返回完整参数版随机森林参数网格。"""
    param_grid = {
        # n_estimators：森林中树的数量。
        # 取值范围：正整数，常见 100 到 1000。
        # 变化趋势：越大通常越稳定、方差越低，但训练更慢；它主要提升稳定性，不是直接让模型更简单。
        "n_estimators": [500],

        # criterion：节点分裂的评价标准。
        # 取值范围："gini"、"entropy"、"log_loss"。
        # 变化趋势：不同准则会改变分裂偏好，通常不是控制过拟合的第一参数。
        "criterion": ["entropy"],

        # max_depth：每棵树的最大深度。
        # 取值范围：正整数或 None；None 表示尽量继续分裂。
        # 变化趋势：越小越保守，越大越容易过拟合。
        "max_depth": [3],

        # min_samples_split：一个节点继续分裂所需的最小样本数。
        # 取值范围：整数 >= 2，或 (0, 1] 的比例；这里使用整数。
        # 变化趋势：越大越不容易继续分裂，模型越保守，越能抑制过拟合。
        "min_samples_split": [200],

        # min_samples_leaf：每个叶节点至少保留的样本数。
        # 取值范围：整数 >= 1，或 (0, 1] 的比例；这里使用整数。
        # 变化趋势：越大叶子越大、树越平滑，过拟合通常越弱，但过大可能欠拟合。
        "min_samples_leaf": [49],


        # max_features：每次分裂时可参与候选的特征数。
        # 取值范围：整数、浮点比例、"sqrt"、"log2" 或 None。
        # 变化趋势：越小，单棵树相关性通常越低、过拟合通常更弱；越大，单棵树更强但更容易学得太细。
        "max_features": [0.7],

        # min_impurity_decrease：允许分裂所需达到的最小纯度提升。
        # 取值范围：大于等于 0 的浮点数，常见 0 到 0.01。
        # 变化趋势：越大越难分裂，树越简单，越能限制过拟合。
        "min_impurity_decrease": [0.008],

        # class_weight：类别权重。
        # 取值范围：None、"balanced"、"balanced_subsample" 或自定义字典。
        # 变化趋势：主要影响对少数类的重视程度，不是专门控制过拟合，但会改变 precision / recall 的平衡。
        "class_weight": [None],

        # ccp_alpha：代价复杂度剪枝强度。
        # 取值范围：大于等于 0 的浮点数，常见 0 到 0.01。
        # 变化趋势：越大剪枝越强，树越简单，越能抑制过拟合；过大可能明显欠拟合。
        "ccp_alpha": [0.015],

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
    model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS)
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

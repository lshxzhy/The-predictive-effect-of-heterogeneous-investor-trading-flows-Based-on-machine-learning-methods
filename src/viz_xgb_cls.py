from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd


MODEL_ID = "xgb_cls"
EXCLUDE_MODEL_COLS = {"Date", "y", "qzenergy_garch_lag1"}


def configure_matplotlib():
    """设置绘图参数。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False


def get_paths(project_root):
    """构造模型文件和可视化结果路径。"""
    processed_dir = project_root / "data" / "processed"
    model_file = project_root / "models" / f"{MODEL_ID}.joblib"
    out_dir = project_root / "outputs" / MODEL_ID
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "processed_dir": processed_dir,
        "model": model_file,
        "importance_png": out_dir / "feature_importance.png",
        "importance_csv": out_dir / "feature_importance.csv",
    }
    return paths


def load_bundle_and_features(paths):
    """读取模型包并提取与训练一致的特征名。"""
    if not paths["model"].exists():
        raise FileNotFoundError(f"未找到模型文件：{paths['model']}")

    bundle = joblib.load(paths["model"])
    model = bundle["model"]

    train_df = pd.read_csv(
        paths["processed_dir"] / "train_model_input.csv",
        parse_dates=["Date"],
    )
    feature_names = [col for col in train_df.columns if col not in EXCLUDE_MODEL_COLS]

    if hasattr(model, "n_features_in_") and len(feature_names) != model.n_features_in_:
        raise ValueError(
            f"特征数量不一致：模型需要 {model.n_features_in_} 个特征，"
            f"当前可视化读取到 {len(feature_names)} 个特征。"
            "请确认训练脚本与可视化脚本的删列逻辑一致，并重新训练模型。"
        )

    return model, feature_names


def build_feature_importance(model, feature_names):
    """整理特征重要性结果。"""
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return importance_df


def save_feature_importance(importance_df, output_file):
    """保存特征重要性表。"""
    importance_df.to_csv(output_file, index=False, encoding="utf-8-sig")


def plot_feature_importance(importance_df, output_file):
    """绘制特征重要性图。"""
    plot_df = importance_df.sort_values("importance", ascending=True).tail(20)

    plt.figure(figsize=(10, 8))
    plt.barh(plot_df["feature"], plot_df["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    """读取本地模型并输出可视化结果。"""
    configure_matplotlib()

    project_root = Path(__file__).resolve().parent.parent
    paths = get_paths(project_root)

    model, feature_names = load_bundle_and_features(paths)
    importance_df = build_feature_importance(model, feature_names)

    save_feature_importance(importance_df, paths["importance_csv"])
    plot_feature_importance(importance_df, paths["importance_png"])

    print("可视化文件已保存：")
    print(paths["importance_png"])
    print(paths["importance_csv"])


if __name__ == "__main__":
    main()

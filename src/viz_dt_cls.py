from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import export_text, plot_tree


MODEL_ID = "dt_cls"


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
        "tree_png": out_dir / "tree_structure.png",
        "importance_png": out_dir / "feature_importance.png",
        "importance_csv": out_dir / "feature_importance.csv",
        "rules_txt": out_dir / "tree_rules.txt",
    }
    return paths


def load_bundle_and_features(paths):
    """读取模型包并提取特征名。"""
    if not paths["model"].exists():
        raise FileNotFoundError(f"未找到模型文件：{paths['model']}")

    bundle = joblib.load(paths["model"])
    model = bundle["model"]

    train_df = pd.read_csv(
        paths["processed_dir"] / "train_model_input.csv",
        parse_dates=["Date"],
    )
    feature_names = [col for col in train_df.columns if col not in ["Date", "y"]]

    return bundle, model, feature_names


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
    plot_df = importance_df.sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["feature"], plot_df["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Decision Tree Feature Importance")
    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close()


def plot_tree_structure(model, feature_names, output_file):
    """绘制决策树结构图。"""
    plt.figure(figsize=(24, 12))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=["0", "1"],
        filled=True,
        rounded=True,
        impurity=True,
        proportion=True,
        precision=3,
        fontsize=9,
    )
    plt.title("Decision Tree Structure")
    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close()


def save_tree_rules(model, feature_names, output_file):
    """导出决策树文本规则。"""
    rules = export_text(
        model,
        feature_names=list(feature_names),
        decimals=4,
    )
    output_file.write_text(rules, encoding="utf-8")


def main():
    """读取本地模型并输出可视化结果。"""
    configure_matplotlib()

    project_root = Path(__file__).resolve().parent.parent
    paths = get_paths(project_root)

    bundle, model, feature_names = load_bundle_and_features(paths)
    importance_df = build_feature_importance(model, feature_names)

    save_feature_importance(importance_df, paths["importance_csv"])
    plot_feature_importance(importance_df, paths["importance_png"])
    plot_tree_structure(model, feature_names, paths["tree_png"])
    save_tree_rules(model, feature_names, paths["rules_txt"])

    print("使用的分类阈值：")
    print(bundle["threshold"])
    print("\n可视化文件已保存：")
    print(paths["tree_png"])
    print(paths["importance_png"])
    print(paths["importance_csv"])
    print(paths["rules_txt"])


if __name__ == "__main__":
    main()

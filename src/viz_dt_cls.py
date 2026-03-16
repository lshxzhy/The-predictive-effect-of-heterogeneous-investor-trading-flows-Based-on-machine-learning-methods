import argparse

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import export_text, plot_tree

from config import RuntimeContext, get_runtime_context
from training_utils import load_model_bundle, require_feature_names


MODEL_ID = "dt_cls"


def configure_matplotlib() -> None:
    """设置绘图参数。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False


def load_model_and_features(ctx: RuntimeContext):
    """读取模型和训练时特征名。"""
    bundle = load_model_bundle(ctx.paths.model_file(MODEL_ID, ctx.scheme_name))
    model = bundle["model"]
    feature_names = require_feature_names(bundle)
    return model, feature_names


def build_feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    """整理特征重要性结果。"""
    return pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)


def plot_feature_importance(importance_df: pd.DataFrame, output_file) -> None:
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


def plot_tree_structure(model, feature_names: list[str], output_file) -> None:
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


def save_tree_rules(model, feature_names: list[str], output_file) -> None:
    """导出树规则文本。"""
    rules = export_text(model, feature_names=list(feature_names), decimals=4)
    output_file.write_text(rules, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    """输出决策树可视化结果。"""
    args = parse_args()
    configure_matplotlib()
    ctx = get_runtime_context(asset_alias=args.asset, horizon=args.horizon)

    model, feature_names = load_model_and_features(ctx)
    importance_df = build_feature_importance(model, feature_names)
    ctx.paths.ensure_model_dirs(MODEL_ID, ctx.scheme_name)

    importance_csv = ctx.paths.output_file(MODEL_ID, ctx.scheme_name, "feature_importance.csv")
    importance_png = ctx.paths.output_file(MODEL_ID, ctx.scheme_name, "feature_importance.png")
    tree_png = ctx.paths.output_file(MODEL_ID, ctx.scheme_name, "tree_structure.png")
    rules_txt = ctx.paths.output_file(MODEL_ID, ctx.scheme_name, "tree_rules.txt")

    importance_df.to_csv(importance_csv, index=False, encoding="utf-8-sig")
    plot_feature_importance(importance_df, importance_png)
    plot_tree_structure(model, feature_names, tree_png)
    save_tree_rules(model, feature_names, rules_txt)

    print("可视化文件已保存：")
    print(tree_png)
    print(importance_png)
    print(importance_csv)
    print(rules_txt)


if __name__ == "__main__":
    main()

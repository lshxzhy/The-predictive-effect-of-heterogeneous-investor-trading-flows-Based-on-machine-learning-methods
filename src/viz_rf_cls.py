import argparse

import matplotlib.pyplot as plt
import pandas as pd

from config import RuntimeContext, get_runtime_context
from training_utils import load_model_bundle, require_feature_names


MODEL_ID = "rf_cls"


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
    plot_df = importance_df.sort_values("importance", ascending=True).tail(20)

    plt.figure(figsize=(10, 8))
    plt.barh(plot_df["feature"], plot_df["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    """输出随机森林可视化结果。"""
    args = parse_args()
    configure_matplotlib()
    ctx = get_runtime_context(asset_alias=args.asset, horizon=args.horizon)

    model, feature_names = load_model_and_features(ctx)
    importance_df = build_feature_importance(model, feature_names)
    ctx.paths.ensure_model_dirs(MODEL_ID, ctx.scheme_name)

    importance_csv = ctx.paths.output_file(MODEL_ID, ctx.scheme_name, "feature_importance.csv")
    importance_png = ctx.paths.output_file(MODEL_ID, ctx.scheme_name, "feature_importance.png")

    importance_df.to_csv(importance_csv, index=False, encoding="utf-8-sig")
    plot_feature_importance(importance_df, importance_png)

    print("可视化文件已保存：")
    print(importance_png)
    print(importance_csv)


if __name__ == "__main__":
    main()

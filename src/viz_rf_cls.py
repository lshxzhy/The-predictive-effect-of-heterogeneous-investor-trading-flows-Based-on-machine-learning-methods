import argparse

import matplotlib.pyplot as plt
import pandas as pd

from config import RuntimeContext, get_runtime_context
from plot_utils import MONO_BAR_COLOR, MONO_EDGE_COLOR, configure_monochrome_matplotlib
from training_utils import load_model_bundle, require_feature_names


MODEL_ID = "rf_cls"


# 可视化脚本只读取已经训练完成的随机森林模型及其特征名。
def load_model_and_features(ctx: RuntimeContext):
    """读取模型和训练时特征名。"""
    bundle = load_model_bundle(ctx.paths.model_file(MODEL_ID, ctx.scheme_name))
    model = bundle["model"]
    feature_names = require_feature_names(bundle)
    return model, feature_names


# 随机森林重要性基于各树分裂贡献聚合后的 feature_importances_。
def build_feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    """整理特征重要性结果。"""
    return pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)


# 随机森林重要性图固定展示前 20 个变量，并统一改成黑白灰条形图。
def plot_feature_importance(importance_df: pd.DataFrame, output_file) -> None:
    """绘制特征重要性图。"""
    plot_df = importance_df.sort_values("importance", ascending=True).tail(20)

    plt.figure(figsize=(10, 8))
    plt.barh(
        plot_df["feature"],
        plot_df["importance"],
        color=MONO_BAR_COLOR,
        edgecolor=MONO_EDGE_COLOR,
    )
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close()


# 可视化脚本依旧要求显式传入 asset 和 horizon。
def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    return parser.parse_args()


# 统一输出随机森林的重要性表和重要性图。
def main() -> None:
    """输出随机森林可视化结果。"""
    args = parse_args()
    configure_monochrome_matplotlib()
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

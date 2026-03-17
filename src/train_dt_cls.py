import argparse

from sklearn.tree import DecisionTreeClassifier

from config import RuntimeContext, get_runtime_context
from model_train_utils import (
    ModelTrainingSpec,
    SearchParameterSpec,
    build_choice_candidates,
    build_float_candidates,
    build_int_candidates,
    evaluate_saved_model_pipeline,
    train_model_pipeline,
)


MODEL_ID = "dt_cls"
RANDOM_STATE = 42


# 决策树的参数搜索重点放在树深、叶子样本量、剪枝和分裂阈值，
# 这些都是最直接的过拟合控制旋钮。
def build_training_spec() -> ModelTrainingSpec:
    """构造决策树训练配置。"""
    search_params = (
        SearchParameterSpec(
            name="criterion",
            meaning="分裂准则。",
            trend_note="更换准则会影响分裂偏好，但不会直接改变树深约束。",
            coarse_candidates=("gini", "entropy"),
            fine_builder=lambda center: build_choice_candidates(center, ("gini", "entropy")),
        ),
        SearchParameterSpec(
            name="max_depth",
            meaning="树的最大深度。",
            trend_note="增大通常提升拟合能力，也更容易过拟合；减小会让树更保守。",
            coarse_candidates=(2, 3, 4, 5, 6, 8),
            fine_builder=lambda center: build_int_candidates(int(center), step=1, min_value=2, max_value=10),
        ),
        SearchParameterSpec(
            name="min_samples_leaf",
            meaning="叶子节点最少样本数。",
            trend_note="增大能明显抑制碎叶子和噪声分裂；减小会提升局部拟合能力。",
            coarse_candidates=(10, 20, 30, 40, 60, 80),
            fine_builder=lambda center: build_int_candidates(int(center), step=10, min_value=5),
        ),
        SearchParameterSpec(
            name="min_samples_split",
            meaning="内部节点继续分裂所需的最少样本数。",
            trend_note="增大时分裂更谨慎；减小时更容易继续向下展开。",
            coarse_candidates=(20, 40, 60, 80, 120, 160),
            fine_builder=lambda center: build_int_candidates(int(center), step=20, min_value=10),
        ),
        SearchParameterSpec(
            name="max_features",
            meaning="单次分裂时可见的特征比例。",
            trend_note="减小会增加随机性并抑制共线噪声；过小会削弱有效信号。",
            coarse_candidates=(0.4, 0.5, 0.6, 0.7, 0.8, 1.0),
            fine_builder=lambda center: build_float_candidates(float(center), step=0.05, min_value=0.3, max_value=1.0),
        ),
        SearchParameterSpec(
            name="ccp_alpha",
            meaning="代价复杂度剪枝强度。",
            trend_note="增大时树会被更强地剪枝；过大可能导致欠拟合。",
            coarse_candidates=(0.0, 0.0005, 0.001, 0.002, 0.005, 0.01),
            fine_builder=lambda center: build_float_candidates(float(center), step=0.0005, min_value=0.0, max_value=0.02),
        ),
        SearchParameterSpec(
            name="min_impurity_decrease",
            meaning="继续分裂所需的最小纯度提升。",
            trend_note="增大时树更难继续分裂；减小时更容易拟合细节。",
            coarse_candidates=(0.0, 0.0002, 0.0005, 0.001, 0.002, 0.005),
            fine_builder=lambda center: build_float_candidates(float(center), step=0.0002, min_value=0.0, max_value=0.01),
        ),
        SearchParameterSpec(
            name="class_weight",
            meaning="类别权重方案。",
            trend_note="balanced 会提高少数类权重，但也可能牺牲总体稳定性。",
            coarse_candidates=(None, "balanced"),
            fine_builder=lambda center: build_choice_candidates(center, (None, "balanced")),
        ),
    )
    initial_params = {
        "criterion": "entropy",
        "max_depth": 4,
        "min_samples_leaf": 30,
        "min_samples_split": 80,
        "max_features": 0.6,
        "ccp_alpha": 0.001,
        "min_impurity_decrease": 0.0005,
        "class_weight": None,
    }
    return ModelTrainingSpec(
        model_id=MODEL_ID,
        display_name="DecisionTreeClassifier",
        initial_params=initial_params,
        search_params=search_params,
        build_model_fn=lambda params: DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            **params,
        ),
    )


def train_model(ctx: RuntimeContext):
    """执行决策树训练与搜索。"""
    return train_model_pipeline(ctx, build_training_spec())


def evaluate_saved_model(ctx: RuntimeContext):
    """读取本地决策树模型并重新输出评估结果。"""
    return evaluate_saved_model_pipeline(ctx, build_training_spec())


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
    """执行训练或重评估。"""
    args = parse_args()
    ctx = get_runtime_context(asset_alias=args.asset, horizon=args.horizon)

    if args.train:
        best_params_df, metrics_df = train_model(ctx)
        print("决策树训练完成，已保存：")
        print(ctx.paths.model_file(MODEL_ID, ctx.scheme_name))
        print(ctx.paths.output_file(MODEL_ID, ctx.scheme_name, "coarse_search.csv"))
        print(ctx.paths.output_file(MODEL_ID, ctx.scheme_name, "fine_search.csv"))
        print(ctx.paths.output_file(MODEL_ID, ctx.scheme_name, "best_params.csv"))
        print(ctx.paths.output_file(MODEL_ID, ctx.scheme_name, "feature_importance.csv"))
        print(ctx.paths.output_file(MODEL_ID, ctx.scheme_name, "metrics.csv"))
        print("\n最优超参数：")
        print(best_params_df)
        print("\n评估结果：")
        print(metrics_df)
    else:
        metrics_df = evaluate_saved_model(ctx)
        print("已重新计算决策树评估结果：")
        print(ctx.paths.output_file(MODEL_ID, ctx.scheme_name, "metrics.csv"))
        print(metrics_df)


if __name__ == "__main__":
    main()

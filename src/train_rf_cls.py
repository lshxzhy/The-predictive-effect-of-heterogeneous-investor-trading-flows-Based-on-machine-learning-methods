import argparse

from sklearn.ensemble import RandomForestClassifier

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


MODEL_ID = "rf_cls"
RANDOM_STATE = 42
N_JOBS = -1


# 随机森林在保持集成稳定性的同时，重点搜索树复杂度、抽样比例和剪枝强度，
# 保守优先，避免在单资产小样本上把树堆得过深。
def build_training_spec() -> ModelTrainingSpec:
    """构造随机森林训练配置。"""
    search_params = (
        SearchParameterSpec(
            name="criterion",
            meaning="单棵树的分裂准则。",
            trend_note="更换准则会改变分裂偏好，但不直接决定森林复杂度。",
            coarse_candidates=("gini", "entropy"),
            fine_builder=lambda center: build_choice_candidates(center, ("gini", "entropy")),
        ),
        SearchParameterSpec(
            name="max_depth",
            meaning="单棵树最大深度。",
            trend_note="增大时单棵树更复杂；减小时森林整体更稳健。",
            coarse_candidates=(2, 3, 4, 5),
            fine_builder=lambda center: build_int_candidates(int(center), step=1, min_value=2, max_value=6),
        ),
        SearchParameterSpec(
            name="min_samples_leaf",
            meaning="叶子节点最少样本数。",
            trend_note="增大能显著降低森林对局部噪声的响应；减小时更灵活但更容易过拟合。",
            coarse_candidates=(20, 30, 40, 60, 80, 120),
            fine_builder=lambda center: build_int_candidates(int(center), step=10, min_value=10, max_value=160),
        ),
        SearchParameterSpec(
            name="min_samples_split",
            meaning="内部节点继续分裂所需的最少样本数。",
            trend_note="增大时分裂更保守；减小时每棵树更容易继续展开。",
            coarse_candidates=(40, 60, 80, 120, 160, 240),
            fine_builder=lambda center: build_int_candidates(int(center), step=20, min_value=20, max_value=320),
        ),
        SearchParameterSpec(
            name="max_features",
            meaning="单棵树分裂时可见的特征比例。",
            trend_note="减小时随机性更强、相关性更低；过小会损失有效信号。",
            coarse_candidates=(0.3, 0.4, 0.5, 0.6, 0.7),
            fine_builder=lambda center: build_float_candidates(float(center), step=0.05, min_value=0.25, max_value=0.8),
        ),
        SearchParameterSpec(
            name="max_samples",
            meaning="每棵树使用的样本抽样比例。",
            trend_note="减小时更随机、更抗过拟合；过小会削弱单棵树信息量。",
            coarse_candidates=(0.5, 0.6, 0.7, 0.8),
            fine_builder=lambda center: build_float_candidates(float(center), step=0.05, min_value=0.4, max_value=0.9),
        ),
        SearchParameterSpec(
            name="ccp_alpha",
            meaning="单棵树的代价复杂度剪枝强度。",
            trend_note="增大时每棵树会被更强剪枝；过大可能欠拟合。",
            coarse_candidates=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.02),
            fine_builder=lambda center: build_float_candidates(float(center), step=0.0005, min_value=0.0, max_value=0.05),
        ),
        SearchParameterSpec(
            name="min_impurity_decrease",
            meaning="继续分裂所需的最小纯度提升。",
            trend_note="增大时树更难继续分裂；减小时更容易吸收局部波动。",
            coarse_candidates=(0.0, 0.0005, 0.001, 0.002, 0.005, 0.01),
            fine_builder=lambda center: build_float_candidates(float(center), step=0.0005, min_value=0.0, max_value=0.02),
        ),
        SearchParameterSpec(
            name="n_estimators",
            meaning="森林中树的数量。",
            trend_note="增大通常更稳但训练更慢；树太多也可能把弱噪声累积放大。",
            coarse_candidates=(200, 300, 400),
            fine_builder=lambda center: build_int_candidates(int(center), step=50, min_value=100, max_value=600),
        ),
        SearchParameterSpec(
            name="class_weight",
            meaning="类别权重方案。",
            trend_note="balanced_subsample 会提高少数类权重，但也可能牺牲整体稳定性。",
            coarse_candidates=(None, "balanced_subsample"),
            fine_builder=lambda center: build_choice_candidates(center, (None, "balanced_subsample")),
        ),
    )
    initial_params = {
        "criterion": "entropy",
        "max_depth": 3,
        "min_samples_leaf": 40,
        "min_samples_split": 120,
        "max_features": 0.5,
        "max_samples": 0.7,
        "ccp_alpha": 0.002,
        "min_impurity_decrease": 0.001,
        "n_estimators": 300,
        "class_weight": None,
    }
    return ModelTrainingSpec(
        model_id=MODEL_ID,
        display_name="RandomForestClassifier",
        initial_params=initial_params,
        search_params=search_params,
        build_model_fn=lambda params: RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            bootstrap=True,
            **params,
        ),
    )


def train_model(ctx: RuntimeContext):
    """执行随机森林训练与搜索。"""
    return train_model_pipeline(ctx, build_training_spec())


def evaluate_saved_model(ctx: RuntimeContext):
    """读取本地随机森林模型并重新输出评估结果。"""
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
        print("随机森林训练完成，已保存：")
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
        print("已重新计算随机森林评估结果：")
        print(ctx.paths.output_file(MODEL_ID, ctx.scheme_name, "metrics.csv"))
        print(metrics_df)


if __name__ == "__main__":
    main()

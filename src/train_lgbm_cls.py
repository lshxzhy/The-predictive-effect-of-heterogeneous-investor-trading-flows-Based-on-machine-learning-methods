import argparse

from lightgbm import LGBMClassifier

from config import RuntimeContext, get_runtime_context
from model_train_utils import (
    ModelTrainingSpec,
    SearchParameterSpec,
    build_float_candidates,
    build_int_candidates,
    build_regularization_candidates,
    build_scaled_float_candidates,
    evaluate_saved_model_pipeline,
    train_model_pipeline,
)


MODEL_ID = "lgbm_cls"
RANDOM_STATE = 42
N_JOBS = -1


# LightGBM 的搜索延续筛选模型的思路，但把网格收得更紧，
# 重点控制叶子数、树深、最小叶子样本、采样和正则，防止单资产单期限过拟合。
def build_training_spec() -> ModelTrainingSpec:
    """构造 LightGBM 训练配置。"""
    search_params = (
        SearchParameterSpec(
            name="learning_rate",
            meaning="每轮 boosting 的步长。",
            trend_note="增大时收敛更快但更激进；减小时更平滑，通常需要更多树。",
            coarse_candidates=(0.005, 0.01, 0.02, 0.03),
            fine_builder=lambda center: build_float_candidates(float(center), step=0.005, min_value=0.005, max_value=0.05),
        ),
        SearchParameterSpec(
            name="n_estimators",
            meaning="Boosting 迭代轮数。",
            trend_note="增大通常提升拟合能力并拉长训练时间；过大更容易过拟合。",
            coarse_candidates=(80, 120, 160, 240, 320),
            fine_builder=lambda center: build_int_candidates(int(center), step=40, min_value=80, max_value=500),
        ),
        SearchParameterSpec(
            name="num_leaves",
            meaning="单棵树允许的最大叶子数。",
            trend_note="增大能增强非线性表达；过大时复杂度上升、过拟合风险更高。",
            coarse_candidates=(4, 6, 8, 12, 16),
            fine_builder=lambda center: build_int_candidates(int(center), step=2, min_value=2, max_value=20),
        ),
        SearchParameterSpec(
            name="max_depth",
            meaning="单棵树最大深度。",
            trend_note="增大允许树更深、拟合更强；减小时结构更保守。",
            coarse_candidates=(2, 3, 4),
            fine_builder=lambda center: build_int_candidates(int(center), step=1, min_value=2, max_value=5),
        ),
        SearchParameterSpec(
            name="min_child_samples",
            meaning="叶子节点最少样本数。",
            trend_note="增大能显著抑制过拟合；减小时模型更容易学习局部噪声。",
            coarse_candidates=(40, 60, 80, 120, 160),
            fine_builder=lambda center: build_int_candidates(int(center), step=10, min_value=20, max_value=220),
        ),
        SearchParameterSpec(
            name="min_child_weight",
            meaning="叶子节点最小二阶梯度和。",
            trend_note="增大后小样本叶子更难形成；减小时分裂更容易继续。",
            coarse_candidates=(0.1, 0.5, 1.0, 2.0, 5.0),
            fine_builder=lambda center: build_scaled_float_candidates(
                float(center),
                factors=(0.75, 1.0, 1.25, 1.5),
                min_value=0.1,
                max_value=8.0,
            ),
        ),
        SearchParameterSpec(
            name="min_split_gain",
            meaning="继续分裂所需的最小损失下降。",
            trend_note="增大时树更稀疏、分裂更保守；减小时更容易继续切分。",
            coarse_candidates=(0.05, 0.1, 0.2, 0.4, 0.8),
            fine_builder=lambda center: build_float_candidates(float(center), step=0.05, min_value=0.05, max_value=1.2),
        ),
        SearchParameterSpec(
            name="subsample",
            meaning="每轮训练的样本抽样比例。",
            trend_note="减小时随机性更强、往往更稳；过小会损失有效信息。",
            coarse_candidates=(0.4, 0.5, 0.6, 0.7),
            fine_builder=lambda center: build_float_candidates(float(center), step=0.05, min_value=0.3, max_value=0.8),
        ),
        SearchParameterSpec(
            name="colsample_bytree",
            meaning="每棵树可见的特征抽样比例。",
            trend_note="减小时有助于抑制共线噪声；过小会削弱有效信号。",
            coarse_candidates=(0.4, 0.5, 0.6, 0.7),
            fine_builder=lambda center: build_float_candidates(float(center), step=0.05, min_value=0.3, max_value=0.8),
        ),
        SearchParameterSpec(
            name="reg_alpha",
            meaning="L1 正则项系数。",
            trend_note="增大通常让模型更稀疏、更保守；过大则可能欠拟合。",
            coarse_candidates=(0.2, 0.5, 1.0, 2.0, 4.0),
            fine_builder=lambda center: [
                candidate
                for candidate in build_regularization_candidates(float(center), max_value=8.0)
                if float(candidate) >= 0.2
            ],
        ),
        SearchParameterSpec(
            name="reg_lambda",
            meaning="L2 正则项系数。",
            trend_note="增大可平滑参数波动、压制过拟合；过大时会削弱区分能力。",
            coarse_candidates=(2.0, 4.0, 8.0, 12.0),
            fine_builder=lambda center: [
                candidate
                for candidate in build_regularization_candidates(float(center), max_value=16.0)
                if float(candidate) >= 2.0
            ],
        ),
        SearchParameterSpec(
            name="scale_pos_weight",
            meaning="正类样本损失权重。",
            trend_note="增大通常提高正类召回倾向；减小时更偏向整体平衡。",
            coarse_candidates=(0.8, 1.0, 1.2),
            fine_builder=lambda center: build_float_candidates(float(center), step=0.1, min_value=0.6, max_value=1.6),
        ),
    )
    initial_params = {
        "learning_rate": 0.01,
        "n_estimators": 160,
        "num_leaves": 8,
        "max_depth": 3,
        "min_child_samples": 80,
        "min_child_weight": 1.0,
        "min_split_gain": 0.2,
        "subsample": 0.6,
        "colsample_bytree": 0.6,
        "reg_alpha": 1.0,
        "reg_lambda": 4.0,
        "scale_pos_weight": 1.0,
    }
    return ModelTrainingSpec(
        model_id=MODEL_ID,
        display_name="LGBMClassifier",
        initial_params=initial_params,
        search_params=search_params,
        build_model_fn=lambda params: LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            importance_type="gain",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            verbosity=-1,
            **params,
        ),
    )


def train_model(ctx: RuntimeContext):
    """执行 LightGBM 训练与搜索。"""
    return train_model_pipeline(ctx, build_training_spec())


def evaluate_saved_model(ctx: RuntimeContext):
    """读取本地 LightGBM 模型并重新输出评估结果。"""
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
        print("LightGBM 训练完成，已保存：")
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
        print("已重新计算 LightGBM 评估结果：")
        print(ctx.paths.output_file(MODEL_ID, ctx.scheme_name, "metrics.csv"))
        print(metrics_df)


if __name__ == "__main__":
    main()

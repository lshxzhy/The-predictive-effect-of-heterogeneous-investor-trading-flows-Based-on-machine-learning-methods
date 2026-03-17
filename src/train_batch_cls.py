import argparse

import pandas as pd

from config import PipelineConfig, get_runtime_context
from prepare_model_data import prepare_model_data_for_context
from train_dt_cls import train_model as train_dt_model
from train_lgbm_cls import train_model as train_lgbm_model
from train_rf_cls import train_model as train_rf_model
from train_xgb_cls import train_model as train_xgb_model


MODEL_TRAINERS = {
    "dt_cls": train_dt_model,
    "rf_cls": train_rf_model,
    "xgb_cls": train_xgb_model,
    "lgbm_cls": train_lgbm_model,
}
MODEL_ALIASES = {
    "dt": "dt_cls",
    "dt_cls": "dt_cls",
    "rf": "rf_cls",
    "rf_cls": "rf_cls",
    "xgb": "xgb_cls",
    "xgb_cls": "xgb_cls",
    "lgbm": "lgbm_cls",
    "lightgbm": "lgbm_cls",
    "lgbm_cls": "lgbm_cls",
}


# 批量脚本允许显式传 all，也允许传逗号分隔列表。
def resolve_asset_aliases(asset_arg: str, config: PipelineConfig) -> list[str]:
    """解析批量训练的资产列表。"""
    if asset_arg.strip().lower() == "all":
        return list(config.asset_specs.keys())

    asset_aliases = [item.strip() for item in asset_arg.split(",") if item.strip()]
    if not asset_aliases:
        raise ValueError("至少需要一个资产简称。")
    if len(set(asset_aliases)) != len(asset_aliases):
        raise ValueError(f"--assets 中存在重复资产简称：{asset_aliases}")
    invalid_assets = [asset for asset in asset_aliases if asset not in config.asset_specs]
    if invalid_assets:
        raise ValueError(f"--assets 中存在未配置资产：{invalid_assets}")
    return asset_aliases


# horizon 同样允许显式传 all，其他情况按整数列表解析。
def resolve_horizons(horizon_arg: str, config: PipelineConfig) -> list[int]:
    """解析批量训练的期限列表。"""
    if horizon_arg.strip().lower() == "all":
        return list(config.label_horizons)

    horizons = [int(item.strip()) for item in horizon_arg.split(",") if item.strip()]
    if not horizons:
        raise ValueError("至少需要一个期限。")
    if len(set(horizons)) != len(horizons):
        raise ValueError(f"--horizons 中存在重复期限：{horizons}")
    invalid_horizons = [horizon for horizon in horizons if horizon not in config.label_horizons]
    if invalid_horizons:
        raise ValueError(f"--horizons 中存在未配置期限：{invalid_horizons}")
    return horizons


# 模型参数允许传短名，也允许传完整脚本对应的 model_id。
def resolve_model_ids(model_arg: str) -> list[str]:
    """解析批量训练的模型列表。"""
    if model_arg.strip().lower() == "all":
        return list(MODEL_TRAINERS.keys())

    raw_models = [item.strip().lower() for item in model_arg.split(",") if item.strip()]
    if not raw_models:
        raise ValueError("至少需要一个模型。")

    resolved_models = []
    for raw_model in raw_models:
        if raw_model not in MODEL_ALIASES:
            raise ValueError(f"--models 中存在未配置模型：{raw_model}")
        resolved_models.append(MODEL_ALIASES[raw_model])

    if len(set(resolved_models)) != len(resolved_models):
        raise ValueError(f"--models 中存在重复模型：{resolved_models}")
    return resolved_models


# 如果模型文件和 metrics 都已存在，则允许在批量循环中跳过。
def should_skip_training(ctx, model_id: str) -> bool:
    """判断当前模型是否可以跳过训练。"""
    return (
        ctx.paths.model_file(model_id, ctx.scheme_name).exists()
        and ctx.paths.output_file(model_id, ctx.scheme_name, "metrics.csv").exists()
    )


# 统一把单次训练结果压成一行，便于批量运行结束后快速核对。
def build_summary_row(
    asset_alias: str,
    horizon: int,
    model_id: str,
    metrics_df: pd.DataFrame,
) -> dict[str, object]:
    """整理单次训练摘要。"""
    row: dict[str, object] = {
        "asset_alias": asset_alias,
        "horizon": horizon,
        "model_id": model_id,
    }
    for dataset_name in ["train", "valid", "test"]:
        dataset_row = metrics_df.loc[metrics_df["dataset"] == dataset_name].iloc[0]
        row[f"{dataset_name}_auc"] = dataset_row["auc"]
        row[f"{dataset_name}_f1"] = dataset_row["f1"]
        row[f"{dataset_name}_ks"] = dataset_row["ks"]
    row["train_valid_auc_gap"] = abs(row["train_auc"] - row["valid_auc"])
    return row


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", required=True)
    parser.add_argument("--horizons", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


# 批量脚本的循环顺序固定为“资产 -> 期限 -> 模型”，
# 每个资产期限组合先准备一次建模数据，再依次训练多个模型。
def main() -> None:
    """批量生成建模数据并训练多个分类模型。"""
    args = parse_args()
    config = PipelineConfig()
    asset_aliases = resolve_asset_aliases(args.assets, config)
    horizons = resolve_horizons(args.horizons, config)
    model_ids = resolve_model_ids(args.models)

    summary_rows: list[dict[str, object]] = []
    total_tasks = len(asset_aliases) * len(horizons) * len(model_ids)
    finished_tasks = 0

    for asset_alias in asset_aliases:
        for horizon in horizons:
            ctx = get_runtime_context(asset_alias=asset_alias, horizon=horizon)
            prepare_model_data_for_context(ctx)
            print(f"[DATA] 已准备建模数据：asset={asset_alias}, horizon={horizon}d")

            for model_id in model_ids:
                finished_tasks += 1
                print(f"[TASK {finished_tasks}/{total_tasks}] 开始训练：model={model_id}, asset={asset_alias}, horizon={horizon}d")

                if args.skip_existing and should_skip_training(ctx, model_id):
                    print(f"[SKIP] 已存在结果，跳过：model={model_id}, asset={asset_alias}, horizon={horizon}d")
                    continue

                metrics_df = MODEL_TRAINERS[model_id](ctx)[1]
                summary_rows.append(
                    build_summary_row(
                        asset_alias=asset_alias,
                        horizon=horizon,
                        model_id=model_id,
                        metrics_df=metrics_df,
                    )
                )

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values(
            ["model_id", "asset_alias", "horizon"]
        ).reset_index(drop=True)
        print("\n批量训练摘要：")
        print(summary_df)


if __name__ == "__main__":
    main()

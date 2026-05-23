from __future__ import annotations

import os
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 第 06 步：训练交易流辅助诊断模型
# 当前主线位置：单资产筛选输入 -> 交易流辅助筛选结果。
# 本文件负责：用候选控制变量解释四个固定交易流指标，生成按目标拆分的辅助诊断目录和汇总总表。
# 主函数：`run(assets, horizon)`。
# 命令行入口：`main()`，负责把命令行参数解析为 `run(...)` 调用。
# 直接输入：`data/processed/<asset>/screening_ready_panel.csv`；`src/config.py` 中的 `FIXED_TRADEFLOW_COLUMNS`、`SCREENING_CANDIDATE_COLUMNS`；`src/common/search_specs.py` 的回归搜索空间。
# 直接输出：`outputs/screening/screen_tradeflow_lgbm/<target>/*` 下的 `pred_*.csv`、`feature_importance.csv`、`metrics.csv`；`outputs/screening/tradeflow_summary/*.csv`；`models/screen_tradeflow_lgbm/<target>/screen_tradeflow_lgbm.joblib`。
# 下游读取：第 07 步正式变量清单和筛选参考总表；第 11 步章节结果。
# 关键修改位置：参与辅助诊断的资产集合、`FIXED_TRADEFLOW_COLUMNS`、`src/common/search_specs.py` 和本文件中的汇总逻辑。
# 变更后重跑起点：第 06 步。第 07 步正式变量清单会合并这里的结果。
# 对应文档：`README.md` 的“主线结构总览”“变量筛选逻辑”和“结果目录与阅读顺序”。
import argparse

import joblib
import pandas as pd
from lightgbm import LGBMRegressor

from src.config import (
    FIXED_TRADEFLOW_COLUMNS,
    MAINLINE_MODELS_DIR,
    SCREENING_ASSETS,
    SCREENING_CANDIDATE_COLUMNS,
    SCREENING_HORIZON,
    SCREENING_OUTPUT_DIR,
    asset_processed_dir,
    parse_assets,
)
from src.common.metrics_utils import regression_metrics
from src.common.paths import ensure_dir
from src.common.search_specs import TRADEFLOW_LGBM_INITIAL_PARAMS


def run(assets: list[str], horizon: int) -> None:
    """用控制变量解释四个固定交易流指标，并导出预测与重要性结果。"""
    # 读取第 2 步生成的单资产筛选面板，再拼接成多资产训练数据。
    pooled_parts = [pd.read_csv(asset_processed_dir(asset) / "screening_ready_panel.csv", parse_dates=["Date"]) for asset in assets]
    pooled = pd.concat(pooled_parts, ignore_index=True)
    # 每个交易流目标都有独立输出目录，避免四个诊断结果混在一起。
    output_base = ensure_dir(SCREENING_OUTPUT_DIR / "screen_tradeflow_lgbm")
    summary_dir = ensure_dir(SCREENING_OUTPUT_DIR / "tradeflow_summary")
    importance_rows = []
    metric_rows = []
    model_index_rows = []

    for target in FIXED_TRADEFLOW_COLUMNS:
        # 这里的目标变量是一个交易流指标，特征仍然是候选控制变量。
        # 之所以四个交易流变量要逐个训练，是因为每个交易流变量代表的行为维度不同，不能混成一个总分数。
        model = LGBMRegressor(random_state=42, n_jobs=-1, objective="regression", verbosity=-1, **TRADEFLOW_LGBM_INITIAL_PARAMS)
        # 继续使用第 2 步的时间切分，保证诊断与正式建模的样本边界一致。
        train_df = pooled.loc[pooled["split"] == "train"].copy()
        valid_df = pooled.loc[pooled["split"] == "valid"].copy()
        test_df = pooled.loc[pooled["split"] == "test"].copy()
        # 只用训练集拟合回归器，验证集和测试集仅用于输出诊断表现。
        model.fit(train_df[SCREENING_CANDIDATE_COLUMNS], train_df[target])
        run_name = f"screening_{horizon}d__tradeflow_target__{target}"
        model_dir = ensure_dir(MAINLINE_MODELS_DIR / "screen_tradeflow_lgbm" / run_name)
        out_dir = ensure_dir(output_base / run_name)
        split_frames = [("train", train_df), ("valid", valid_df), ("test", test_df)]
        for split_name, split_df in split_frames:
            # 保存每段样本的真实值和预测值，便于查看控制变量对交易流的解释能力。
            pred = model.predict(split_df[SCREENING_CANDIDATE_COLUMNS])
            pd.DataFrame({"Date": split_df["Date"], "dataset": split_name, "y_true": split_df[target], "y_pred": pred}).to_csv(
                out_dir / f"pred_{split_name}.csv", index=False
            )
            metrics = regression_metrics(split_df[target].to_numpy(), pred)
            metric_rows.append({"target_feature": target, "dataset": split_name, **metrics})
        # 每个目标各自导出重要性，第 7 步再合成筛选参考总表。
        importance_df = pd.DataFrame({"feature_name": SCREENING_CANDIDATE_COLUMNS, "importance_gain": model.feature_importances_})
        importance_df["importance_rank"] = importance_df["importance_gain"].rank(method="dense", ascending=False).astype(int)
        importance_df["target_feature"] = target
        importance_df.to_csv(out_dir / "feature_importance.csv", index=False)
        importance_rows.append(importance_df)
        # 这里的 metrics.csv 不是回归指标总表，而是记录诊断模型采用的是默认初始参数。
        pd.DataFrame([{"target_feature": target, "params_source": "initial_params_only"}]).to_csv(out_dir / "metrics.csv", index=False)
        # 保存模型文件是为了保证诊断结果可追溯，不会被第 11 步重新训练。
        joblib.dump({"model": model, "target_feature": target, "feature_columns": SCREENING_CANDIDATE_COLUMNS}, model_dir / "screen_tradeflow_lgbm.joblib")
        model_index_rows.append({"target_feature": target, "model_dir": str(model_dir), "output_dir": str(out_dir)})

    # 汇总目录下的三张表分别回答：
    # 1）哪个候选变量在解释四个交易流时更重要；
    # 2）各交易流目标在 train/valid/test 的拟合指标如何；
    # 3）每个目标对应的模型和输出文件放在哪里。
    pd.concat(importance_rows, ignore_index=True).to_csv(summary_dir / "screening_tradeflow_lgbm_importance_summary.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(summary_dir / "screening_tradeflow_lgbm_metrics.csv", index=False)
    pd.DataFrame(model_index_rows).to_csv(summary_dir / "screening_tradeflow_lgbm_model_index.csv", index=False)


def main() -> None:
    """命令行入口；业务逻辑在 run(assets, horizon) 中。"""
    parser = argparse.ArgumentParser(description="第 06 步：分别用候选控制变量解释四个固定交易流变量，生成辅助筛选参考。")
    parser.add_argument("--assets", default=None, help="逗号分隔的资产代码列表。通常与第 03、04 步保持一致；若不传，则按当前主线筛选资产范围执行。")
    parser.add_argument("--horizon", type=int, default=None, help="当前诊断结果的期限标记，主要用于输出命名。若不传，则按当前主线筛选期限执行。")
    args = parser.parse_args()
    run(parse_assets(args.assets) if args.assets else SCREENING_ASSETS.copy(), args.horizon if args.horizon is not None else SCREENING_HORIZON)


if __name__ == "__main__":
    main()


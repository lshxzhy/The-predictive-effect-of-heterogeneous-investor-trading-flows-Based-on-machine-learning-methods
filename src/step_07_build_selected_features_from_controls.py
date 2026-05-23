from __future__ import annotations

import os
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 第 07 步：生成正式变量清单和筛选参考总表
# 当前主线位置：统一筛选结果 -> 正式控制变量清单。
# 本文件负责：把确定的正式控制变量写成 `selected_features_best8_controls.csv`，并合并多来源筛选诊断生成 `screening_reference_mainline_controls.csv`。
# 主函数：`run(controls, output_name)`。
# 命令行入口：`main()`，负责把命令行参数解析为 `run(...)` 调用。
# 直接输入：第 04、05、06 步输出的缺失率表、相关性表、统一筛选重要性表和交易流辅助诊断汇总表；`src/config.py` 中的 `BEST8_CONTROL_COLUMNS`。
# 直接输出：`data/processed/screening/selected_features_best8_controls.csv`；`data/processed/screening/screening_reference_mainline_controls.csv`。
# 下游读取：第 08 步默认读取正式变量清单；第 11 步读取正式变量清单和筛选参考总表。
# 关键修改位置：`src/config.py` 的 `BEST8_CONTROL_COLUMNS`；本文件中正式变量清单字段、参考总表字段和输出命名。
# 变更后重跑起点：第 07 步。建模输入、训练结果、章节图表和项目清单都会受影响。
# 对应文档：`README.md` 的“主线结构总览”“变量筛选逻辑”和“常见修改场景”。
import argparse

import pandas as pd

from src.config import BEST8_CONTROL_COLUMNS, SCREENING_CANDIDATE_COLUMNS, SCREENING_DIR, SCREENING_OUTPUT_DIR
from src.common.paths import ensure_dir


# 正式建模读取 `selected_features_best8_controls.csv`。
# 下面这个文件用于汇总“为什么保留/删除某个控制变量”的参考信息。
# 两个文件的角色不同：前者会被第 08 步真正读入，后者仅作参考，不会自动改变建模变量。
REFERENCE_OUTPUT_NAME = "screening_reference_mainline_controls.csv"


def build_linear_reference(corr_df: pd.DataFrame) -> pd.DataFrame:
    """按与 4 个交易流变量的最大绝对 Pearson 相关系数生成线性参考排名。"""
    linear_df = (
        corr_df.groupby("candidate_feature", as_index=False)
        .agg(linear_max_abs_pearson_corr=("abs_pearson_corr", "max"))
        .rename(columns={"candidate_feature": "feature_name"})
        .sort_values(["linear_max_abs_pearson_corr", "feature_name"], ascending=[False, True])
        .reset_index(drop=True)
    )
    linear_df["linear_rank"] = linear_df["linear_max_abs_pearson_corr"].rank(method="dense", ascending=False).astype("Int64")
    return linear_df


def build_nonlinear_reference(tradeflow_importance_df: pd.DataFrame) -> pd.DataFrame:
    """按交易流辅助回归里的最佳重要性排名生成非线性参考。"""
    nonlinear_df = (
        tradeflow_importance_df.groupby("feature_name", as_index=False)
        .agg(
            nonlinear_best_importance_rank=("importance_rank", "min"),
            nonlinear_mean_importance_gain=("importance_gain", "mean"),
            nonlinear_target_count=("target_feature", "nunique"),
        )
        .sort_values(["nonlinear_best_importance_rank", "feature_name"], ascending=[True, True])
        .reset_index(drop=True)
    )
    nonlinear_df["nonlinear_rank"] = nonlinear_df["nonlinear_best_importance_rank"].rank(method="dense", ascending=True).astype("Int64")
    return nonlinear_df


def build_reference_frame(controls: list[str]) -> pd.DataFrame:
    """合并 pooled screening、缺失统计、线性相关和交易流回归重要性，生成筛选参考表。"""
    # 统一筛选模型的重要性：回答“哪些候选控制变量对涨跌预测更有贡献”。
    importance_df = pd.read_csv(SCREENING_OUTPUT_DIR / "unified_lgbm" / "screening_lgbm_importance.csv")
    # 缺失统计：回答“变量原始缺失是否严重”。
    missing_df = pd.read_csv(SCREENING_OUTPUT_DIR / "diagnostics" / "screening_missing_before_imputation.csv")
    # 线性相关：回答“变量是否与固定交易流指标高度线性相关”。
    corr_df = pd.read_csv(SCREENING_OUTPUT_DIR / "diagnostics" / "screening_trade_flow_corr.csv")
    # 辅助回归重要性：回答“变量是否能非线性解释固定交易流指标”。
    tradeflow_importance_df = pd.read_csv(SCREENING_OUTPUT_DIR / "tradeflow_summary" / "screening_tradeflow_lgbm_importance_summary.csv")

    # controls 是最终决定保留的控制变量，代码只负责落盘，不自动做最终取舍。
    # 也就是说，本步骤不会根据 importance 或相关性自动删列。
    selected = set(controls)
    ordered = controls + [col for col in SCREENING_CANDIDATE_COLUMNS if col not in selected]
    selection_order = {feature_name: idx + 1 for idx, feature_name in enumerate(ordered)}

    # 以完整候选变量清单为底表，逐项合并不同来源的参考指标。
    # 这样每个候选变量都能在同一行看到“预测重要性、缺失率、线性相关、辅助回归重要性”。
    reference_df = pd.DataFrame({"feature_name": SCREENING_CANDIDATE_COLUMNS})
    reference_df = reference_df.merge(importance_df, on="feature_name", how="left")
    reference_df = reference_df.merge(missing_df, on="feature_name", how="left")
    reference_df = reference_df.merge(build_linear_reference(corr_df), on="feature_name", how="left")
    reference_df = reference_df.merge(build_nonlinear_reference(tradeflow_importance_df), on="feature_name", how="left")
    reference_df["selected_for_model"] = reference_df["feature_name"].isin(selected)
    reference_df["excluded_from_model"] = ~reference_df["selected_for_model"]
    reference_df["drop_source"] = reference_df["selected_for_model"].map(
        {
            True: "best8_mainline",
            False: "excluded_from_best8_mainline",
        }
    )
    reference_df["selection_order"] = reference_df["feature_name"].map(selection_order).astype("Int64")
    reference_df["union_rank"] = (
        reference_df[["linear_rank", "nonlinear_rank"]]
        .min(axis=1)
        .astype("Int64")
    )
    return reference_df


def run(controls: list[str], output_name: str) -> None:
    """把固定控制变量清单写成主线可直接消费的 selected_features 文件和筛选参考总表。"""
    selected = set(controls)
    # 防止命令行里写入不存在的变量名，避免第 8 步准备模型输入时报错。
    unknown = selected - set(SCREENING_CANDIDATE_COLUMNS)
    if unknown:
        raise ValueError(f"unknown controls: {sorted(unknown)}")

    reference_df = build_reference_frame(controls)
    ensure_dir(SCREENING_DIR)
    # selected_features_best8_controls.csv 是第 08 步默认读取的正式建模输入。
    # 只要第 08 步不额外指定 --selected-features-path，它读的就是这里写出的文件。
    selected_features_df = (
        reference_df.sort_values(["selection_order", "feature_name"], ascending=[True, True])
        .reset_index(drop=True)
        .copy()
    )
    selected_features_df.to_csv(SCREENING_DIR / output_name, index=False)

    # screening_reference_mainline_controls.csv 是参考表，不会自动改变模型变量。
    # 这张表可以理解为“为什么最后选这 8 个控制变量”的证据总表。
    screening_reference_df = (
        reference_df.rename(
            columns={
                "importance_rank": "screening_importance_rank",
                "importance_gain": "screening_importance_gain",
            }
        )
        .sort_values(
            ["selected_for_model", "union_rank", "screening_importance_rank", "feature_name"],
            ascending=[False, True, True, True],
        )
        .reset_index(drop=True)
    )
    screening_reference_df.to_csv(SCREENING_DIR / REFERENCE_OUTPUT_NAME, index=False)


def main() -> None:
    """命令行入口；业务逻辑在 run(controls, output_name) 中。"""
    parser = argparse.ArgumentParser(description="第 07 步：把控制变量清单写成正式建模输入，并生成筛选参考总表。")
    parser.add_argument("--controls", default=None, help="逗号分隔的控制变量代码列表。这些变量会真正进入第 08 步正式建模；若不传，则按当前主线控制变量配置执行。")
    parser.add_argument("--output-name", default="selected_features_best8_controls.csv", help="正式建模变量文件名。通常保持默认即可。")
    args = parser.parse_args()
    controls = [item.strip() for item in args.controls.split(",") if item.strip()] if args.controls else BEST8_CONTROL_COLUMNS.copy()
    run(controls, args.output_name)


if __name__ == "__main__":
    main()

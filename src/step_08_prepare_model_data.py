from __future__ import annotations

import os
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 第 08 步：生成建模输入
# 当前主线位置：特征面板与正式变量清单 -> 建模输入目录。
# 本文件负责：按资产、期限和方案，把 `feature_panel.csv` 与正式变量清单整理成 `train/valid/test` 的 `*_prepared.csv` 和 `*_model_input.csv`。
# 主函数：`run(asset, horizon, scheme, experiment_tag, selected_features_path)`。
# 命令行入口：`main()`，负责把命令行参数解析为 `run(...)` 调用。
# 直接输入：`data/processed/<asset>/feature_panel.csv`；`data/processed/screening/selected_features_best8_controls.csv`；`src/config.py` 中的 `FIXED_TRADEFLOW_COLUMNS`、`PAPER_HORIZONS`、`MAINLINE_SCHEMES`、`MAINLINE_EXPERIMENT_TAGS`。
# 直接输出：`data/processed/<asset>/horizons/<horizon>d/<scheme>__<experiment_tag>/train_prepared.csv`；`valid_prepared.csv`；`test_prepared.csv`；`train_model_input.csv`；`valid_model_input.csv`；`test_model_input.csv`。
# 下游读取：第 09 步训练目录；第 11 步章节结果回看 `*_prepared.csv` 和建模目录。
# 关键修改位置：正式预测资产、期限、方案、实验标签；正式变量清单读取方式；第 08 步中的特征拼接和目录命名逻辑。
# 变更后重跑起点：第 08 步。分类训练和章节图表都会读取这里生成的目录。
# 对应文档：`README.md` 的“主线结构总览”“文件流转与关键中间文件”和“重跑关系”。
import argparse

import pandas as pd

from pathlib import Path

from src.config import FIXED_TRADEFLOW_COLUMNS, MAINLINE_SCHEMES, PAPER_ASSETS, PAPER_HORIZONS, SCREENING_DIR, asset_horizon_dir, asset_processed_dir, experiment_tag_for_horizon, parse_scheme
from src.common.paths import ensure_dir
from src.common.preprocessing import apply_train_impute_and_scale


def selected_features(path: str | None = None) -> list[str]:
    """读取主线控制变量文件，并返回被保留的控制变量列表。"""
    # 注意:
    # - 正式建模只读取 `selected_features_best8_controls.csv`。
    # - `screening_reference_mainline_controls.csv` 仅作参考，不会被本函数自动消费。
    feature_path = Path(path) if path else SCREENING_DIR / "selected_features_best8_controls.csv"
    # selected_for_model=True 的变量才进入正式建模；参考表不会在这里被自动读取。
    df = pd.read_csv(feature_path)
    return df.loc[df["selected_for_model"], "feature_name"].tolist()


def run(
    asset: str,
    horizon: int,
    scheme: str,
    selected_features_path: str | None = None,
    experiment_tag: str | None = None,
) -> None:
    """把单资产特征面板整理成分类模型直接读取的三份 split 输入文件。"""
    # feature_panel.csv 来自第 2 步，包含所有候选特征和全部预测期限目标。
    feature_panel = pd.read_csv(asset_processed_dir(asset) / "feature_panel.csv", parse_dates=["Date"])
    # 读取第 7 步确定的正式控制变量清单。
    selected = selected_features(selected_features_path)
    # with_tradeflow4 方案 = 4 个固定交易流变量 + 控制变量；no_tradeflow4 方案 = 仅控制变量。
    # 如果需要比较“加入交易流”和“不加入交易流”的区别，就分别运行两次本命令。
    feature_cols = selected if scheme == "no_tradeflow4" else FIXED_TRADEFLOW_COLUMNS + selected
    target_label_col = f"target_label_{horizon}d"
    target_return_col = f"target_return_{horizon}d"
    # prepared 表保留未来收益，便于核对结果；model_input 表只保留分类目标和特征，供模型训练。
    # 后续第 11 步做重点结果表和图表解释时，经常会回看 prepared 表。
    keep_cols = ["Date", "split", target_return_col, target_label_col, *feature_cols]
    df = feature_panel[keep_cols].copy()
    # 对正式建模变量再次做训练集口径的填补和标准化，确保每个资产/期限/方案输入独立干净。
    df, _ = apply_train_impute_and_scale(
        df,
        feature_cols=feature_cols,
        split_col="split",
        skip_scale_cols=[col for col in ["IND_SECTOR_TV_ene_norm", "INS_SECTOR_TV_ene_norm"] if col in feature_cols],
    )
    # 模型训练要求二分类标签为整数。
    df[target_label_col] = df[target_label_col].astype(int)
    # 每个资产、期限、方案和实验标签都有独立目录，避免不同结果互相覆盖。
    # experiment_tag 变化后，输出目录名也会变化，因此需要与第 09、10 步保持一致。
    out_dir = ensure_dir(asset_horizon_dir(asset, horizon, scheme, experiment_tag))
    for split in ["train", "valid", "test"]:
        # 保持第 2 步的时间切分，不重新抽样。
        subset = df.loc[df["split"] == split].copy()
        prepared = subset[["Date", target_return_col, target_label_col, *feature_cols]].copy()
        model_input = subset[["Date", target_label_col, *feature_cols]].copy()
        # prepared 用于结果核对和复查。
        prepared.to_csv(out_dir / f"{split}_prepared.csv", index=False)
        # model_input 是第 09 步训练脚本真正读取的文件。
        model_input.to_csv(out_dir / f"{split}_model_input.csv", index=False)


def main() -> None:
    """命令行入口；业务逻辑在 run(...) 中。"""
    parser = argparse.ArgumentParser(description="第 08 步：按资产、期限和方案生成 train/valid/test 的建模输入文件。")
    parser.add_argument("--asset", default=None, help="单个资产代码，例如 index、eg、bu、jm、pp。若不传，则按当前主线正式预测资产范围全跑。")
    parser.add_argument("--horizon", type=int, default=None, help="预测期限，例如 1 或 22。若不传，则按当前主线正式预测期限范围全跑。")
    parser.add_argument("--scheme", default=None, help="特征方案。若不传，则按当前主线方案范围全跑。")
    parser.add_argument("--selected-features-path", default=None, help="正式控制变量文件路径。默认读取第 07 步生成的 selected_features_best8_controls.csv。")
    parser.add_argument("--experiment-tag", default=None, help="实验标签。显式单次运行时可手动指定；不传则按当前主线期限标签映射自动选择。")
    args = parser.parse_args()
    explicit_requested = any(value is not None for value in [args.asset, args.horizon, args.scheme, args.experiment_tag])
    if not explicit_requested:
        for asset in PAPER_ASSETS:
            for horizon in PAPER_HORIZONS:
                experiment_tag = experiment_tag_for_horizon(horizon)
                for scheme in MAINLINE_SCHEMES:
                    run(asset, horizon, scheme, args.selected_features_path, experiment_tag)
        return
    missing = [flag for flag, value in {"--asset": args.asset, "--horizon": args.horizon, "--scheme": args.scheme}.items() if value is None]
    if missing:
        parser.error("显式运行模式需要同时提供 --asset、--horizon 和 --scheme。无参点击运行则会按主线配置全量执行。")
    run(args.asset, args.horizon, parse_scheme(args.scheme), args.selected_features_path, args.experiment_tag or experiment_tag_for_horizon(args.horizon))


if __name__ == "__main__":
    main()


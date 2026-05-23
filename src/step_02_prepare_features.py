from __future__ import annotations

import os
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 第 02 步：构造特征面板
# 当前主线位置：单资产原始面板 -> 特征面板与筛选输入。
# 本文件负责：从 `merged_raw_panel.csv` 生成衍生特征、预测目标、时间切分、`feature_panel.csv` 和 `screening_ready_panel.csv`。
# 主函数：`run(asset)`。
# 命令行入口：`main()`，负责把命令行参数解析为 `run(...)` 调用。
# 直接输入：`data/processed/<asset>/merged_raw_panel.csv`；`src/common/feature_registry.py` 的特征与目标定义；`src/common/preprocessing.py` 的预处理逻辑。
# 直接输出：`data/processed/<asset>/feature_panel.csv`；`feature_desc_stats.csv`；`screening_ready_panel.csv`；`screening_preprocess_stats.csv`。
# 下游读取：第 03、04、06、08 步。
# 关键修改位置：`src/common/feature_registry.py`；`src/common/preprocessing.py`；`src/config.py` 中的 `HORIZONS` 和 `FIXED_TRADEFLOW_COLUMNS`。
# 变更后重跑起点：第 02 步。后续筛选、建模输入、训练结果和章节图表都依赖这里的字段结构。
# 对应文档：`README.md` 的“主线结构总览”和“文件流转与关键中间文件”。
import argparse

import numpy as np
import pandas as pd

from src.config import ASSET_SPECS, FIXED_TRADEFLOW_COLUMNS, SCREENING_ASSETS, SCREENING_FEATURE_COLUMNS, asset_processed_dir
from src.common.feature_registry import (
    add_targets,
    assign_time_split,
    build_derived_features,
    build_feature_panel,
    build_screening_ready_panel,
    feature_desc_stats,
    trim_to_longest_horizon,
)
from src.common.paths import ensure_dir
from src.common.preprocessing import apply_train_impute_and_scale


def run(asset: str) -> None:
    """基于单资产原始面板生成特征面板、筛选面板和预处理统计。"""
    # 先校验资产是否在 config.py 的资产主表中，保证路径和 sheet 口径一致。
    if asset not in ASSET_SPECS:
        raise ValueError(f"unknown asset: {asset}")
    # 第 1 步已经把该资产的中间文件写入 data/processed/<asset>/。
    asset_dir = ensure_dir(asset_processed_dir(asset))
    merged_path = asset_dir / "merged_raw_panel.csv"
    # Date 按日期读取，避免后续时间切分时被当成普通字符串排序。
    df = pd.read_csv(merged_path, parse_dates=["Date"])
    # 先把正负无穷统一视为缺失，避免均值、标准差和模型训练被异常数值污染。
    df = df.replace([np.inf, -np.inf], np.nan)
    # 先在原始列基础上构造衍生变量。
    # 这里生成的变量既包含后续主线会直接用到的控制变量，也包含统一筛选阶段需要比较的候选变量。
    df = build_derived_features(df)
    # 为 1/22/44/66/88 日期限同时生成未来收益和涨跌标签。
    # 这样后续如果主线期限从 1 日、22 日换成其他期限，只需要先在 config.py 中调整期限口径，
    # 再从第 02 步开始重跑，不需要回到原始 Excel 重新整理。
    df = add_targets(df)
    # 按最长 88 日预测期限裁掉尾部无未来收益的样本。
    # 这样可以保证留下来的每一行样本，对所有预测期限都有完整目标值。
    df = trim_to_longest_horizon(df)
    # 按时间顺序切成训练集、验证集和测试集，不能随机打乱。
    # 时间序列任务不能随机切分，否则未来信息会泄漏到训练阶段。
    df, _ = assign_time_split(df)
    # 整理成标准 feature_panel.csv，后续建模和筛选都从这里取字段。
    # 可以把 feature_panel.csv 理解为“这个资产最完整的标准中间表”。
    feature_panel = build_feature_panel(df)
    feature_panel.to_csv(asset_dir / "feature_panel.csv", index=False)
    # 描述统计只用于查看变量分布，不参与后续模型训练。
    feature_desc_stats(feature_panel).to_csv(asset_dir / "feature_desc_stats.csv", index=False)

    # screening_ready_panel 是 pooled screening 的单资产输入，只保留筛选所需字段。
    # 第 03、04、06 步不会再直接读 merged_raw_panel.csv，而是读这个已经整理好的筛选面板。
    screening_ready = build_screening_ready_panel(feature_panel)
    # 缺失填补和标准化严格只用训练集统计量，验证集/测试集只套用训练集规则。
    # 如果需要查看填补值、均值、标准差等具体统计量，请直接查看后面写出的 screening_preprocess_stats.csv。
    screening_ready, stats_df = apply_train_impute_and_scale(
        screening_ready,
        feature_cols=SCREENING_FEATURE_COLUMNS,
        split_col="split",
        skip_scale_cols=["IND_SECTOR_TV_ene_norm", "INS_SECTOR_TV_ene_norm"],
    )
    # 分类目标必须转成整数，避免 LightGBM 等模型把它识别成连续值。
    screening_ready["target_label_1d"] = screening_ready["target_label_1d"].astype(int)
    # screening_ready_panel.csv 供第 03 步拼接长面板、第 04 步做筛选诊断、第 06 步做交易流辅助诊断读取。
    screening_ready.to_csv(asset_dir / "screening_ready_panel.csv", index=False)
    # screening_preprocess_stats.csv 记录每个变量的填补值、均值、标准差和是否标准化。
    # 如果需要查看某一列为什么会被标准化，或者为什么没有被标准化，就看这个文件。
    stats_df.to_csv(asset_dir / "screening_preprocess_stats.csv", index=False)


def main() -> None:
    """命令行入口；业务逻辑在 run(asset) 中。"""
    parser = argparse.ArgumentParser(description="第 02 步：在 merged_raw_panel.csv 基础上构造 feature_panel.csv 和 screening_ready_panel.csv。")
    # --asset 控制本次只处理一个资产。
    # 如果主线资产发生替换，应对新资产重新执行第 01 步和第 02 步，再继续后面的筛选或建模。
    parser.add_argument("--asset", default=None, help="资产代码，例如 eg、bu、jm、pp、index。若不传，则按当前主线筛选资产范围顺序全跑。")
    args = parser.parse_args()
    if args.asset:
        run(args.asset)
        return
    for asset in SCREENING_ASSETS:
        run(asset)


if __name__ == "__main__":
    main()


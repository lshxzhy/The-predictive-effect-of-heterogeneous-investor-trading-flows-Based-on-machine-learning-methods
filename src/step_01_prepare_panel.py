from __future__ import annotations

import os
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 第 01 步：合并原始数据
# 当前主线位置：原始 Excel -> 单资产原始面板。
# 本文件负责：读取两份原始 Excel，按资产生成 `merged_raw_panel.csv` 和 `raw_missing_stats.csv`。
# 主函数：`run(asset)`。
# 命令行入口：`main()`，负责把命令行参数解析为 `run(...)` 调用。
# 直接输入：`data/raw/核心变量时间序列日度.xlsx`；`data/raw/指标数据补充.xlsx`；`src/config.py` 中的 `ASSET_SPECS`。
# 直接输出：`data/processed/<asset>/merged_raw_panel.csv`；`data/processed/<asset>/raw_missing_stats.csv`。
# 下游读取：第 02 步 `src.step_02_prepare_features`。
# 关键修改位置：`src/config.py` 的 `ASSET_SPECS`；`src/common/excel_utils.py` 的原始字段读取和列校验逻辑。
# 变更后重跑起点：第 01 步。单资产原始面板是后续全部步骤的起点。
# 对应文档：`README.md` 的“主线结构总览”和“文件流转与关键中间文件”。
import argparse

from src.config import ASSET_SPECS, SCREENING_ASSETS, asset_processed_dir
from src.common.excel_utils import load_core_raw, load_market_raw, missing_stats
from src.common.paths import ensure_dir


def run(asset: str) -> None:
    """读取原始 Excel，生成单资产原始面板和原始缺失统计。"""
    # 先检查资产代码是否在 config.py 的资产主表中，避免输入错误后生成错误目录。
    if asset not in ASSET_SPECS:
        raise ValueError(f"unknown asset: {asset}")
    # 核心变量文件包含交易流、分歧、隐含波动率和利率类变量。
    core_df = load_core_raw()
    # 市场指标文件按资产分 sheet，这里只读取本次命令指定的资产。
    market_df = load_market_raw(asset)

    # 以 Date 为唯一时间键做左连接：核心变量日期保留，市场指标按同日补入。
    merged = core_df.merge(market_df, on="Date", how="left")
    # 补充资产简称和资产代码，后续多资产拼接时靠这两列识别来源。
    merged["asset_alias"] = asset
    merged["asset_code"] = ASSET_SPECS[asset]["asset_code"]
    # 固定列数校验用于防止原始 Excel 多列、少列或列名变化后静默进入后续步骤。
    # 如果这里报错，通常说明原始 Excel 列名或字段数量发生了变化，应先检查原始数据文件，而不是直接跳过校验。
    if merged.shape[1] != 26:
        raise ValueError(f"{asset}: expected merged_raw_panel 26 columns, got {merged.shape[1]}")

    # 单资产中间结果统一写到 data/processed/<asset>/，不存在时自动创建。
    out_dir = ensure_dir(asset_processed_dir(asset))
    # merged_raw_panel.csv 是第 2 步唯一读取的主表。
    # 第 2 步就是从这个文件继续读取数据。
    merged.to_csv(out_dir / "merged_raw_panel.csv", index=False)
    # raw_missing_stats.csv 仅用于检查原始数据缺失情况，不参与训练。
    # 这个文件主要回答“原始 Excel 本身哪几列缺得多”，不会被后续模型自动读取。
    missing_stats(merged).to_csv(out_dir / "raw_missing_stats.csv", index=False)


def main() -> None:
    """命令行入口；业务逻辑在 run(asset) 中。"""
    parser = argparse.ArgumentParser(description="第 01 步：读取两份原始 Excel，并为单个资产生成 merged_raw_panel.csv。")
    # --asset 控制本次命令只处理一个资产。
    # 首次执行主线时，通常需要对当前主线涉及的每个资产各运行一次本命令。
    parser.add_argument("--asset", default=None, help="资产代码，例如 eg、bu、jm、pp、index。若不传，则按当前主线筛选资产范围顺序全跑。")
    args = parser.parse_args()
    if args.asset:
        run(args.asset)
        return
    for asset in SCREENING_ASSETS:
        run(asset)


if __name__ == "__main__":
    main()


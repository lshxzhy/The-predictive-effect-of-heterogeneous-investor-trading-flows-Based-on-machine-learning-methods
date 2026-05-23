from __future__ import annotations

import os
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 第 03 步：拼接筛选长面板
# 当前主线位置：单资产筛选输入 -> 统一筛选长表。
# 本文件负责：把多个资产的 `screening_ready_panel.csv` 合并成统一 pooled screening 输入表 `screening_long_panel.csv`。
# 主函数：`run(assets)`。
# 命令行入口：`main()`，负责把命令行参数解析为 `run(...)` 调用。
# 直接输入：`data/processed/<asset>/screening_ready_panel.csv`；`src/config.py` 中的 `SCREENING_ASSETS`、`SCREENING_CANDIDATE_COLUMNS`。
# 直接输出：`data/processed/screening/screening_long_panel.csv`。
# 下游读取：第 04、05 步。
# 关键修改位置：`src/config.py` 的 `SCREENING_ASSETS`、`SCREENING_CANDIDATE_COLUMNS`；本文件中的长表拼接、列筛选和资产顺序组织方式。
# 变更后重跑起点：第 03 步。第 04 到第 07 步都建立在统一筛选长表上。
# 对应文档：`README.md` 的“主线结构总览”“文件流转与关键中间文件”和“变量筛选逻辑”。
import argparse

import pandas as pd

from src.config import SCREENING_ASSETS, SCREENING_CANDIDATE_COLUMNS, SCREENING_DIR, asset_processed_dir, parse_assets
from src.common.paths import ensure_dir


def run(assets: list[str]) -> None:
    """按 train/valid/test 顺序拼接多资产筛选长表。"""
    parts = []
    # 先按 split 再按资产拼接，保证输出顺序稳定，便于复核。
    # 之所以不是按资产整段拼接，是为了让 pooled screening 长面板保持“先训练、后验证、最后测试”的固定顺序。
    for split in ["train", "valid", "test"]:
        for asset in assets:
            # 每个资产的 screening_ready_panel.csv 已在第 2 步完成填补和标准化。
            path = asset_processed_dir(asset) / "screening_ready_panel.csv"
            df = pd.read_csv(path, parse_dates=["Date"])
            # 统一长面板只保留筛选候选变量、资产标识、时间切分和 1 日涨跌目标。
            # 当前统一筛选的口径固定是“用 1 日涨跌标签评估候选控制变量”，因此这里不带入 22/44/66/88 日期限目标。
            subset = df.loc[df["split"] == split, ["Date", "asset_alias", "asset_code", "split", *SCREENING_CANDIDATE_COLUMNS, "target_label_1d"]]
            # 筛选模型不能接收缺失候选变量；如仍有缺失，说明第 2 步预处理没有正确完成。
            if subset[SCREENING_CANDIDATE_COLUMNS].isna().any().any():
                raise ValueError(f"{asset}: screening_ready_panel still has NaN candidates")
            parts.append(subset)
    # 多资产合并后形成 pooled screening 的唯一输入表。
    out = pd.concat(parts, ignore_index=True)
    # 固定列数校验用于防止候选变量清单变化后没有同步说明文档。
    if out.shape[1] != 32:
        raise ValueError(f"screening_long_panel expected 32 columns, got {out.shape[1]}")
    # screening_long_panel.csv 是第 05 步统一筛选模型唯一读取的训练输入表。
    ensure_dir(SCREENING_DIR)
    out.to_csv(SCREENING_DIR / "screening_long_panel.csv", index=False)


def main() -> None:
    """命令行入口；业务逻辑在 run(assets) 中。"""
    parser = argparse.ArgumentParser(description="第 03 步：把多个资产的 screening_ready_panel.csv 拼成统一筛选长面板。")
    parser.add_argument("--assets", default=None, help="逗号分隔的资产代码列表，例如 index,sc,lu,pp,eg,bu,jm。若不传，则按当前主线筛选资产范围执行。")
    args = parser.parse_args()
    run(parse_assets(args.assets) if args.assets else SCREENING_ASSETS.copy())


if __name__ == "__main__":
    main()


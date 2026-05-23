from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# 本模块负责：按训练集统计量执行缺失填补和标准化，避免未来信息泄漏。
# 直接服务的步骤：第 02 步和第 08 步。
# 关键修改位置：填补值计算方式、标准化规则、`skip_scale_cols` 的处理方式。
# 变更后重跑起点：第 02 步或第 08 步，取决于修改影响的是筛选输入还是正式建模输入。


@dataclass
class ColumnPreprocessStats:
    feature_name: str
    fill_value: float
    mean: float
    std: float
    scaled: bool


def apply_train_impute_and_scale(
    df: pd.DataFrame,
    feature_cols: list[str],
    split_col: str = "split",
    skip_scale_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """用训练集统计量完成缺失填补和标准化，并返回逐列处理记录。"""
    # skip_scale_cols 用来告诉函数“哪些列需要保留原始量纲，不做标准化”。
    # 例如交易流比例类变量通常更适合保留原始含义，因此会在调用处传入这个列表。
    skip_scale = set(skip_scale_cols or [])
    out = df.copy()
    stats_records: list[dict[str, object]] = []
    train_mask = out[split_col] == "train"
    if not train_mask.any():
        raise ValueError("train split is empty")

    for column in feature_cols:
        # 先把无穷值统一转成缺失，再按训练集统计量处理。
        series = out[column].replace([np.inf, -np.inf], np.nan)
        train_series = series.loc[train_mask]
        # 填补值、均值和标准差都只用训练集计算，避免未来信息泄漏。
        # 当前默认用训练集的中位数做缺失填补，因为它对极端值更稳健。
        fill_value = float(train_series.median()) if train_series.notna().any() else 0.0
        series = series.fillna(fill_value)
        # 这里记录的 mean 和 std 也只来自训练集。
        # 如需核对某个变量的预处理数值，可回查调用方落盘的 preprocess_stats 文件。
        mean = float(series.loc[train_mask].mean())
        std = float(series.loc[train_mask].std(ddof=0))
        scaled = column not in skip_scale
        if scaled:
            # 如果训练集标准差为 0，说明这列在训练阶段没有波动。
            # 这时无法按常规公式标准化，所以统一写成 0，避免除以 0。
            if pd.isna(std) or std == 0:
                series = pd.Series(0.0, index=series.index)
            else:
                series = (series - mean) / std
        out[column] = series
        stats_records.append(
            {
                "feature_name": column,
                "fill_value": fill_value,
                "train_mean": mean,
                "train_std": std,
                "scaled": scaled,
            }
        )

    # 预处理完成后不允许再保留缺失。
    # 如果这里报错，通常说明某些列在前面没有被正确填补，应该先检查输入面板或 skip_scale_cols 是否写错。
    if out[feature_cols].isna().any().any():
        raise ValueError("NaN remains after imputation and scaling")
    return out, pd.DataFrame(stats_records)

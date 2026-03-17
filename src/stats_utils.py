from pathlib import Path

import pandas as pd


# 描述统计只面向数值列，并允许显式排除目标列，
# 输出结构固定为“字段名 + 描述统计量 + 偏度峰度”。
def save_descriptive_stats(
    df: pd.DataFrame,
    desc_file: Path,
    exclude_cols: list[str] | None = None,
) -> None:
    """保存描述性统计表。"""
    exclude_cols = exclude_cols or []
    numeric_cols = [
        col
        for col in df.select_dtypes(include=["number"]).columns
        if col not in exclude_cols
    ]

    if not numeric_cols:
        raise ValueError("没有可用于描述性统计的数值列。")

    stats = df[numeric_cols].describe(
        percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    ).T
    stats["skew"] = df[numeric_cols].skew()
    stats["kurt"] = df[numeric_cols].kurt()
    stats.index.name = "column_name"
    stats.reset_index().to_csv(desc_file, index=False, encoding="utf-8-sig")

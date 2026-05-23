from __future__ import annotations

# 本模块负责：按主线固定口径读取两份原始 Excel，并在读取阶段完成字段筛选、日期校验、数值校验和缺失统计。
# 直接服务的步骤：第 01 步 `src.step_01_prepare_panel`。
# 关键修改位置：`src/config.py` 中的原始文件名、资产 sheet 映射和字段白名单；本文件中的读取、重命名和校验逻辑。
# 变更后重跑起点：第 01 步。原始表读取口径变化会影响单资产原始面板及其后全部步骤。

from dataclasses import dataclass

import pandas as pd

from src.config import (
    ASSET_SPECS,
    CORE_SOURCE_COLUMNS,
    MARKET_SOURCE_COLUMNS,
    RAW_CORE_FILE_NAME,
    RAW_DIR,
    RAW_MARKET_FILE_NAME,
)


@dataclass(frozen=True)
class ValidationReport:
    """记录单张原始表清洗后的行列规模，便于复核读取是否正常。"""

    table_name: str
    rows: int
    cols: int


def _strict_parse_date_series(raw: pd.Series, table_name: str) -> pd.Series:
    """严格解析日期列，遇到非法日期值时直接报错。"""

    parsed = pd.to_datetime(raw, errors="coerce")
    missing_input = raw.isna()
    invalid = (~missing_input) & parsed.isna()
    if invalid.any():
        bad_values = raw.loc[invalid].astype(str).head(5).tolist()
        raise ValueError(f"{table_name}: invalid Date values {bad_values}")
    return parsed


def _coerce_numeric_series(raw: pd.Series, table_name: str, column: str) -> pd.Series:
    """把数值列强制转成数值类型，遇到非法字符时直接报错。"""

    numeric = pd.to_numeric(raw, errors="coerce")
    invalid = raw.notna() & numeric.isna()
    if invalid.any():
        bad_values = raw.loc[invalid].astype(str).head(5).tolist()
        raise ValueError(f"{table_name}: invalid numeric values in {column}: {bad_values}")
    return numeric


def clean_and_validate_raw_table(df: pd.DataFrame, table_name: str) -> tuple[pd.DataFrame, ValidationReport]:
    """统一清洗原始表，并检查空表、日期列、重复日期和排序问题。"""

    if df.empty:
        raise ValueError(f"{table_name}: empty table")
    if "Date" not in df.columns:
        raise ValueError(f"{table_name}: missing Date column")

    # 统一复制一份再清洗，避免调用方误改原始 DataFrame。
    df = df.copy()
    other_cols = [col for col in df.columns if col != "Date"]
    date_blank_with_values = df["Date"].isna() & df[other_cols].notna().any(axis=1)
    if date_blank_with_values.any():
        raise ValueError(f"{table_name}: blank Date with non-empty values exists")

    # 所有主线步骤都默认 Date 是严格可解析并升序排列的交易日索引。
    df["Date"] = _strict_parse_date_series(df["Date"], table_name)
    df = df.loc[df["Date"].notna()].copy()

    for column in other_cols:
        # 原始 Excel 有时会混入字符串或格式化字符，这里提前拦住，后续特征工程就不用重复防守。
        df[column] = _coerce_numeric_series(df[column], table_name, column)

    if df["Date"].duplicated().any():
        dupes = df.loc[df["Date"].duplicated(), "Date"].dt.strftime("%Y-%m-%d").tolist()[:5]
        raise ValueError(f"{table_name}: duplicated Date values {dupes}")

    df = df.sort_values("Date").reset_index(drop=True)
    if not df["Date"].is_monotonic_increasing:
        raise ValueError(f"{table_name}: Date is not sorted ascending")

    return df, ValidationReport(table_name=table_name, rows=len(df), cols=df.shape[1])


def load_core_raw() -> pd.DataFrame:
    """读取核心行为与宏观变量原始表，并按主线口径筛列校验。"""

    path = RAW_DIR / RAW_CORE_FILE_NAME
    # 这份表提供交易流、分歧指标和利差类变量，是第 01 步合并时的核心底表。
    df = pd.read_excel(path)
    keep_cols = ["Date"] + CORE_SOURCE_COLUMNS
    missing = set(keep_cols) - set(df.columns)
    if missing:
        raise ValueError(f"core raw missing columns: {sorted(missing)}")
    # 主线只保留 config.py 中登记的核心列，避免无关字段混入后续面板。
    df = df[keep_cols].copy()
    df, _ = clean_and_validate_raw_table(df, "core_raw")
    return df


def load_market_raw(asset: str) -> pd.DataFrame:
    """读取指定资产的市场指标原始表，并按主线口径筛列校验。"""

    if asset not in ASSET_SPECS:
        raise KeyError(f"unknown asset {asset}")
    path = RAW_DIR / RAW_MARKET_FILE_NAME
    sheet_name = ASSET_SPECS[asset]["sheet_name"]
    # 市场指标文件按资产分 sheet 存放。
    # 新增资产时，除了更新 config.py，还要确保原始 Excel 中存在对应 sheet。
    df = pd.read_excel(path, sheet_name=sheet_name, header=5)
    rename_map = {df.columns[0]: "Date"}
    if "DMI_2" in df.columns:
        # 原始表里这列有时叫 DMI_2，主线统一改回 DMI，避免后续出现两种列名。
        rename_map["DMI_2"] = "DMI"
    df = df.rename(columns=rename_map)
    keep_cols = ["Date"] + MARKET_SOURCE_COLUMNS
    missing = set(keep_cols) - set(df.columns)
    if missing:
        raise ValueError(f"market raw missing columns for {asset}: {sorted(missing)}")
    # 这里只保留第 01 步真正会参与合并的市场指标列。
    df = df[keep_cols].copy()
    df, _ = clean_and_validate_raw_table(df, f"market_raw::{asset}")
    return df


def missing_stats(df: pd.DataFrame) -> pd.DataFrame:
    """按列统计缺失个数和缺失率。"""

    total_rows = len(df)
    records = []
    for column in df.columns:
        # 第 01 步会把这张统计表单独落盘，用于核对原始合并后各列的缺失情况。
        missing_count = int(df[column].isna().sum())
        records.append(
            {
                "column_name": column,
                "missing_count": missing_count,
                "missing_rate": missing_count / total_rows if total_rows else 0.0,
            }
        )
    return pd.DataFrame(records).sort_values(["missing_rate", "column_name"], ascending=[False, True])

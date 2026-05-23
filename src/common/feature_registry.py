from __future__ import annotations

from dataclasses import dataclass

# 本模块负责：定义第 02 步使用的衍生特征、预测目标、时间切分和标准面板列结构。
# 直接服务的步骤：第 02 步直接调用；第 03、04、06、08 步继续消费这里定义出的字段结构。
# 关键修改位置：衍生变量构造函数、目标变量定义、时间切分规则、`FEATURE_PANEL_X_COLUMNS` 与 `MAINLINE_MODEL_FEATURES`。
# 变更后重跑起点：第 02 步。字段、目标或切分变化会影响筛选输入、建模输入、训练结果和图表结果。

import numpy as np
import pandas as pd

from src.config import (
    CANONICAL_MAINLINE_CONTROL_COLUMNS,
    FEATURE_PANEL_DROP_COLUMNS,
    FIXED_TRADEFLOW_COLUMNS,
    HORIZONS,
    SCREENING_CANDIDATE_COLUMNS,
)


# 这组变量会在 `build_derived_features()` 中从原始列进一步构造出来。
# 新增衍生控制变量时，通常需要：
# 1）先在 `build_derived_features()` 里写出构造公式；
# 2）再把变量名补到这里；
# 3）根据用途决定是否加入 `SCREENING_CANDIDATE_COLUMNS` 或正式控制变量清单。
DERIVED_FEATURE_COLUMNS = [
    "cred",
    "liqu",
    "ITVvar_x_dolsha",
    "log_return_lag1",
    "log_return_abs_lag1",
    "log_return_lag5",
    "log_return_abs_lag5",
    "log_return_lag22",
    "log_return_abs_lag22",
    "vol_h5d",
    "mom_h5d",
    "vol_h22d",
    "mom_h22d",
]

# `feature_panel.csv` 中保留的全部候选特征列，顺序固定，便于后续复现实验。
# 第 02 步写出的 feature_panel.csv 就严格按这里的顺序保留列。
FEATURE_PANEL_X_COLUMNS = [
    "IND_SECTOR_TV_ene_norm",
    "INS_SECTOR_TV_ene_norm",
    "ITVvar",
    "sigpre",
    "sigpre30",
    "dolsha30",
    "dolsha",
    "iVX",
    "volume",
    "amt",
    "turn",
    "MACD",
    "RSI",
    "OBV",
    "BIAS",
    "BOLL",
    "PVT",
    "DMI",
    "cred",
    "liqu",
    "ITVvar_x_dolsha",
    "log_return_lag1",
    "log_return_abs_lag1",
    "log_return_lag5",
    "log_return_abs_lag5",
    "log_return_lag22",
    "log_return_abs_lag22",
    "vol_h5d",
    "mom_h5d",
    "vol_h22d",
    "mom_h22d",
]

# 论文主线最终建模的默认变量口径：4 个交易流 + 固定控制变量。
# 第 08 步在没有额外指定变量文件时，会围绕这套逻辑准备正式建模输入。
MAINLINE_MODEL_FEATURES = FIXED_TRADEFLOW_COLUMNS + CANONICAL_MAINLINE_CONTROL_COLUMNS


@dataclass(frozen=True)
class SplitBoundaries:
    train_end_date: pd.Timestamp
    valid_end_date: pd.Timestamp


def build_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """在原始面板上补出主线衍生特征。"""
    out = df.copy()
    # 宏观/利差类衍生变量。
    out["cred"] = out["FirmBondAA10Y"] - out["ChBond10Y"]
    out["liqu"] = out["R6M"] - out["ChBond3M"]
    out["ITVvar_x_dolsha"] = out["ITVvar"] * out["dolsha"]

    # 收益、波动率与动量全部基于收盘价序列构造。
    # 这里保留 1 日、5 日、22 日三档历史收益与波动信息，是为了兼顾短期行为、周度动量和月度波动特征。
    daily_log_return = np.log(out["close"]).diff()
    out["log_return_lag1"] = daily_log_return
    out["log_return_abs_lag1"] = daily_log_return.abs()
    out["log_return_lag5"] = daily_log_return.shift(4)
    out["log_return_abs_lag5"] = daily_log_return.shift(4).abs()
    out["log_return_lag22"] = daily_log_return.shift(21)
    out["log_return_abs_lag22"] = daily_log_return.shift(21).abs()
    out["vol_h5d"] = daily_log_return.rolling(window=5, min_periods=5).std()
    out["vol_h22d"] = daily_log_return.rolling(window=22, min_periods=22).std()
    out["mom_h5d"] = np.log(out["close"] / out["close"].shift(5))
    out["mom_h22d"] = np.log(out["close"] / out["close"].shift(22))
    return out


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """为全部预测期限补出收益目标和二分类目标。"""
    out = df.copy()
    for horizon in HORIZONS:
        target_return_col = f"target_return_{horizon}d"
        target_label_col = f"target_label_{horizon}d"
        # 目标收益定义为从 t 到 t+h 的累计对数收益。
        out[target_return_col] = np.log(out["close"].shift(-horizon) / out["close"])
        # 当前主线分类任务统一预测“未来累计收益是否为正”。
        # 如果要改成其他分类目标形式，例如更高阈值的涨跌方向，就应在这里修改目标定义，
        # 然后从第 02 步开始重跑后续全部主线步骤。
        out[target_label_col] = (out[target_return_col] > 0).astype("float64")
        out.loc[out[target_return_col].isna(), target_label_col] = np.nan
    return out


def trim_to_longest_horizon(df: pd.DataFrame) -> pd.DataFrame:
    """按最长 88 日目标裁样，保证所有期限目标都完整可用。"""
    # 先按最长预测期限裁样，后面的 1/22/44/66 日目标自然都不会再缺未来值。
    trimmed = df.loc[df["target_return_88d"].notna() & df["target_label_88d"].notna()].copy()
    for horizon in HORIZONS:
        if trimmed[f"target_return_{horizon}d"].isna().any() or trimmed[f"target_label_{horizon}d"].isna().any():
            raise ValueError(f"target columns still contain NaN after 88d trim for horizon {horizon}")
    return trimmed


def assign_time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, SplitBoundaries]:
    """按时间顺序切分 train、valid、test。"""
    if df.empty:
        raise ValueError("cannot split empty dataframe")
    n_dates = len(df)
    # 当前主线固定采用 8:1:1 的时间顺序切分，不打乱样本。
    # 修改切分比例时，应从这里调整，并从第 02 步开始重跑，
    # 因为 `feature_panel.csv`、`screening_ready_panel.csv`、第 08 步 prepared 文件和后续训练结果都会跟着变化。
    train_count = int(n_dates * 0.8)
    valid_count = int(n_dates * 0.1)
    test_count = n_dates - train_count - valid_count
    if min(train_count, valid_count, test_count) <= 0:
        raise ValueError(f"invalid split sizes: {train_count}, {valid_count}, {test_count}")
    train_end_date = df.iloc[train_count - 1]["Date"]
    valid_end_date = df.iloc[train_count + valid_count - 1]["Date"]
    out = df.copy()
    out["split"] = "test"
    out.loc[out["Date"] <= train_end_date, "split"] = "train"
    out.loc[(out["Date"] > train_end_date) & (out["Date"] <= valid_end_date), "split"] = "valid"
    return out, SplitBoundaries(train_end_date=train_end_date, valid_end_date=valid_end_date)


def build_feature_panel(df: pd.DataFrame) -> pd.DataFrame:
    """整理出主线统一使用的单资产特征面板。"""
    keep_cols = [
        "Date",
        "asset_alias",
        "asset_code",
        *FEATURE_PANEL_X_COLUMNS,
        *[f"target_return_{h}d" for h in HORIZONS],
        *[f"target_label_{h}d" for h in HORIZONS],
        "split",
    ]
    missing = set(keep_cols) - set(df.columns)
    if missing:
        raise ValueError(f"feature panel missing columns: {sorted(missing)}")
    out = df[keep_cols].copy()
    # 保留显式列数校验，避免新增/漏删字段后悄悄污染下游实验。
    # 新增或删除变量后，这里的预期列数也要同步修改，否则会被当成口径不一致直接拦住。
    expected_col_count = 45
    if out.shape[1] != expected_col_count:
        raise ValueError(f"feature_panel expected {expected_col_count} columns, got {out.shape[1]}")
    return out


def build_screening_ready_panel(feature_panel: pd.DataFrame) -> pd.DataFrame:
    """从特征面板中抽出 pooled screening 需要的字段。"""
    keep_cols = [
        "Date",
        "asset_alias",
        "asset_code",
        "split",
        *FIXED_TRADEFLOW_COLUMNS,
        *SCREENING_CANDIDATE_COLUMNS,
        "target_label_1d",
    ]
    out = feature_panel[keep_cols].copy()
    # 该面板只用于“统一长面板筛选参考”，不包含多期限目标和多余说明列。
    # 统一筛选默认只比较 1 日涨跌标签，因此这里固定保留 target_label_1d。
    if out.shape[1] != 36:
        raise ValueError(f"screening_ready_panel expected 36 columns, got {out.shape[1]}")
    return out


def feature_desc_stats(feature_panel: pd.DataFrame) -> pd.DataFrame:
    """输出单资产特征描述统计表。"""
    exclude_cols = {"Date", "asset_alias", "asset_code", "split"}
    exclude_cols.update({f"target_return_{h}d" for h in HORIZONS})
    exclude_cols.update({f"target_label_{h}d" for h in HORIZONS})
    numeric_cols = [
        col
        for col in feature_panel.columns
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(feature_panel[col])
    ]
    return feature_panel[numeric_cols].describe().T.reset_index().rename(columns={"index": "feature_name"})


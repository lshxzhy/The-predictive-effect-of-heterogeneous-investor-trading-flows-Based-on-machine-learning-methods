import argparse
from pathlib import Path

import pandas as pd

from config import AssetRuntimeContext, get_asset_context


def load_core_data(file_path: Path, ctx: AssetRuntimeContext) -> pd.DataFrame:
    """读取旧公共变量表。"""
    df = pd.read_excel(file_path)
    df = df.loc[:, ["Date", *ctx.config.core_source_cols]].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    df = df.drop_duplicates(subset=["Date"]).reset_index(drop=True)
    return df


def load_market_sheet(
    file_path: Path,
    sheet_name: str,
    ctx: AssetRuntimeContext,
) -> pd.DataFrame:
    """读取并清洗单个资产的补充行情表。"""
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=5)
    df.columns = [str(col).strip() for col in df.columns]
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = df.rename(columns={df.columns[0]: "Date"})
    if "DMI_2" in df.columns:
        df = df.rename(columns={"DMI_2": "DMI"})

    df = df.loc[:, ["Date", *ctx.config.market_source_cols]].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    df = df.drop_duplicates(subset=["Date"]).reset_index(drop=True)

    for col in ctx.config.market_source_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.loc[:, ["Date", *ctx.config.market_source_cols]]


def build_merged_raw_panel(
    core_df: pd.DataFrame,
    asset_df: pd.DataFrame,
    ctx: AssetRuntimeContext,
) -> pd.DataFrame:
    """按日期合并公共变量和单个资产行情。"""
    merged_df = core_df.merge(asset_df, on="Date", how="left")
    merged_df["asset_alias"] = ctx.asset_alias
    merged_df["asset_code"] = ctx.asset_spec.code

    expected_cols = [
        "Date",
        *ctx.config.core_source_cols,
        *ctx.config.market_source_cols,
        "asset_alias",
        "asset_code",
    ]
    return merged_df.loc[:, expected_cols].sort_values("Date").reset_index(drop=True)


def save_merged_raw_panel(merged_df: pd.DataFrame, ctx: AssetRuntimeContext) -> None:
    """保存原始合并层。"""
    merged_df.to_csv(
        ctx.paths.merged_raw_panel_file(),
        index=False,
        encoding="utf-8-sig",
    )


def build_raw_missing_stats(
    merged_df: pd.DataFrame,
    ctx: AssetRuntimeContext,
) -> pd.DataFrame:
    """汇总原始合并层的缺失情况。"""
    rows: list[dict[str, object]] = []
    total_rows = len(merged_df)

    for col in merged_df.columns:
        null_count = int(merged_df[col].isna().sum())
        is_turn_col = col in ctx.config.raw_diagnostic_cols
        rows.append(
            {
                "column_name": col,
                "non_null_count": int(total_rows - null_count),
                "null_count": null_count,
                "null_ratio": null_count / total_rows if total_rows else pd.NA,
                "diagnostic_role": "turn" if is_turn_col else "general",
            }
        )

    missing_df = pd.DataFrame(rows)
    missing_df["diagnostic_priority"] = missing_df["diagnostic_role"].map(
        {"turn": 0, "general": 1}
    )
    missing_df = missing_df.sort_values(
        by=["diagnostic_priority", "null_count", "column_name"],
        ascending=[True, False, True],
    )
    return missing_df.drop(columns=["diagnostic_priority"]).reset_index(drop=True)


def save_raw_missing_stats(merged_df: pd.DataFrame, ctx: AssetRuntimeContext) -> None:
    """保存原始缺失诊断表。"""
    build_raw_missing_stats(merged_df, ctx).to_csv(
        ctx.paths.raw_missing_stats_file(),
        index=False,
        encoding="utf-8-sig",
    )


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True)
    return parser.parse_args()


def main() -> None:
    """执行单个资产的原始合并和缺失诊断。"""
    args = parse_args()
    ctx = get_asset_context(asset_alias=args.asset)

    core_df = load_core_data(ctx.paths.core_raw_file, ctx)
    asset_df = load_market_sheet(
        ctx.paths.extra_raw_file,
        ctx.asset_spec.sheet_name,
        ctx=ctx,
    )

    merged_df = build_merged_raw_panel(core_df, asset_df, ctx)
    ctx.paths.ensure_asset_dirs()
    save_merged_raw_panel(merged_df, ctx)
    save_raw_missing_stats(merged_df, ctx)

    print("原始合并层和缺失诊断已保存：")
    print(ctx.paths.merged_raw_panel_file())
    print(ctx.paths.raw_missing_stats_file())
    print(f"当前资产：{ctx.asset_alias}")
    print(f"原始合并层形状：{merged_df.shape}")


if __name__ == "__main__":
    main()

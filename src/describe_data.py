from pathlib import Path
import pandas as pd


def iqr_outlier_count(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return ((series < lower) | (series > upper)).sum()


def main():
    project_root = Path(__file__).resolve().parent.parent
    raw_file = project_root / "data" / "raw" / "核心变量时间序列日度.xlsx"
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)

    df = pd.read_excel(raw_file)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    raw_feature_cols = [
        "ITVvar",
        "sigpre",
        "sigpre30",
        "dolsha",
        "dolsha30",
        "qzenergy",
        "qzenergy_abs",
        "qzenergy_garch",
        "iVX",
        "FirmBondAA10Y",
        "ChBond10Y",
        "ChBond3M",
        "R6M",
    ]

    stats = df[raw_feature_cols].describe(
        percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    ).T
    stats["skew"] = df[raw_feature_cols].skew()
    stats["kurt"] = df[raw_feature_cols].kurt()

    outlier_summary = pd.DataFrame(
        {
            "outlier_count_iqr": [iqr_outlier_count(df[col]) for col in raw_feature_cols],
        },
        index=raw_feature_cols,
    )

    quantile_summary = pd.DataFrame(
        {
            "p01": df[raw_feature_cols].quantile(0.01),
            "p99": df[raw_feature_cols].quantile(0.99),
        }
    )

    stats_path = output_dir / "descriptive_stats.csv"
    outlier_path = output_dir / "outlier_summary.csv"
    quantile_path = output_dir / "quantile_summary.csv"

    stats.to_csv(stats_path, encoding="utf-8-sig")
    outlier_summary.to_csv(outlier_path, encoding="utf-8-sig")
    quantile_summary.to_csv(quantile_path, encoding="utf-8-sig")

    print(stats)
    print("\n")
    print(outlier_summary)
    print("\nSaved files:")
    print(stats_path)
    print(outlier_path)
    print(quantile_path)


if __name__ == "__main__":
    main()

from pathlib import Path
import pandas as pd


NORMALIZED_COLS = [
    "IND_SECTOR_TV_ene_norm",
    "INS_SECTOR_TV_ene_norm",
]


def load_data(file_path):
    df = pd.read_excel(file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def build_target_and_features(df):
    df = df.copy()

    df["y"] = (df["qzenergy"] > 0).astype(int)
    df["cred"] = df["FirmBondAA10Y"] - df["ChBond10Y"]
    df["liqu"] = df["R6M"] - df["ChBond3M"]

    df = df.drop(columns=["FirmBondAA10Y", "ChBond10Y", "ChBond3M", "R6M"])
    return df


def train_test_split_by_time(df, train_ratio=0.8):
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def zscore_with_train_stats(train_df, test_df, cols):
    train_df = train_df.copy()
    test_df = test_df.copy()

    stats_rows = []

    for col in cols:
        mean_val = train_df[col].mean()
        std_val = train_df[col].std()

        if std_val == 0:
            continue

        train_df[col] = (train_df[col] - mean_val) / std_val
        test_df[col] = (test_df[col] - mean_val) / std_val

        stats_rows.append(
            {"feature": col, "train_mean": mean_val, "train_std": std_val}
        )

    stats_df = pd.DataFrame(stats_rows)
    return train_df, test_df, stats_df


def main():
    project_root = Path(__file__).resolve().parent.parent
    raw_file = project_root / "data" / "raw" / "核心变量时间序列日度.xlsx"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(raw_file)
    df = build_target_and_features(df)

    train_df, test_df = train_test_split_by_time(df, train_ratio=0.8)

    zscore_cols = [
        col for col in train_df.columns
        if col not in ["Date", "y"] + NORMALIZED_COLS
    ]

    train_df, test_df, scaling_stats = zscore_with_train_stats(
        train_df, test_df, zscore_cols
    )

    train_df.to_csv(processed_dir / "train_prepared.csv", index=False)
    test_df.to_csv(processed_dir / "test_prepared.csv", index=False)
    scaling_stats.to_csv(processed_dir / "scaling_stats.csv", index=False)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print("\nTrain target distribution:")
    print(train_df["y"].value_counts(normalize=True))
    print("\nTest target distribution:")
    print(test_df["y"].value_counts(normalize=True))
    print("\nSaved files:")
    print(processed_dir / "train_prepared.csv")
    print(processed_dir / "test_prepared.csv")
    print(processed_dir / "scaling_stats.csv")


if __name__ == "__main__":
    main()

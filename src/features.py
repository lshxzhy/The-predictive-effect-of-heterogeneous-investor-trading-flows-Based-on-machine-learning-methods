from pathlib import Path
import pandas as pd


def build_lag_features(df, lag=1):
    """为除日期和标签外的所有特征构造滞后项。"""
    df = df.copy()

    feature_cols = [col for col in df.columns if col not in ["Date", "y"]]

    for col in feature_cols:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)

    lag_cols = [f"{col}_lag{lag}" for col in feature_cols]
    df = df[["Date", "y"] + lag_cols].dropna().reset_index(drop=True)

    return df


def main():
    """读取预处理后的训练集和测试集，并生成可建模的滞后特征数据。"""
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"

    train_file = processed_dir / "train_prepared.csv"
    test_file = processed_dir / "test_prepared.csv"

    train_df = pd.read_csv(train_file, parse_dates=["Date"])
    test_df = pd.read_csv(test_file, parse_dates=["Date"])

    train_lagged = build_lag_features(train_df, lag=1)
    test_lagged = build_lag_features(test_df, lag=1)

    train_lagged.to_csv(processed_dir / "train_model_input.csv", index=False)
    test_lagged.to_csv(processed_dir / "test_model_input.csv", index=False)

    print("Train lagged shape:", train_lagged.shape)
    print("Test lagged shape:", test_lagged.shape)
    print("\nTrain columns:")
    print(train_lagged.columns.tolist())


if __name__ == "__main__":
    main()

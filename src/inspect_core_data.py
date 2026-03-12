from pathlib import Path
import pandas as pd


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_file = project_root / "data" / "raw" / "核心变量时间序列日度.xlsx"

    df = pd.read_excel(data_file)

    print("Shape:")
    print(df.shape)

    print("\nDtypes:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nDate range:")
    print(df["Date"].min(), "to", df["Date"].max())

if __name__ == "__main__":
    main()
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_file = project_root / "data" / "raw" / "核心变量时间序列日度.xlsx"
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)

    df = pd.read_excel(data_file)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    numeric_cols = [col for col in df.columns if col != "Date"]

    for col in numeric_cols:
        plt.figure(figsize=(10, 4))
        plt.plot(df["Date"], df[col])
        plt.title(col)
        plt.xlabel("Date")
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(output_dir / f"{col}_trend.png")
        plt.close()


if __name__ == "__main__":
    main()

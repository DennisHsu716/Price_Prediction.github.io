# /content/drive/MyDrive/Time Series Price Prediction/src/data/features.py
import argparse
import os
import numpy as np
import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # 1) 日期處理與排序
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
        df = df.sort_values("Date").reset_index(drop=True)

    # 2) 轉數值型別（CSV 讀進來可能是字串）
    num_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 3) 選價格欄（優先 Adj Close）
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    if price_col not in df.columns:
        raise ValueError("No price column found. Expected 'Adj Close' or 'Close'.")

    # 4) 基礎特徵
    df = df.dropna(subset=[price_col]).reset_index(drop=True)
    df["Return"] = df[price_col].pct_change(fill_method=None).fillna(0.0)
    df["LogReturn"] = np.log(df[price_col]).diff().fillna(0.0)
    df["MA_5"] = df[price_col].rolling(5).mean()
    df["MA_10"] = df[price_col].rolling(10).mean()
    df["MA_20"] = df[price_col].rolling(20).mean()
    df["Volatility_20"] = df["Return"].rolling(20).std()

    # 5) 目標欄位（明日報酬率）
    df["Target"] = df["Return"].shift(-1)

    # 6) 丟掉因為 rolling/shift 產生的 NA
    df = df.dropna().reset_index(drop=True)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # 讀檔（若沒有 Date 欄就用第一欄）
    try:
        df = pd.read_csv(args.input, parse_dates=["Date"])
    except Exception:
        df = pd.read_csv(args.input)
        if "Date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "Date"})

    df = add_features(df)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"saved features: {args.out} shape: {df.shape}")

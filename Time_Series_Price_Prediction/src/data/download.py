
import argparse, pandas as pd, yfinance as yf

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = yf.download(args.ticker, start=args.start, end=args.end, auto_adjust=False)
    df.to_csv(args.out)
    print("saved:", args.out, "shape:", df.shape)

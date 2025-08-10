
import argparse, pandas as pd, numpy as np
from sklearn.metrics import mean_absolute_error

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--preds", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.truth)
    # 這裡僅示範：用 MA_10 當簡單預測 → 計算 MAE
    y_true = df["Target"].values
    y_pred = df["MA_10"].pct_change().shift(-1).fillna(0.0).values
    mae = mean_absolute_error(y_true, y_pred)
    print("MAE (demo):", mae)

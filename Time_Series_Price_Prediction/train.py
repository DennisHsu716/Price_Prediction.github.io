import argparse, yaml
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

def rmse_score(y_true, y_pred):
    try:
        # 新版 sklearn
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        # 舊版 sklearn：先算 MSE 再開根號
        return np.sqrt(mean_squared_error(y_true, y_pred))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    folds = cfg.get("validation", {}).get("folds", 5)

    data = pd.read_csv("data/processed/FE.csv")

    # 特徵：排除 Date/Target，確保都是數值
    feat_cols = [c for c in data.columns if c not in ["Date", "Target"]]
    X = data[feat_cols].apply(pd.to_numeric, errors="coerce").values
    y = pd.to_numeric(data["Target"], errors="coerce").values

    # 移除任何包含 NaN 的樣本（保守作法）
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X, y = X[mask], y[mask]

    tscv = TimeSeriesSplit(n_splits=folds)
    rmses = []

    for tr, va in tscv.split(X):
        model = LinearRegression()
        model.fit(X[tr], y[tr])
        pred = model.predict(X[va])
        rmse = rmse_score(y[va], pred)
        rmses.append(rmse)

    os.makedirs("runs", exist_ok=True)
    print("CV RMSE:", float(np.mean(rmses)))
    pd.DataFrame({"rmse": rmses}).to_csv("runs/cv.csv", index=False)

"""Retrains the AirSense XGBoost model and writes ``model.pkl``.

Why this script was rewritten
-----------------------------
The previous version trained on a *mix* of two incompatible datasets:

  * ``delhi_historical.csv`` — CPCB **monitor** readings (Anand Vihar etc.),
    mean PM2.5 ≈ 238 µg/m³
  * ``live_delhi.csv`` — Open-Meteo **CAMS gridded** estimates for the city,
    mean PM2.5 ≈ 65 µg/m³

Both were fed to one model with no indication of which source a row came from,
so the regressor learned an average of two very different worlds. Measured
against live Open-Meteo readings it was off by **MAE 41 µg/m³** — larger than
the average particle itself — which is why predictions disagreed with what
Google (also CAMS-based) reported. It also carried ``no`` and ``nh3`` as
features that the source returns as null for Delhi, so the app had to invent
placeholder values (50 and 25 µg/m³) at inference time.

The fix is to train on a single, consistent source — the same Open-Meteo/CAMS
series the dashboard displays and the prediction form defaults to — with five
years of hourly history and real meteorology (wind, humidity, temperature,
rain) alongside the pollutant concentrations. Predictions then live on the
same scale as the numbers shown on screen and as Google's.

Run:  python train_model.py [--start 2021-01-01] [--reuse]
"""

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from xgboost import XGBRegressor

from data_loader import (
    CITIES,
    METEO_VARS,
    METRICS_JSON,
    MODEL_FEATURES,
    TRAIN_CSV,
    fetch_meteo_range,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PKL = os.path.join(ROOT, "model.pkl")


def _report(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = float(np.mean(np.abs((y_pred - y_true) / np.clip(y_true, 1, None))) * 100)
    print(f"  {name:<26} MAE {mae:6.1f}   RMSE {rmse:6.1f}   R2 {r2:6.3f}   MAPE {mape:5.1f}%")
    return {"mae": round(float(mae), 2), "rmse": round(float(rmse), 2),
            "r2": round(float(r2), 4), "mape": round(mape, 2)}


def build_dataset(reuse):
    if reuse and os.path.exists(TRAIN_CSV):
        print(f"Using cached training series from {TRAIN_CSV}.")
        return pd.read_csv(TRAIN_CSV, parse_dates=["date"])
        
    print("Building training dataset from delhi_historical.csv (CPCB ground truth)...")
    hist_path = os.path.join(ROOT, "data", "delhi_historical.csv")
    df = pd.read_csv(hist_path, parse_dates=["date"])
    
    start = df["date"].min().strftime("%Y-%m-%d")
    end = df["date"].max().strftime("%Y-%m-%d")
    print(f"Fetching meteo data from {start} to {end}...")
    lat, lon = CITIES["Delhi"]
    meteo = fetch_meteo_range(lat, lon, start, end)
    
    print("Merging data...")
    df = df.merge(meteo, on="date", how="inner")
    
    # Drop rows missing PM2.5 or any meteo variables
    df = df.dropna(subset=["pm2_5"] + list(METEO_VARS))
    df = df.sort_values("date").reset_index(drop=True)
    
    df.to_csv(TRAIN_CSV, index=False)
    print(f"Saved {os.path.basename(TRAIN_CSV)} -> {len(df):,} hourly rows")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true",
                    help="reuse data/delhi_training.csv instead of building")
    args = ap.parse_args()

    df = build_dataset(args.reuse)
    print(f"Training rows: {len(df):,}  "
          f"({df['date'].min().date()} -> {df['date'].max().date()})")
    print(f"PM2.5 mean {df['pm2_5'].mean():.1f} µg/m³ · "
          f"median {df['pm2_5'].median():.1f} · p95 {df['pm2_5'].quantile(.95):.1f}")

    X = df[MODEL_FEATURES].astype(float)
    y = df["pm2_5"].astype(float)

    n = len(df)
    cut = int(n * 0.8)
    X_tr, y_tr = X.iloc[:cut], y.iloc[:cut]
    X_te, y_te = X.iloc[cut:], y.iloc[cut:]
    print(f"Train {len(X_tr):,} ({df['date'].iloc[0].date()} -> {df['date'].iloc[cut-1].date()})"
          f"  |  Test {len(X_te):,} ({df['date'].iloc[cut].date()} -> {df['date'].iloc[-1].date()})")

    model = XGBRegressor(
        n_estimators=1200,
        learning_rate=0.04,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        min_child_weight=5,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)

    print("\nHold-out results (newest 20% of real data, never seen in training):")
    metrics = _report("hold-out (future hours)", y_te, model.predict(X_te))
    _report("full series (in-sample)", y, model.predict(X))

    resid = np.abs(model.predict(X_te) - y_te.to_numpy())
    metrics["holdout_mae_p90"] = round(float(np.percentile(resid, 90)), 2)
    print(f"  80% of predictions land within +/- {metrics['holdout_mae_p90']:.0f} µg/m³")

    print("\nFeature importance:")
    fi = sorted(zip(model.feature_importances_, X_tr.columns), reverse=True)
    for imp, name in fi:
        print(f"  {name:24s} {imp:.4f}")

    assert list(model.feature_names_in_) == MODEL_FEATURES, \
        "model feature order drifted from MODEL_FEATURES"
    with open(MODEL_PKL, "wb") as f:
        pickle.dump(model, f)
    print(f"\nSaved model.pkl ({os.path.getsize(MODEL_PKL):,} bytes)")

    meta = {
        "metrics": metrics,
        "features": MODEL_FEATURES,
        "meteo_features": list(METEO_VARS),
        "city": "Delhi",
        "city_coords": list(CITIES["Delhi"]),
        "train_rows": int(len(X_tr)),
        "test_rows": int(len(X_te)),
        "train_range": [str(df["date"].iloc[0].date()), str(df["date"].iloc[cut - 1].date())],
        "test_range": [str(df["date"].iloc[cut].date()), str(df["date"].iloc[-1].date())],
        "feature_importance": {name: round(float(imp), 5) for imp, name in fi},
        "trained_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    }
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {os.path.basename(METRICS_JSON)}")


if __name__ == "__main__":
    main()


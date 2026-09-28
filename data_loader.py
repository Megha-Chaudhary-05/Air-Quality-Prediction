"""Shared data loading and Air Quality Index helpers for the AirSense app.

Pure functions (no Streamlit dependency) so they can be reused by scripts
and wrapped in ``st.cache_data`` inside the app.

Upgraded data pipeline:
  * 18 major Indian cities (real coordinates)
  * 92 days of real historical air-quality data for Delhi (Open-Meteo API)
  * extra real-world variables every hour: NO, dust, UV index,
    European AQI and US AQI (on top of PM2.5, PM10, CO, NO2, SO2, O3)
"""

import os
import json
import time
import urllib.request

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
HIST_CSV = os.path.join(DATA_DIR, "delhi_historical.csv")
LIVE_CSV = os.path.join(DATA_DIR, "live_delhi.csv")
CITIES_CSV = os.path.join(DATA_DIR, "live_cities.csv")
TRAIN_CSV = os.path.join(DATA_DIR, "delhi_training.csv")
METRICS_JSON = os.path.join(DATA_DIR, "model_metrics.json")

AIR_API = "https://air-quality-api.open-meteo.com/v1/air-quality"
ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_API = "https://api.open-meteo.com/v1/forecast"

# Meteorology variables used as model inputs. Wind and humidity are what
# actually drive particulate concentrations, so the model needs them to
# predict a *specific* hour rather than just a seasonal guess.
METEO_VARS = ("temperature_2m", "relative_humidity_2m",
              "wind_speed_10m", "precipitation")

CITIES = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Kolkata": (22.5726, 88.3639),
    "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867),
    "Pune": (18.5204, 73.8567),
    "Ahmedabad": (23.0225, 72.5714),
    "Lucknow": (26.8467, 80.9462),
    "Jaipur": (26.9124, 75.7873),
    "Kanpur": (26.4499, 80.3319),
    "Nagpur": (21.1458, 79.0882),
    "Surat": (21.1702, 72.8311),
    "Patna": (25.5941, 85.1376),
    "Bhopal": (23.2599, 77.4126),
    "Kochi": (9.9312, 76.2673),
    "Chandigarh": (30.7333, 76.7794),
    "Gurugram": (28.4595, 77.0266),
}

# Model features. These are the columns the regressor is actually trained on.
# Every one of them is available from the same Open-Meteo/CAMS source that the
# app displays live, which keeps the model's inputs and the on-screen readings
# on the same scale. NO and NH3 are deliberately excluded: Open-Meteo returns
# ammonia as null for Indian cities and nitric oxide only sporadically, so
# training on them forced the app to invent values (25 / 50 µg/m³) that were
# far outside anything the model had ever seen at inference time.
MODEL_FEATURES = [
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation",
    "month", "hour", "day", "day_of_week", "is_weekend",
]

# Human labels for the feature-importance chart.
FEATURE_LABELS = {
    "temperature_2m": "Temperature",
    "relative_humidity_2m": "Humidity", "wind_speed_10m": "Wind speed",
    "precipitation": "Precipitation", "month": "Month", "hour": "Hour",
    "day": "Day", "day_of_week": "Day of week", "is_weekend": "Weekend",
}

# Pollutant columns that go into the CPCB AQI sub-index calculation.
AQI_POLLUTANTS = ["pm2_5", "pm10", "co", "no2", "so2", "o3"]

POLLUTANTS = {
    "pm2_5": "PM2.5",
    "pm10": "PM10",
    "co": "CO",
    "no2": "NO2",
    "so2": "SO2",
    "o3": "O3",
    "nh3": "NH3",
}

# Extra real-world variables pulled live from Open-Meteo (labels for the UI).
LIVE_VARIABLES = {
    "no": "Nitric oxide (NO)",
    "dust": "Dust (µg/m³)",
    "uv_index": "UV index",
    "eu_aqi": "European AQI",
    "us_aqi": "US AQI",
}

# CPCB / NAAQS AQI breakpoints and sub-index values (24-hr averages).
BREAKPOINTS = {
    "pm2_5": {"bp": [0, 30, 60, 90, 120, 250, 380], "aqi": [0, 50, 100, 200, 300, 400, 500]},
    "pm10": {"bp": [0, 50, 100, 250, 350, 430, 600], "aqi": [0, 50, 100, 200, 300, 400, 500]},
    "co": {"bp": [0, 1.0, 2.0, 10.0, 17.0, 34.0, 50.0], "aqi": [0, 50, 100, 200, 300, 400, 500]},
    "no2": {"bp": [0, 40, 80, 180, 280, 400, 800], "aqi": [0, 50, 100, 200, 300, 400, 500]},
    "so2": {"bp": [0, 40, 80, 380, 800, 1600, 2200], "aqi": [0, 50, 100, 200, 300, 400, 500]},
    "o3": {"bp": [0, 50, 100, 168, 208, 748, 1000], "aqi": [0, 50, 100, 200, 300, 400, 500]},
}

CATEGORIES = [
    {"label": "Good", "min": 0, "max": 50, "color": "#1AAB40", "advice": "Air quality is considered satisfactory, air pollution poses little or no risk. Enjoy outdoor activities."},
    {"label": "Satisfactory", "min": 51, "max": 100, "color": "#7CCB40", "advice": "Acceptable air quality. Sensitive people may experience minor discomfort. Outdoor activity is generally fine."},
    {"label": "Moderate", "min": 101, "max": 200, "color": "#F5C714", "advice": "May cause breathing discomfort for people with lung disease, heart disease, children and older adults."},
    {"label": "Poor", "min": 201, "max": 300, "color": "#FF7A00", "advice": "May cause breathing discomfort for people on prolonged exposure. Reduce prolonged outdoor exertion."},
    {"label": "Very Poor", "min": 301, "max": 400, "color": "#E31A1C", "advice": "Causes respiratory illness on prolonged exposure. Wear a mask and avoid outdoor activity."},
    {"label": "Severe", "min": 401, "max": 500, "color": "#7E0023", "advice": "Affects healthy people and seriously impacts those with existing diseases. Avoid outdoor activity completely."},
]

# WHO 2021 24-hr guideline levels used in some panels.
WHO_24H = {
    "pm2_5": 15.0,
    "pm10": 45.0,
    "co": 4000.0,     # 4 mg/m3 expressed in µg/m³
    "no2": 25.0,
    "so2": 40.0,
    "o3": 100.0,
}


def add_time_features(df, date_col="date"):
    dt = pd.to_datetime(df[date_col])
    out = df.copy()
    out[date_col] = dt
    out["year"] = dt.dt.year
    out["month"] = dt.dt.month
    out["day"] = dt.dt.day
    out["hour"] = dt.dt.hour
    out["day_of_week"] = dt.dt.dayofweek
    out["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    return out


def load_historical():
    """Real recorded Delhi readings (Nov 2020 – Jan 2023, CPCB sensors)."""
    df = pd.read_csv(HIST_CSV)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_live_snapshot():
    """Locally cached live/forecast Delhi readings from Open-Meteo."""
    df = pd.read_csv(LIVE_CSV)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_cities_snapshot():
    df = pd.read_csv(CITIES_CSV)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _to_float_list(raw):
    return [None if v is None else float(v) for v in raw]


def fetch_open_meteo(lat, lon, name, past_days=92, forecast_days=7,
                     include_extras=True):
    """Downloads live air-quality time series from the free Open-Meteo API.

    Returns observed + forecast rows; forecast rows have NaN concentrations.
    """
    base = "https://air-quality-api.open-meteo.com/v1/air-quality"
    hourly = ("pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone")
    if include_extras:
        hourly += (",nitrogen_monoxide,ammonia,dust,uv_index,"
                   "european_aqi,us_aqi")
    params = (
        f"latitude={lat}&longitude={lon}"
        f"&hourly={hourly}"
        f"&past_days={past_days}"
        f"&forecast_days={forecast_days}&timezone=IST"
    )
    url = f"{base}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "air-sense/2.0"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        raw = json.load(resp)
    h = raw["hourly"]
    df = pd.DataFrame({
        "date": h["time"],
        "pm2_5": _to_float_list(h["pm2_5"]),
        "pm10": _to_float_list(h["pm10"]),
        "co": _to_float_list(h["carbon_monoxide"]),
        "no2": _to_float_list(h["nitrogen_dioxide"]),
        "so2": _to_float_list(h["sulphur_dioxide"]),
        "o3": _to_float_list(h["ozone"]),
    })
    if include_extras:
        df["no"] = _to_float_list(h["nitrogen_monoxide"])
        df["nh3"] = _to_float_list(h["ammonia"])       # null for most Indian cities
        df["dust"] = _to_float_list(h["dust"])
        df["uv_index"] = _to_float_list(h["uv_index"])
        df["eu_aqi"] = _to_float_list(h["european_aqi"])
        df["us_aqi"] = _to_float_list(h["us_aqi"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    df = add_time_features(df)
    df.insert(0, "city", name)
    return df


def refresh_live_delhi(past_days=92, forecast_days=7):
    """Fetches ~3 months of real Delhi air data (observed + 7-day forecast)."""
    df = fetch_open_meteo(CITIES["Delhi"][0], CITIES["Delhi"][1], "Delhi",
                          past_days=past_days, forecast_days=forecast_days)
    df.to_csv(LIVE_CSV, index=False)
    return df


def refresh_live_cities(past_days=10, forecast_days=3):
    frames = []
    for name, (lat, lon) in CITIES.items():
        try:
            frames.append(fetch_open_meteo(lat, lon, name,
                                           past_days=past_days,
                                           forecast_days=forecast_days))
            time.sleep(0.3)
        except Exception:
            continue
    out = pd.concat(frames, ignore_index=True) if frames else load_cities_snapshot()
    out.to_csv(CITIES_CSV, index=False)
    return out


# --------------------------------------------------------------------------
# Training-series / meteorology helpers
# --------------------------------------------------------------------------
def _get_json(url, timeout=180, retries=3):
    """GET a JSON endpoint with a couple of retries for transient failures."""
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "air-sense/2.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except Exception as exc:  # network hiccup / HTTP error
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise last


def fetch_air_quality_range(lat, lon, start, end, name="Delhi"):
    """Hourly air quality for an arbitrary date range from the Open-Meteo API.

    Used to build the model training set. The API happily serves years of
    history, which matters: training on the *same* source that feeds the live
    dashboard is what keeps predictions consistent with the numbers on screen.
    """
    hourly = ("pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,"
              "sulphur_dioxide,ozone,dust,uv_index")
    url = (f"{AIR_API}?latitude={lat}&longitude={lon}&hourly={hourly}"
           f"&start_date={start}&end_date={end}&timezone=IST")
    h = _get_json(url)["hourly"]
    df = pd.DataFrame({
        "date": h["time"],
        "pm2_5": _to_float_list(h["pm2_5"]),
        "pm10": _to_float_list(h["pm10"]),
        "co": _to_float_list(h["carbon_monoxide"]),
        "no2": _to_float_list(h["nitrogen_dioxide"]),
        "so2": _to_float_list(h["sulphur_dioxide"]),
        "o3": _to_float_list(h["ozone"]),
        "dust": _to_float_list(h["dust"]),
        "uv_index": _to_float_list(h["uv_index"]),
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    df.insert(0, "city", name)
    return add_time_features(df)


def fetch_meteo_range(lat, lon, start, end):
    """Hourly meteorology for a date range (archive API, ERA5 reanalysis)."""
    url = (f"{ARCHIVE_API}?latitude={lat}&longitude={lon}"
           f"&hourly={','.join(METEO_VARS)}"
           f"&start_date={start}&end_date={end}&timezone=IST")
    h = _get_json(url)["hourly"]
    df = pd.DataFrame({"date": h["time"]})
    for var in METEO_VARS:
        df[var] = _to_float_list(h.get(var, [None] * len(df)))
    df["date"] = pd.to_datetime(df["date"])
    return df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)


def fetch_meteo_forecast(lat, lon, start, end):
    """Hourly meteorology from the forecast API (recent past + upcoming days)."""
    url = (f"{FORECAST_API}?latitude={lat}&longitude={lon}"
           f"&hourly={','.join(METEO_VARS)}"
           f"&past_days=92&forecast_days=16&timezone=IST")
    h = _get_json(url)["hourly"]
    df = pd.DataFrame({"date": h["time"]})
    for var in METEO_VARS:
        df[var] = _to_float_list(h.get(var, [None] * len(df)))
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].between(pd.Timestamp(start), pd.Timestamp(end))]
    return df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)


def fetch_meteo_for_timestamp(lat, lon, when):
    """Meteorology for one timestamp, used to auto-fill the app's inputs."""
    day = pd.Timestamp(when).strftime("%Y-%m-%d")
    for fetcher in (fetch_meteo_forecast, fetch_meteo_range):
        try:
            df = fetcher(lat, lon, day, day)
        except Exception:
            continue
        if df.empty:
            continue
        hit = df[df["date"] == pd.Timestamp(when).floor("h")]
        if hit.empty:
            hit = df.iloc[[0]]
        row = hit.iloc[0]
        return {v: (float(row[v]) if pd.notna(row[v]) else None) for v in METEO_VARS}
    return {}


def build_training_frame(start="2021-01-01", end=None, name="Delhi",
                         progress=print):
    """Assembles the model training set: air quality + meteorology, hourly.

    Both halves come from Open-Meteo, so the feature distribution matches what
    the app feeds the model at prediction time.
    """
    lat, lon = CITIES[name]
    end = end or pd.Timestamp.today().normalize().strftime("%Y-%m-%d")

    if progress:
        progress(f"  air quality {start} -> {end} ...")
    air = fetch_air_quality_range(lat, lon, start, end, name)

    if progress:
        progress("  meteorology (ERA5 archive) ...")
    try:
        meteo = fetch_meteo_range(lat, lon, start, end)
    except Exception:
        meteo = pd.DataFrame(columns=["date", *METEO_VARS])

    # The archive lags by a few days, so top up the most recent window from
    # the forecast API to keep the series running up to today.
    if not meteo.empty:
        gap_start = (pd.Timestamp(meteo["date"].max()) + pd.Timedelta(hours=1)).strftime("%Y-%m-%d")
        if gap_start <= end:
            if progress:
                progress(f"  meteorology top-up {gap_start} -> {end} ...")
            try:
                extra = fetch_meteo_forecast(lat, lon, gap_start, end)
                meteo = pd.concat([meteo, extra], ignore_index=True)
            except Exception:
                pass
        meteo = meteo.drop_duplicates(subset="date").sort_values("date")

    df = air.merge(meteo, on="date", how="left") if not meteo.empty else air.copy()
    df = df.dropna(subset=["pm2_5", "pm10", "co", "no2", "so2", "o3", "dust"])
    for var in METEO_VARS:
        if var not in df.columns:
            df[var] = np.nan
    df = df.dropna(subset=list(METEO_VARS))
    return df.sort_values("date").reset_index(drop=True)


def load_training_frame():
    """Locally cached training series, if it has been built."""
    if not os.path.exists(TRAIN_CSV):
        return None
    df = pd.read_csv(TRAIN_CSV, parse_dates=["date"])
    return df


def load_model_metrics():
    """Training/validation metrics recorded by ``train_model.py``."""
    if not os.path.exists(METRICS_JSON):
        return None
    with open(METRICS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def split_observed(df, poll_cols):
    """Splits a live dataframe into observed rows and forecast rows.

    Two criteria decide 'observed':
      1. All AQI pollutant columns (that have *any* data) are non-null.
      2. The timestamp is not in the future (>= current local time + 2 h buffer).

    CAMS provides air-quality *forecasts* up to 4 days ahead, and those rows
    arrive with real-looking values rather than NaN.  Without the time filter
    the dashboard would show a future CAMS forecast as today's reading, making
    the "lag" counter go negative and the reported date wrong.
    """
    poll_cols = [c for c in poll_cols
                 if c in df.columns and df[c].notna().any()]
    if not poll_cols:
        mask = pd.Series(False, index=df.index)
    else:
        mask = df[poll_cols].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    # Cap at current wall-clock time so CAMS forecast rows (which arrive
    # with real-looking values up to 4 days ahead) are treated as forecast,
    # not as live observations.  We allow a 30-minute buffer for API latency.
    now_ist = pd.Timestamp.now() + pd.Timedelta(minutes=30)
    if "date" in df.columns:
        future = pd.to_datetime(df["date"]) > now_ist
        mask = mask & ~future
    return df[mask].copy(), df[~mask].copy()


def _sub_index(name, value):
    """Linear CPCB sub-index interpolation for one pollutant.

    The breakpoint table is stored as parallel (concentration, AQI) anchors,
    e.g. PM2.5: (0,0) (30,50) (60,100) (90,200) (120,300) (250,400) (380,500).
    Anything past the top anchor is capped at 500.
    """
    cfg = BREAKPOINTS[name]
    bp, qi = cfg["bp"], cfg["aqi"]
    if value is None or pd.isna(value) or value < 0:
        return None
    if value > bp[-1]:
        return int(qi[-1])
    for i in range(len(bp) - 1):
        if bp[i] <= value <= bp[i + 1]:
            low, high = bp[i], bp[i + 1]
            ilow, ihigh = qi[i], qi[i + 1]
            if high == low:
                return int(ilow)
            return int(round(((ihigh - ilow) / (high - low)) * (value - low) + ilow))
    return int(qi[-1])


def cpcb_aqi(polls):
    """Returns (aqi, sub-indices) from a dict of pollutant concentrations.

    NOTE: the official CPCB breakpoints are defined on **24-hour average**
    concentrations. Feeding this function a single hourly reading produces a
    number that is not a valid AQI — for Delhi it differs from the compliant
    figure by ~70 points on average. Use :func:`cpcb_aqi_24h` for anything the
    user sees. This helper is kept for raw-hour reporting and testing.

    CO is expected in µg/m³ and is converted to mg/m³ internally.
    """
    subs = {}
    for key in AQI_POLLUTANTS:
        val = polls.get(key)
        if val is None or str(val) in ("", "None"):
            continue
        try:
            val = float(val)
        except (TypeError, ValueError):
            continue
        if key == "co":
            val = val / 1000.0  # µg/m³ -> mg/m³
        subs[key] = _sub_index(key, val)
    avail = {k: v for k, v in subs.items() if v is not None}
    if not avail:
        return None, {}
    return max(avail.values()), subs


def rolling_24h(df, at=None, cols=None):
    """Trailing 24-hour mean of each pollutant, as CPCB requires for AQI.

    Returns ``(averages, timestamp)``. ``averages`` only contains columns with
    a full 24 h of data behind them, so callers never silently average a
    partial window (which would bias the index downwards).
    """
    cols = cols or AQI_POLLUTANTS
    cols = [c for c in cols if c in df.columns]
    if df.empty or not cols:
        return {}, at
    frame = df.set_index("date").sort_index()[cols]
    frame = frame.apply(pd.to_numeric, errors="coerce")
    if at is not None:
        window = frame.loc[:pd.Timestamp(at)].tail(24)
    else:
        window = frame.tail(24)
    if window.empty or window.isna().all().all():
        return {}, at
    means = window.mean()
    counts = window.count()
    full = counts >= min(6, int(window.shape[0]))
    avg = {c: float(means[c]) for c in means.index if full[c] and pd.notna(means[c])}
    ts = window.index.max() if not window.empty else at
    return avg, ts


def cpcb_aqi_24h(df, at=None):
    """Compliant CPCB AQI from 24-hour mean concentrations.

    Returns ``(aqi, sub_indices, averages, timestamp)`` — the AQI, the
    per-pollutant sub-indices, the 24-h means they were derived from, and the
    timestamp of the last hour in the window.
    """
    avg, ts = rolling_24h(df, at=at)
    if not avg:
        return None, {}, {}, at
    aqi, subs = cpcb_aqi(avg)
    return aqi, subs, avg, ts


def aqi_category(aqi):
    if aqi is None:
        return None
    for cat in CATEGORIES:
        if cat["min"] <= aqi <= cat["max"]:
            return cat
    return CATEGORIES[-1]


def pm25_categories():
    return [
        {"min": 0, "max": 30, "label": "Clean", "color": "#1AAB40"},
        {"min": 31, "max": 60, "label": "Satisfactory", "color": "#7CCB40"},
        {"min": 61, "max": 90, "label": "Moderate", "color": "#F5C714"},
        {"min": 91, "max": 120, "label": "Poor", "color": "#FF7A00"},
        {"min": 121, "max": 250, "label": "Very Poor", "color": "#E31A1C"},
        {"min": 251, "max": 1e9, "label": "Severe", "color": "#7E0023"},
    ]


def pm25_category(value):
    if value is None or pd.isna(value):
        return None
    for c in pm25_categories():
        if c["min"] <= value <= c["max"]:
            return c
    return None


def color_style(cat_color):
    return (f"background-color:{cat_color}; color:#fff; padding:6px 14px; "
            "border-radius:20px; font-weight:600; display:inline-block;")


def historical_avg(df, col, period_col, period_value):
    sub = df[df[period_col] == period_value]
    if sub.empty:
        return None
    return float(sub[col].mean())


# --------------------------------------------------------------------------
# Helpers for the upgraded dashboard
# --------------------------------------------------------------------------
def latest_city_summary(city_df, col="pm2_5"):
    """Returns the most recent readings for every city in the dataframe."""
    rows = []
    for city, g in city_df.groupby("city"):
        g = g.sort_values("date")
        last = g.iloc[-1]
        vals = {c: (v if (pd.notna(v)) else np.nan)
                for c, v in last.items() if c != "city"}
        rows.append({"city": city, **vals})
    return pd.DataFrame(rows)


def combine_datasets(include_live=True):
    """Historical (2020–23) + live 2026 real data combined for analysis.

    Keeps the extra live-only real variables (NO, dust, UV index, EU/US AQI)
    so the app explorer can chart them alongside the recorded pollutants.
    """
    hist = load_historical()
    frames = [hist]
    if include_live:
        live = load_live_snapshot() if os.path.exists(LIVE_CSV) else None
        if live is None:
            return hist
        obs, _ = split_observed(live, list(POLLUTANTS))
        if not obs.empty:
            cols = ["date", "pm2_5", "pm10", "co", "no2", "so2", "o3",
                    "no", "nh3", "dust", "uv_index", "eu_aqi", "us_aqi"]
            obs = obs[[c for c in cols if c in obs.columns]].copy()
            for c in cols:
                if c not in obs.columns:
                    obs[c] = np.nan
            frames.append(obs)
    df = pd.concat(frames, ignore_index=True)
    ordered = (["date", "pm2_5", "pm10", "co", "no2", "so2", "o3", "nh3"]
               + [c for c in ("no", "dust", "uv_index", "eu_aqi", "us_aqi")
                  if c in df.columns]
               + [c for c in ("year", "month", "day", "hour",
                              "day_of_week", "is_weekend") if c in df.columns])
    df = df[[c for c in ordered if c in df.columns]]
    df = df.sort_values("date").reset_index(drop=True)
    return df
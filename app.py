"""AirSense — Delhi Air Quality Prediction & Live Dashboard (Streamlit).

  * 18 Indian cities with real live readings (Open-Meteo / CAMS)
  * ~3 months of real Delhi history + 7-day forecast
  * XGBoost PM2.5 model trained on 4 years of the *same* source the dashboard
    displays, plus real meteorology (wind, humidity, temperature, rain)
  * CPCB-compliant AQI computed from 24-hour mean concentrations
  * dark theme, dense but balanced layout

Run with:  streamlit run app.py
"""

import pickle
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import (
    AQI_POLLUTANTS,
    CITIES,
    FEATURE_LABELS,
    MODEL_FEATURES,
    POLLUTANTS,
    WHO_24H,
    aqi_category,
    cpcb_aqi,
    cpcb_aqi_24h,
    fetch_meteo_for_timestamp,
    latest_city_summary,
    load_cities_snapshot,
    load_historical,
    load_live_snapshot,
    load_model_metrics,
    pm25_category,
    refresh_live_cities,
    refresh_live_delhi,
    rolling_24h,
    split_observed,
)

st.set_page_config(
    page_title="AirSense · India Air Quality",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Theme & shared style
# ----------------------------------------------------------------------------
BG = "#07080c"
PANEL = "#12141c"
PANEL_2 = "#181b25"
STROKE = "#262b38"
TEXT = "#e8ecf4"
MUTED = "#98a2b8"
ACCENT = "#38bdf8"
ACCENT_2 = "#22c55e"
PLOT_BG = "#0d0f16"
PLOT_GRID = "#232838"

CUSTOM_CSS = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Poppins:wght@500;600;700&display=swap');

  html, body, [class*="css"], .stApp, button, input, select, textarea {{
      font-family: 'Inter', 'Poppins', sans-serif;
  }}

  /* ---- page background ---- */
  .stApp {{ background: {BG}; color: {TEXT}; }}
  html[data-testid="stAppViewContainer"] > .main {{ background: {BG}; }}
  .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px; }}
  [data-testid="stHeader"] {{ background: transparent; }}

  ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
  ::-webkit-scrollbar-track {{ background: {BG}; }}
  ::-webkit-scrollbar-thumb {{ background: #2b3140; border-radius: 8px; }}
  ::-webkit-scrollbar-thumb:hover {{ background: #3b4356; }}

  /* ---- headings ---- */
  h1, h2, h3, h4, h5 {{ color: {TEXT}; letter-spacing: -0.01em; }}
  h1 {{ font-size: 1.9rem !important; }}
  h2 {{ font-size: 1.5rem !important; }}
  h3 {{ font-size: 1.15rem !important; }}
  h4 {{ font-size: 1rem !important; }}
  p, span, label, li, td {{ color: {TEXT}; }}
  [data-testid="stCaptionContainer"] p,
  .stCaption, small {{ color: {MUTED} !important; }}

  /* ---- hero ---- */
  .hero {{
      background: linear-gradient(115deg, #0b2a3d 0%, #0d3b2e 55%, #1a1030 100%);
      border: 1px solid {STROKE};
      border-left: 5px solid {ACCENT};
      border-radius: 16px; padding: 1.4rem 1.7rem; margin-bottom: 1.4rem;
  }}
  .hero h1 {{ color: #fff; font-size: 1.75rem; margin: 0; font-weight: 700; }}
  .hero p {{ color: #b8c6dd; margin: 0.35rem 0 0 0; font-size: 0.92rem; }}

  /* ---- cards / panels ---- */
  .card {{
      background: {PANEL}; border: 1px solid {STROKE};
      border-radius: 14px; padding: 0.95rem 1.1rem;
  }}
  .panel {{
      background: {PANEL}; border: 1px solid {STROKE};
      border-radius: 14px; padding: 1rem 1.1rem; height: 100%;
  }}
  .metric-tile {{
      background: linear-gradient(160deg, {PANEL_2} 0%, {PANEL} 100%);
      border: 1px solid {STROKE}; border-radius: 12px;
      padding: 0.75rem 0.6rem; text-align: center; height: 100%;
  }}
  .metric-tile .v {{ font-size: 1.5rem; font-weight: 800; line-height: 1.15; }}
  .metric-tile .l {{ color: {MUTED}; font-size: 0.68rem; text-transform: uppercase;
                    letter-spacing: 0.06em; margin-top: 0.2rem; }}

  /* ---- badges ---- */
  .badge {{ padding: 3px 12px; border-radius: 20px; color: #0b0b0b;
            font-weight: 700; font-size: 0.76rem; display: inline-block; }}
  .chip {{ padding: 2px 9px; border-radius: 20px; color: #0b0b0b;
           font-size: 0.7rem; font-weight: 700; display: inline-block; }}
  .soft-chip {{
      padding: 3px 10px; border-radius: 20px; font-size: 0.7rem; font-weight: 600;
      background: {PANEL_2}; border: 1px solid {STROKE}; color: {MUTED};
      display: inline-block;
  }}

  /* ---- section headers with accent rule ---- */
  .sec {{ display: flex; align-items: center; gap: 0.55rem; margin: 1.4rem 0 0.55rem 0; }}
  .sec .bar {{ width: 4px; height: 1.05rem; border-radius: 3px;
               background: linear-gradient(180deg, {ACCENT}, {ACCENT_2}); }}
  .sec .t {{ font-size: 1.02rem; font-weight: 700; color: {TEXT}; }}

  /* ---- buttons ---- */
  .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{
      width: 100%;
      background: linear-gradient(135deg, #1b2233 0%, #232b40 100%);
      color: {TEXT};
      border: 1px solid #333c52;
      border-radius: 11px;
      font-weight: 600; font-size: 0.86rem;
      padding: 0.5rem 0.9rem;
      box-shadow: 0 1px 0 rgba(255,255,255,0.04) inset, 0 4px 14px rgba(0,0,0,0.45);
      transition: transform 0.12s ease, box-shadow 0.12s ease, border-color 0.12s ease;
  }}
  .stButton > button:hover, .stDownloadButton > button:hover {{
      border-color: {ACCENT};
      box-shadow: 0 0 0 1px rgba(56,189,248,0.35), 0 6px 20px rgba(0,0,0,0.55);
      transform: translateY(-1px);
      color: #fff;
  }}
  .stButton > button:active {{ transform: translateY(0); }}
  .stButton > button:focus-visible {{ outline: 2px solid {ACCENT}; outline-offset: 2px; }}

  /* primary action */
  .stButton > button[kind="primary"] {{
      background: linear-gradient(135deg, #0284c7 0%, #0ea5e9 55%, #22c55e 160%);
      color: #04121c; border: none; font-weight: 800; font-size: 0.95rem;
      letter-spacing: 0.03em; padding: 0.62rem 1rem;
      box-shadow: 0 4px 18px rgba(14,165,233,0.32);
  }}
  .stButton > button[kind="primary"]:hover {{
      box-shadow: 0 6px 26px rgba(34,197,94,0.38);
      color: #04121c;
  }}
  /* keep the hero predict button compact instead of full-bleed */
  div.predict-cta {{ max-width: 340px; margin: 0.9rem auto 0.2rem auto; }}

  /* ---- inputs ---- */
  [data-baseweb="slider"] div[role="slider"] {{ background: {ACCENT} !important; }}
  [data-testid="stSlider"] [data-baseweb="slider"] div[aria-label] {{ box-shadow: none; }}

  [data-testid="stExpander"] {{
      border: 1px solid {STROKE}; border-radius: 12px; background: {PANEL};
  }}
  [data-testid="stExpander"] summary {{ color: {TEXT}; font-weight: 600; }}

  .stTabs [data-baseweb="tab-list"] {{ gap: 0.3rem; border-bottom: 1px solid {STROKE}; }}
  .stTabs [data-baseweb="tab"] {{
      background: transparent; color: {MUTED}; border-radius: 9px 9px 0 0;
      padding: 0.42rem 0.85rem; font-size: 0.86rem; font-weight: 600;
  }}
  .stTabs [aria-selected="true"] {{ color: {ACCENT} !important; }}
  .stTabs [data-baseweb="tab-highlight"] {{ background: {ACCENT}; }}
  .stTabs [data-baseweb="tab-border"] {{ background: {ACCENT}; }}

  /* ---- dataframes ---- */
  [data-testid="stDataFrame"] {{
      border: 1px solid {STROKE}; border-radius: 12px; overflow: hidden;
  }}

  /* ---- sidebar ---- */
  [data-testid="stSidebar"] {{
      background: #0b0d13; border-right: 1px solid {STROKE};
  }}
  [data-testid="stSidebar"] .stRadio label {{
      padding: 0.3rem 0.5rem; border-radius: 9px; font-size: 0.88rem; font-weight: 500;
  }}
  [data-testid="stSidebar"] [data-baseweb="radio"] circle {{ stroke: {ACCENT}; }}

  /* ---- alerts ---- */
  .stAlert {{ border-radius: 12px; border: 1px solid {STROKE}; background: {PANEL}; }}
  [data-testid="stAlertInfo"] {{ background: rgba(56,189,248,0.07); border-color: rgba(56,189,248,0.3); }}
  [data-testid="stAlertSuccess"] {{ background: rgba(34,197,94,0.08); border-color: rgba(34,197,94,0.3); }}
  [data-testid="stAlertWarning"] {{ background: rgba(245,199,20,0.08); border-color: rgba(245,199,20,0.3); }}
  [data-testid="stAlertError"] {{ background: rgba(227,26,28,0.09); border-color: rgba(227,26,28,0.35); }}

  /* ---- metrics ---- */
  [data-testid="stMetric"] {{
      background: {PANEL}; border: 1px solid {STROKE};
      border-radius: 13px; padding: 0.75rem 0.9rem;
  }}
  [data-testid="stMetricLabel"] p {{ color: {MUTED}; font-size: 0.74rem;
                                    text-transform: uppercase; letter-spacing: 0.05em; }}
  [data-testid="stMetricValue"] {{ font-size: 1.6rem; font-weight: 800; color: {TEXT}; }}

  /* number inputs / date-time pickers on dark */
  .stDateInput input, .stTimeInput input, .stNumberInput input {{
      background: {PANEL_2} !important; border-color: {STROKE} !important; color: {TEXT} !important;
  }}
  div[data-baseweb="select"] > div {{ background: {PANEL_2}; border-color: {STROKE}; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def section(title, icon=""):
    """A compact section header with a coloured accent bar."""
    st.markdown(
        f'<div class="sec"><span class="bar"></span>'
        f'<span class="t">{icon} {title}</span></div>', unsafe_allow_html=True)


def hero(title, subtitle):
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True)


def badge(text, color):
    return f'<span class="badge" style="background-color:{color}">{text}</span>'


def chip(text, color):
    return f'<span class="chip" style="background-color:{color}">{text}</span>'


def metric_tile(value, label, color=TEXT):
    return (f'<div class="metric-tile"><div class="v" style="color:{color}">{value}</div>'
            f'<div class="l">{label}</div></div>')


def tile_row(tiles):
    """Renders a row of metric tiles spread across the full width."""
    for col, (value, label, color) in zip(st.columns(len(tiles)), tiles):
        with col:
            st.markdown(metric_tile(value, label, color), unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Plotly theme
# ----------------------------------------------------------------------------
def style_figure(fig, height=380, title=None, showlegend=True, y_title=None):
    """Applies the dark theme to any Plotly figure and tidies the layout."""
    fig.update_layout(
        template=None,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT, family="Inter, Poppins, sans-serif", size=12),
        title=dict(text=title, font=dict(size=15, color=TEXT),
                   x=0.01, xanchor="left", y=0.97, yanchor="top") if title else None,
        showlegend=showlegend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        margin=dict(l=10, r=10, t=52 if title else 26, b=10),
        hoverlabel=dict(bgcolor=PANEL_2, bordercolor=STROKE, font=dict(color=TEXT, size=12)),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=PLOT_GRID, zerolinecolor=PLOT_GRID, linecolor=STROKE,
                     tickfont=dict(color=MUTED), gridwidth=1, showline=True)
    fig.update_yaxes(gridcolor=PLOT_GRID, zerolinecolor=PLOT_GRID, linecolor=STROKE,
                     tickfont=dict(color=MUTED), gridwidth=1, showline=True,
                     title=dict(text=y_title, font=dict(color=MUTED, size=11)) if y_title else None)
    return fig



AQI_BAND_COLORS = ["#1AAB40", "#7CCB40", "#F5C714", "#FF7A00", "#E31A1C", "#7E0023"]


def empty(name):
    return np.nan if pd.isna(name) else name


# ----------------------------------------------------------------------------
# Cached assets
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def cached_metrics():
    return load_model_metrics()


@st.cache_data(show_spinner="Loading recorded Delhi readings …")
def cached_historical():
    return load_historical()


@st.cache_data(ttl=1800, show_spinner="Fetching live air-quality data …")
def cached_live():
    try:
        return refresh_live_delhi(past_days=92, forecast_days=7)
    except Exception:
        return load_live_snapshot()



@st.cache_data(ttl=3600, show_spinner="Fetching multi-city readings …")
def cached_cities():
    try:
        return refresh_live_cities(past_days=5, forecast_days=3)
    except Exception:
        return load_cities_snapshot()



@st.cache_data(ttl=86400, show_spinner=False)
def cached_meteo(ts):
    """Meteorology for one timestamp (auto-fills the prediction form)."""
    lat, lon = CITIES["Delhi"]
    try:
        return fetch_meteo_for_timestamp(lat, lon, ts)
    except Exception:
        return {}


# ----------------------------------------------------------------------------
# Plotting helpers
# ----------------------------------------------------------------------------
def aqi_band_figure(aqi):
    """Horizontal CPCB AQI colour band with a marker at the current value."""
    edges = [0, 50, 100, 200, 300, 400, 500]
    fig = go.Figure()
    for i, c in enumerate(AQI_BAND_COLORS):
        fig.add_shape(type="rect", x0=edges[i], x1=edges[i + 1], y0=0, y1=1,
                      line=dict(width=0), fillcolor=c, layer="below")
    if aqi is not None:
        fig.add_trace(go.Scatter(
            x=[min(max(aqi, 0), 500)], y=[0.5], mode="markers",
            marker=dict(color="#ffffff", size=15, symbol="triangle-down",
                        line=dict(color="#0b0b0b", width=1.5)),
            hovertemplate=f"AQI {aqi}<extra></extra>", name=""))
    fig.update_xaxes(range=[-10, 510], showgrid=False, showline=False,
                     zeroline=False, tickvals=edges,
                     tickfont=dict(color="#cfd6e4", size=11))
    fig.update_yaxes(range=[0, 1], visible=False)
    fig.update_layout(height=96, margin=dict(l=8, r=8, t=4, b=4),
                      showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT))
    return fig


def aqi_gauge(aqi, cat, subtitle=None):
    """Semicircular AQI gauge (CPCB scale 0–500)."""
    color = cat["color"] if cat else "#999999"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=min(max(aqi if aqi is not None else 0, 0), 500),
        number={"suffix": "", "font": {"size": 42, "color": color}},
        title={"text": subtitle or "CPCB AQI", "font": {"size": 12, "color": MUTED}},
        gauge={
            "axis": {"range": [0, 500], "tickwidth": 1, "tickcolor": "#4b5265",
                     "tickfont": {"size": 10, "color": MUTED}, "ticks": "outside"},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [{"range": edges, "color": c} for edges, c in zip(
                [[0, 50], [50, 100], [100, 200], [200, 300], [300, 400], [400, 500]],
                AQI_BAND_COLORS)],
            "threshold": {"line": {"color": "#ffffff", "width": 3},
                          "thickness": 0.85, "value": aqi or 0},
        }))
    fig.update_layout(height=270, margin=dict(l=28, r=28, t=30, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=TEXT, family="Inter, sans-serif"))
    return fig


def pollutant_radar(subs, aqi_labels=None):
    """Radar of sub-index contribution of each pollutant to the AQI."""
    names = {"pm2_5": "PM2.5", "pm10": "PM10", "co": "CO",
             "no2": "NO2", "so2": "SO2", "o3": "O3"}
    vals, thetas = [], []
    for k, label in names.items():
        v = subs.get(k) if subs else None
        if v is not None:
            thetas.append(label)
            vals.append(v)
    if not thetas or len(thetas) < 3:
        return None
    fig = go.Figure(go.Scatterpolar(
        r=vals + vals[:1], theta=thetas + thetas[:1], fill="toself",
        line=dict(color=ACCENT, width=2),
        fillcolor="rgba(56,189,248,0.18)",
        hovertemplate="%{theta}: sub-index %{r}<extra></extra>"))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 500],
                            tickvals=[100, 200, 300, 400, 500],
                            gridcolor=PLOT_GRID, linecolor=STROKE,
                            tickfont=dict(color=MUTED, size=10)),
            angularaxis=dict(gridcolor=PLOT_GRID, linecolor=STROKE,
                             tickfont=dict(color=TEXT, size=12))),
        height=320, margin=dict(l=40, r=40, t=30, b=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, font=dict(color=TEXT, family="Inter, sans-serif"))
    return fig


def sub_index_table(subs, avgs=None):
    """Table of pollutant → 24-h mean → sub-index → band."""
    rows = []
    for k in AQI_POLLUTANTS:
        v = subs.get(k) if subs else None
        if v is None:
            continue
        band = aqi_category(v)
        rows.append({
            "Pollutant": POLLUTANTS[k],
            "24-h mean": (f"{avgs[k]:,.1f}" if avgs and k in avgs else "—"),
            "Sub-index": f"{v}",
            "Band": band["label"] if band else "—",
        })
    return pd.DataFrame(rows)


def line_chart(df, y_cols, title, y_title="µg/m³", color_map=None, dash=None):
    fig = px.line(df, x="date", y=y_cols, color_discrete_map=color_map)
    fig.update_traces(line=dict(width=2.2))
    if dash:
        fig.update_traces(line=dict(dash="dash"))
    fig.update_layout(hoverlabel_namelength=-1)
    return style_figure(fig, height=370, title=title, y_title=y_title)


def features_importance():
    model = load_model()
    fi = model.feature_importances_
    names = list(getattr(model, "feature_names_in_", MODEL_FEATURES))
    df = pd.DataFrame({"feature": names, "importance": fi})
    df["feature"] = df["feature"].map(lambda n: FEATURE_LABELS.get(n, n))
    return df.sort_values("importance")


def model_prediction(inputs):
    """Runs the regressor on an ordered dict of MODEL_FEATURES."""
    model = load_model()
    row = [[float(inputs[f]) for f in MODEL_FEATURES]]
    val = float(model.predict(np.array(row, dtype=float))[0])
    return max(val, 0.0)


def model_confidence():
    """(half-width, label) of the model's reported accuracy band."""
    m = cached_metrics() or {}
    holdout = (m.get("metrics") or {}).get("holdout_mae_p90")
    if holdout is None:
        return None, "no recorded metrics"
    return float(holdout), f"±{holdout:.0f} µg/m³ (80% of cases)"


# ----------------------------------------------------------------------------
# Shared live-state helper
# ----------------------------------------------------------------------------
def observed_frame():
    """The observed (non-forecast) Delhi readings, ordered oldest → newest."""
    live = cached_live()
    obs, _ = split_observed(live, AQI_POLLUTANTS)
    if obs.empty:
        return pd.DataFrame(), live
    obs = obs.sort_values("date").reset_index(drop=True)
    return obs, live


def latest_observation(obs):
    """The most recent hour that has *every* AQI pollutant reported.

    Using max(date) alone is unsafe: Open-Meteo occasionally leaves a single
    variable null in the newest hour, which used to push the app's "current"
    card back to a stale reading.
    """
    if obs.empty:
        return None, None
    complete = obs[AQI_POLLUTANTS].notna().all(axis=1)
    usable = obs[complete]
    if usable.empty:
        return None, None
    row = usable.iloc[-1]
    return row.to_dict(), row["date"]


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
def _predict_inputs_from_row(row, when):
    """Best-effort feature dict from a live observation row."""
    if not row:
        return None
    ts = pd.Timestamp(when)
    meteo = cached_meteo(ts) or {}
    g = lambda k, fb: (float(row[k]) if row.get(k) is not None
                       and pd.notna(row.get(k)) else fb)
    return {
        "pm10": g("pm10", 180.0),
        "co": g("co", 900.0),
        "no2": g("no2", 45.0),
        "o3": g("o3", 60.0),
        "so2": g("so2", 20.0),
        "dust": g("dust", 60.0),
        "temperature_2m": meteo.get("temperature_2m", 25.0),
        "relative_humidity_2m": meteo.get("relative_humidity_2m", 55.0),
        "wind_speed_10m": meteo.get("wind_speed_10m", 8.0),
        "precipitation": meteo.get("precipitation", 0.0),
        "month": ts.month, "hour": ts.hour, "day": ts.day,
        "day_of_week": ts.dayofweek, "is_weekend": int(ts.dayofweek >= 5),
    }


def page_predict():
    hero("🌱 Predict Air Quality",
         "Feed in the pollutant readings for an hour and the model estimates PM2.5. "
         "The overall CPCB AQI is then derived from 24-hour mean concentrations, "
         "the way the official index is defined.")

    obs, _ = observed_frame()
    row, last_ts = latest_observation(obs)

    # ---- current conditions banner -------------------------------------
    if row is not None:
        live_pm25 = float(row["pm2_5"])
        pm_cat = pm25_category(live_pm25)
        aqi_now, subs_now, avg_now, _ = cpcb_aqi_24h(obs, at=last_ts)
        cat_now = aqi_category(aqi_now)
        lag = pd.Timestamp.now() - pd.Timestamp(last_ts)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(metric_tile(f"{live_pm25:,.0f}", "PM2.5 this hour (µg/m³)",
                                    pm_cat["color"] if pm_cat else TEXT),
                        unsafe_allow_html=True)
        with c2:
            st.markdown(metric_tile(f"{aqi_now}" if aqi_now is not None else "—",
                                    "CPCB AQI (24-h basis)",
                                    cat_now["color"] if cat_now else TEXT),
                        unsafe_allow_html=True)
        with c3:
            st.markdown(metric_tile(f"{avg_now.get('pm2_5', float('nan')):,.0f}"
                                    if avg_now else "—",
                                    "PM2.5 24-h mean (µg/m³)", ACCENT),
                        unsafe_allow_html=True)
        with c4:
            driver = max(subs_now.items(), key=lambda kv: kv[1])[0] if subs_now else None
            st.markdown(metric_tile(
                f'<span class="soft-chip">{POLLUTANTS.get(driver, "—")}</span>',
                "Driving pollutant", ACCENT_2), unsafe_allow_html=True)
        st.caption(
            f"Live source: Open-Meteo / CAMS · observation stamped "
            f"{pd.Timestamp(last_ts).strftime('%d %b %Y, %H:%M IST')} "
            f"({lag.total_seconds() / 3600:.1f} h ago) · hourly resolution.")
        st.markdown("")

    # ---- input form -----------------------------------------------------
    section("Date, time & meteorology", "🕒")
    c_d, c_t, c_m = st.columns([2, 2, 8])
    with c_d:
        d = st.date_input("Date", value=(last_ts.date() if last_ts else date.today()),
                          key="in_date")
    with c_t:
        _default_time = last_ts.time() if last_ts else datetime.strptime("12:00", "%H:%M").time()
        t = st.time_input("Hour", value=_default_time, key="in_time")
    ts = pd.Timestamp(datetime.combine(d, t))
    with c_m:
        meteo = cached_meteo(ts) or {}
        chips = []
        for var, unit in (("temperature_2m", "°C"),
                          ("relative_humidity_2m", "%"),
                          ("wind_speed_10m", "km/h"),
                          ("precipitation", "mm")):
            val = meteo.get(var)
            shown = f"{val:.0f} {unit}" if val is not None else "n/a"
            chips.append(f'<span class="soft-chip">{FEATURE_LABELS[var]}: {shown}</span>')
        st.markdown(" ".join(chips) + '<div style="height:.35rem"></div>',
                    unsafe_allow_html=True)
        st.caption("Weather is pulled automatically for the selected hour — "
                   "wind and humidity are what drive particle levels.")

    values = {}
    c_a, c_b, c_c = st.columns([4, 4, 4])
    with c_a:
        values["temperature_2m"] = st.number_input(
            "Temperature (°C)", -10.0, 55.0,
            round(float(meteo.get("temperature_2m", 25.0)), 1), step=0.5,
            key="in_temp")
    with c_b:
        values["relative_humidity_2m"] = st.number_input(
            "Humidity (%)", 0.0, 100.0,
            round(float(meteo.get("relative_humidity_2m", 55.0)), 1), step=1.0,
            key="in_rh")
    with c_c:
        w1, w2 = st.columns(2)
        with w1:
            values["wind_speed_10m"] = st.number_input(
                "Wind (km/h)", 0.0, 120.0,
                round(float(meteo.get("wind_speed_10m", 8.0)), 1), step=0.5,
                key="in_wind")
        with w2:
            values["precipitation"] = st.number_input(
                "Rain (mm)", 0.0, 100.0,
                round(float(meteo.get("precipitation", 0.0)), 1), step=0.5,
                key="in_rain")

    values.update({
        "month": ts.month, "hour": ts.hour, "day": ts.day,
        "day_of_week": ts.dayofweek, "is_weekend": int(ts.dayofweek >= 5),
    })

    # ---- compact centred action button ---------------------------------
    st.markdown('<div class="predict-cta">', unsafe_allow_html=True)
    run = st.button("🚀  PREDICT PM2.5 & AQI", type="primary",
                    use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if not run:
        st.markdown("---")
        section("How the number is produced", "🧠")
        m = cached_metrics() or {}
        met = m.get("metrics") or {}
        _, conf = model_confidence()
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            st.markdown(metric_tile(f"{met.get('mae', '—')}", "Model MAE (µg/m³)", ACCENT),
                        unsafe_allow_html=True)
        with cc2:
            st.markdown(metric_tile(f"{met.get('r2', '—')}", "Model R²", ACCENT_2),
                        unsafe_allow_html=True)
        with cc3:
            st.markdown(metric_tile(f"{m.get('train_rows', '—'):,}"
                                    if m.get("train_rows") else "—",
                                    "Training hours", TEXT),
                        unsafe_allow_html=True)
        st.caption(
            "Trained on the same Open-Meteo/CAMS series the dashboard shows, so the "
            "estimate is on the same scale as the readings above and as Google's "
            f"Delhi figure. Typical error: {conf}.")
        fi = features_importance()
        fig = px.bar(fi.sort_values("importance"), x="importance", y="feature",
                     orientation="h", color="importance",
                     color_continuous_scale=[[0, "#0e7490"], [1, "#22c55e"]])
        fig.update_layout(coloraxis_showscale=False)
        style_figure(fig, height=max(240, 26 * len(fi)), title=None, showlegend=False)
        fig.update_layout(xaxis=dict(title="Relative importance", gridcolor=PLOT_GRID,
                                     linecolor=STROKE, tickfont=dict(color=MUTED)),
                          yaxis=dict(gridcolor="rgba(0,0,0,0)", linecolor="rgba(0,0,0,0)",
                                     tickfont=dict(color=TEXT, size=11)),
                          margin=dict(l=10, r=10, t=14, b=10))
        st.plotly_chart(fig, use_container_width=True)
        return

    pred_pm25 = model_prediction(values)
    pm_cat = pm25_category(pred_pm25)

    # CPCB sub-indices are defined on 24-hour mean concentrations.
    # We combine the past 23 hours of real PM2.5 readings with this 1 hour prediction.
    hour_row = {"date": ts, "pm2_5": pred_pm25}
    combined = pd.DataFrame([hour_row])
    if not obs.empty:
        basis = obs[obs["date"] < ts].tail(23)
        if not basis.empty:
            combined = pd.concat([basis[["date", "pm2_5"]], combined], ignore_index=True)
    avg, _ = rolling_24h(combined, at=ts, cols=["pm2_5"])
    aqi, subs = cpcb_aqi(avg)
    cat = aqi_category(aqi)

    st.markdown("---")
    section("Prediction result", "🎯")

    c_gauge, c_score = st.columns([5, 7])
    with c_gauge:
        st.plotly_chart(aqi_gauge(aqi, cat), use_container_width=True)
        st.plotly_chart(aqi_band_figure(aqi), use_container_width=True)
    with c_score:
        p1, p2, p3 = st.columns(3)
        with p1:
            st.markdown(
                f'<div class="panel" style="text-align:center">'
                f'<div style="font-size:.7rem;letter-spacing:.06em;color:{MUTED};'
                f'text-transform:uppercase">Predicted PM2.5</div>'
                f'<div style="font-size:2.5rem;font-weight:800;line-height:1.1;'
                f'color:{pm_cat["color"] if pm_cat else TEXT}">{pred_pm25:,.0f}'
                f'<span style="font-size:.9rem;color:{MUTED}"> µg/m³</span></div>'
                f'<div style="margin-top:.35rem">'
                f'{badge(pm_cat["label"], pm_cat["color"]) if pm_cat else ""}</div>'
                f'</div>', unsafe_allow_html=True)
        with p2:
            st.markdown(
                f'<div class="panel" style="text-align:center">'
                f'<div style="font-size:.7rem;letter-spacing:.06em;color:{MUTED};'
                f'text-transform:uppercase">Overall CPCB AQI</div>'
                f'<div style="font-size:2.5rem;font-weight:800;line-height:1.1;'
                f'color:{cat["color"] if cat else TEXT}">'
                f'{aqi if aqi is not None else "—"}</div>'
                f'<div style="margin-top:.35rem">'
                f'{badge(cat["label"], cat["color"]) if cat else ""}</div></div>',
                unsafe_allow_html=True)
        with p3:
            _, conf = model_confidence()
            st.markdown(
                f'<div class="panel" style="text-align:center">'
                f'<div style="font-size:.7rem;letter-spacing:.06em;color:{MUTED};'
                f'text-transform:uppercase">Confidence</div>'
                f'<div style="font-size:1.5rem;font-weight:800;line-height:1.6;'
                f'color:{ACCENT}">{conf.split(" ")[0]}</div>'
                f'<div style="font-size:.7rem;color:{MUTED};margin-top:.3rem">'
                f'80% of predictions</div></div>', unsafe_allow_html=True)

        half, _ = model_confidence()
        st.markdown(
            f'<div class="card" style="margin-top:.7rem">'
            f'<b style="color:{cat["color"] if cat else ACCENT}">'
            f'{cat["label"] if cat else "—"}</b> — {cat["advice"] if cat else ""}'
            f'</div>', unsafe_allow_html=True)
        if half:
            st.caption(
                f"The estimate carries roughly ±{half:.0f} µg/m³ of uncertainty. "
                "PM2.5 alone dominates the AQI, and the AQI is reported on the "
                "CPCB 24-hour-average basis rather than from this single hour.")

    section("Which pollutant set the index", "🧩")
    c_tbl, c_radar = st.columns([7, 5])
    with c_tbl:
        tbl = sub_index_table(subs, avg)
        st.dataframe(tbl, hide_index=True, use_container_width=True,
                     height=min(320, 60 + 36 * max(len(tbl), 1)))
        st.caption("Sub-index values use the official CPCB 24-h breakpoints. "
                   "The overall AQI is the worst sub-index.")
    with c_radar:
        radar = pollutant_radar(subs)
        if radar is not None:
            st.plotly_chart(radar, use_container_width=True)

    # ---- context -------------------------------------------------------
    hist = cached_historical()
    if hist is not None and not hist.empty:
        section("Context", "📐")
        month = ts.month
        hist_avg = hist[hist["month"] == month]["pm2_5"].mean()
        hour_avg = (hist[(hist["month"] == month) & (hist["hour"] == ts.hour)]["pm2_5"].mean())
        if pd.notna(hist_avg) and hist_avg:
            delta = pred_pm25 - hist_avg
            pct = delta / hist_avg * 100
            colour = ACCENT_2 if delta < 0 else "#FF7A00"
            sign_txt = "cleaner" if delta < 0 else "dirtier"
        else:
            delta, pct, colour, sign_txt = None, None, MUTED, ""

        tiles = [
            (f"{pred_pm25:,.0f}", "Predicted PM2.5 (µg/m³)", TEXT),
            (f"{hist_avg:,.0f}" if pd.notna(hist_avg) else "—",
             f"Recorded monthly mean ({ts.strftime('%b')})", ACCENT),
            (f"{hour_avg:,.0f}" if pd.notna(hour_avg) else "—",
             f"Recorded mean for {ts.strftime('%H:%M')} hours", ACCENT_2),
        ]
        if delta is not None:
            tiles.append((f"{delta:+,.0f} ({pct:+.0f}%)", f"{sign_txt} than usual", colour))
        tile_row(tiles)
        st.caption("Recorded baseline: CPCB Delhi monitor data, Nov 2020 – Jan 2023. "
                   "Note this sits on a different scale to the CAMS-derived live "
                   "readings used for the prediction — treat it as seasonal context, "
                   "not a like-for-like target.")

    if row is not None and pd.notna(row.get("pm2_5")):
        actual = float(row["pm2_5"])
        st.info(
            f"Sanity check — the live CAMS sensor reported **{actual:,.0f} µg/m³** "
            f"at {pd.Timestamp(last_ts).strftime('%d %b %H:%M')}. "
            f"Model estimate {pred_pm25:,.0f} µg/m³ "
            f"({'within' if abs(pred_pm25 - actual) <= max(20, 0.5 * actual) else 'outside'} "
            "the model's usual error band).")


def page_live():
    hero("📡 Live Air Quality — Delhi & 18 Indian cities",
         "Real-time readings from the Open-Meteo / CAMS model. 92 days of "
         "Delhi history, a 7-day forecast, and the current hour for 18 cities.")

    # ---- compact toolbar ------------------------------------------------
    t1, t2, t3 = st.columns([2, 2, 8])
    with t1:
        refreshed = st.button("🔄  Refresh", use_container_width=True)
    with t2:
        stale = st.button("⚡  Clear cache", use_container_width=True)
    with t3:
        st.caption("Readings refresh hourly; the cache is dropped after 30 minutes.")

    if refreshed or stale:
        with st.spinner("Contacting Open-Meteo API …"):
            try:
                if refreshed:
                    refresh_live_delhi(past_days=92, forecast_days=7)
                    refresh_live_cities(past_days=10, forecast_days=3)
                cached_live.clear()
                cached_cities.clear()
                st.success("Live data updated.")
            except Exception as exc:
                st.error(f"Could not reach the API: {exc}")

    obs, live = observed_frame()
    if obs.empty:
        st.warning("No live readings available. Showing the last saved snapshot.")
        obs, _ = observed_frame()
        if obs.empty:
            st.stop()

    row, last_ts = latest_observation(obs)
    aqi, subs, avgs, aqi_ts = cpcb_aqi_24h(obs, at=last_ts)
    cat = aqi_category(aqi)
    pm_now = float(row["pm2_5"]) if row else np.nan
    pm_cat = pm25_category(pm_now)
    lag_h = (pd.Timestamp.now() - pd.Timestamp(last_ts)).total_seconds() / 3600

    # ---- headline row ---------------------------------------------------
    c_g, c_tiles = st.columns([4, 8])
    with c_g:
        st.plotly_chart(aqi_gauge(aqi, cat, "CPCB AQI · 24-h average"),
                        use_container_width=True)
    with c_tiles:
        st.markdown(
            f'<div class="card" style="margin-bottom:.55rem">'
            f'<span style="color:{MUTED};font-size:.72rem;letter-spacing:.06em;'
            f'text-transform:uppercase">Delhi · latest complete hour</span><br>'
            f'<b style="font-size:1rem">{pd.Timestamp(last_ts).strftime("%A, %d %b %Y · %H:%M IST")}</b><br>'
            f'<span style="color:{MUTED};font-size:.78rem">'
            f'{lag_h:.1f} h ago · CAMS gridded estimate</span></div>',
            unsafe_allow_html=True)
        tiles = [
            ("PM2.5", pm_now, pm_cat["color"] if pm_cat else TEXT),
            ("PM10", row.get("pm10"), ACCENT),
            ("CO", row.get("co"), "#c084fc"),
            ("NO2", row.get("no2"), ACCENT_2),
            ("SO2", row.get("so2"), "#fbbf24"),
            ("O3", row.get("o3"), "#38bdf8"),
        ]
        cols = st.columns(6)
        for col, (name, val, colour) in zip(cols, tiles):
            with col:
                shown = f"{val:,.0f}" if val is not None and pd.notna(val) else "—"
                st.markdown(metric_tile(shown, name, colour), unsafe_allow_html=True)

        st.markdown('<div style="height:.4rem"></div>', unsafe_allow_html=True)
        extra = [("Dust", row.get("dust")), ("UV index", row.get("uv_index")),
                 ("US AQI", row.get("us_aqi")), ("EU AQI", row.get("eu_aqi")),
                 ("PM2.5 24-h", avgs.get("pm2_5") if avgs else None),
                 ("NO", row.get("no"))]
        cols = st.columns(6)
        for col, (name, val) in zip(cols, extra):
            with col:
                if val is not None and pd.notna(val):
                    fmt = f"{val:,.1f}" if name in ("UV index",) else f"{val:,.0f}"
                else:
                    fmt = "—"
                st.markdown(metric_tile(fmt, name, MUTED), unsafe_allow_html=True)

    st.markdown('<div style="height:.6rem"></div>', unsafe_allow_html=True)
    st.plotly_chart(aqi_band_figure(aqi), use_container_width=True)
    st.caption("The band marker is the 24-hour-average AQI. The CPCB breakpoints "
               "are defined on 24-hour means, so an instantaneous hourly reading "
               "mapped straight onto them is not a valid index.")

    if cat:
        st.markdown(
            f'<div class="card" style="border-left:4px solid {cat["color"]}">'
            f'<b style="color:{cat["color"]}">{cat["label"]}</b> — {cat["advice"]}'
            f'</div>', unsafe_allow_html=True)

    # ---- tabs -----------------------------------------------------------
    tab_now, tab_sub, tab_fc, tab_cities = st.tabs(
        ["🕐 Last 48 hours", "🧩 Sub-indices", "🔮 7-day forecast", "🏙️ 18 cities"])

    with tab_now:
        recent = obs[obs["date"] >= last_ts - timedelta(hours=48)]
        fig = line_chart(recent, ["pm2_5", "pm10"],
                         "PM2.5 & PM10 — last 48 hours",
                         color_map={"pm2_5": "#E31A1C", "pm10": "#FF7A00"})
        fig.add_hline(y=60, line_dash="dot", line_color="#7CCB40",
                      annotation_text="CPCB 24-h limit 60",
                      annotation_position="top left",
                      annotation_font_color="#7CCB40")
        st.plotly_chart(fig, use_container_width=True)

        g1, g2 = st.columns(2)
        with g1:
            fig2 = line_chart(recent, ["co", "no2", "so2", "o3"],
                              "Gaseous pollutants — last 48 hours",
                              color_map={"co": "#c084fc", "no2": "#1AAB40",
                                         "so2": "#F5C714", "o3": "#38bdf8"})
            st.plotly_chart(fig2, use_container_width=True)
        with g2:
            met_cols = [c for c in ("dust", "uv_index") if c in recent.columns]
            if met_cols:
                fig3 = line_chart(recent, met_cols,
                                  "Dust & UV index — last 48 hours",
                                  y_title="mixed units",
                                  color_map={"dust": "#B45309", "uv_index": "#7C3AED"})
                st.plotly_chart(fig3, use_container_width=True)

        with st.expander("🔍 Raw hourly readings (last 24 h)"):
            show = ["date", "pm2_5", "pm10", "co", "no2", "so2", "o3"] + \
                [c for c in ("dust", "uv_index", "us_aqi", "eu_aqi")
                 if c in recent.columns]
            st.dataframe(recent.tail(24)[show].set_index("date").round(1),
                         use_container_width=True)

    with tab_sub:
        c1, c2 = st.columns([6, 6])
        with c1:
            tbl = sub_index_table(subs, avgs)
            st.dataframe(tbl, hide_index=True, use_container_width=True)
            st.caption(f"24-hour window ending {pd.Timestamp(aqi_ts).strftime('%d %b %H:%M')} "
                       "(AQI_POLLUTANTS 24-h means → CPCB breakpoints).")
        with c2:
            radar = pollutant_radar(subs)
            if radar is not None:
                st.plotly_chart(radar, use_container_width=True)
            driver = max(subs.items(), key=lambda kv: kv[1])[0] if subs else None
            if driver:
                st.info(f"**{POLLUTANTS[driver]}** is setting the index right now "
                        f"(sub-index {subs[driver]}).")

    with tab_fc:
        _, fcast = split_observed(live, AQI_POLLUTANTS)
        if fcast.empty:
            st.info("No forecast window published yet — try refreshing.")
        else:
            fc = fcast[fcast["date"] > last_ts]
            if fc.empty:
                st.info("The API currently publishes no hours beyond the last "
                        "observation. Try refreshing in a little while.")
            else:
                daily = fc.set_index("date")[["pm2_5", "pm10"]].resample("D").mean()
                daily = daily.dropna(subset=["pm2_5"]).reset_index()
                if not daily.empty:
                    fig = line_chart(daily, ["pm2_5", "pm10"],
                                     "Daily mean forecast — PM2.5 & PM10",
                                     color_map={"pm2_5": "#E31A1C", "pm10": "#FF7A00"})
                    fig.add_hline(y=60, line_dash="dot", line_color="#7CCB40",
                                  annotation_text="safe limit 60",
                                  annotation_position="bottom right",
                                  annotation_font_color="#7CCB40")
                    st.plotly_chart(fig, use_container_width=True)

                    p25 = daily["pm2_5"].to_numpy(dtype=float)
                    p10 = daily["pm10"].to_numpy(dtype=float)
                    rows = pd.DataFrame({
                        "Date": daily["date"].dt.date,
                        "PM2.5 (µg/m³)": pd.Series(p25).round().astype("Int64"),
                        "PM10 (µg/m³)": pd.Series(p10).round().astype("Int64"),
                        "Level": [pm25_category(v)["label"] if pm25_category(v) else "—"
                                  for v in p25],
                        "Colour": [pm25_category(v)["color"] if pm25_category(v) else "#999"
                                   for v in p25],
                    })
                    st.dataframe(rows.drop(columns="Colour").set_index("Date"),
                                 use_container_width=True)
                    st.caption("Model output for planning only — not an official "
                               "CPCB forecast.")

    with tab_cities:
        city_view = cached_cities()
        obs_c, _ = split_observed(city_view, AQI_POLLUTANTS)
        if obs_c.empty:
            st.info("No city readings right now.")
        else:
            summary = latest_city_summary(obs_c, "pm2_5")
            geo = pd.DataFrame([{"city": k, "lat": v[0], "lon": v[1]}
                                for k, v in CITIES.items()])
            mdf = summary.merge(geo, on="city", how="left").dropna(subset=["pm2_5"])
            mdf["label"] = mdf["pm2_5"].round(0).astype(int).astype(str)
            mdf["level"] = mdf["pm2_5"].map(
                lambda v: pm25_category(v)["label"] if pm25_category(v) else "—")
            if not mdf.empty:
                c_map, c_bar = st.columns([5, 7])
                with c_map:
                    fig = px.scatter_geo(
                        mdf, lat="lat", lon="lon", text="label", size="pm2_5",
                        color="pm2_5", hover_name="city",
                        hover_data={"pm2_5": True, "level": True,
                                    "lat": False, "lon": False},
                        color_continuous_scale=[(0, "#1AAB40"), (0.5, "#F5C714"),
                                                (0.8, "#E31A1C"), (1, "#7E0023")],
                        range_color=[0, 250], projection="mercator")
                    fig.update_geos(showcountries=True, countrycolor="#2b3140",
                                    bgcolor=PLOT_BG, projection_scale=6.8,
                                    center=dict(lat=22.5, lon=79.5),
                                    showland=True, landcolor="#141824",
                                    showocean=True, oceancolor="#0b0d14",
                                    lakecolor="#0b0d14")
                    fig.update_layout(height=480,
                                      margin=dict(l=0, r=0, t=0, b=0),
                                      paper_bgcolor="rgba(0,0,0,0)",
                                      font=dict(color=TEXT),
                                      coloraxis_colorbar=dict(
                                          title="PM2.5", outlinewidth=0,
                                          tickfont=dict(color=MUTED)))
                    st.plotly_chart(fig, use_container_width=True)
                with c_bar:
                    bar_df = mdf.sort_values("pm2_5")
                    fig2 = px.bar(bar_df, x="city", y="pm2_5", text="label",
                                  color="pm2_5",
                                  color_continuous_scale=[(0, "#1AAB40"),
                                                          (0.5, "#F5C714"),
                                                          (0.8, "#E31A1C"),
                                                          (1, "#7E0023")])
                    style_figure(fig2, height=480, title=None, showlegend=False,
                                 y_title="PM2.5 µg/m³")
                    fig2.update_layout(xaxis_tickangle=-45,
                                       coloraxis_showscale=False)
                    fig2.add_hline(y=60, line_dash="dot", line_color="#1AAB40",
                                   annotation_text="safe limit 60",
                                   annotation_position="top left",
                                   annotation_font_color="#1AAB40")
                    st.plotly_chart(fig2, use_container_width=True)

                st.markdown("#### City snapshot — latest hour")
                view_cols = ["city", "date", "pm2_5", "pm10", "co", "no2", "so2",
                             "o3"] + [c for c in ("dust", "uv_index", "us_aqi", "eu_aqi")
                                      if c in summary.columns]
                tbl = (summary[view_cols].sort_values("pm2_5", ascending=False)
                       .set_index("city").round(1))
                st.dataframe(tbl, use_container_width=True)

                worst = mdf.nlargest(3, "pm2_5")
                best = mdf.nsmallest(3, "pm2_5")
                c_w, c_b = st.columns(2)
                with c_w:
                    st.markdown("**Worst right now**")
                    for _, r in worst.iterrows():
                        c_ = pm25_category(r["pm2_5"])
                        st.markdown(
                            f'{badge(r["city"], c_["color"] if c_ else "#999")} '
                            f'<span style="color:{MUTED}">{r["pm2_5"]:,.0f} µg/m³</span>',
                            unsafe_allow_html=True)
                with c_b:
                    st.markdown("**Cleanest right now**")
                    for _, r in best.iterrows():
                        c_ = pm25_category(r["pm2_5"])
                        st.markdown(
                            f'{badge(r["city"], c_["color"] if c_ else "#999")} '
                            f'<span style="color:{MUTED}">{r["pm2_5"]:,.0f} µg/m³</span>',
                            unsafe_allow_html=True)

                with st.expander("📍 City list & coordinates"):
                    st.dataframe(pd.DataFrame(
                        [{"City": k, "Latitude": v[0], "Longitude": v[1]}
                         for k, v in CITIES.items()]), hide_index=True,
                        use_container_width=True)

    # ---- model cross-check ---------------------------------------------
    with st.expander("🧪 Cross-check: what does the ML model say for this hour?"):
        if row is not None:
            inputs = _predict_inputs_from_row(row, last_ts)
            pred = model_prediction(inputs)
            c_a, c_b, c_c = st.columns(3)
            with c_a:
                st.metric("Model estimate (µg/m³)", f"{pred:,.0f}")
            with c_b:
                st.metric("Live CAMS PM2.5 (µg/m³)", f"{pm_now:,.0f}")
            with c_c:
                half, _ = model_confidence()
                st.metric("Typical model error (µg/m³)", f"{half:,.0f}" if half else "—")
            st.caption("Both numbers come from the same CAMS source, so a small gap "
                       "here is the model doing its job rather than a data mismatch.")


def page_explorer():
    hero("📊 Air Quality Data Explorer",
         "The recorded Delhi dataset (Nov 2020 – Jan 2023) alongside the latest "
         "live CAMS readings. Switch the source on the left to see how the two "
         "compare — they are on different scales, which matters.")

    hist = cached_historical()
    include_live = st.sidebar.checkbox("Include live CAMS data", value=True)

    if include_live:
        from data_loader import combine_datasets
        df = combine_datasets(include_live=True)
    else:
        df = hist.copy()

    if df.empty:
        st.warning("No data available.")
        return

    st.caption(f"**{len(df):,} hourly records** · "
               f"{df['date'].min().date()} → {df['date'].max().date()}")
    st.plotly_chart(aqi_band_figure(None), use_container_width=True)

    all_polts = {**POLLUTANTS}
    if include_live:
        all_polts["dust"] = "Dust"
        all_polts["uv_index"] = "UV index"

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["📈 Trends", "🍂 Seasonality", "📊 Distributions", "🔗 Correlations",
         "🏙️ Cities", "🗃️ Dataset"])

    with tab1:
        pollutants = st.multiselect(
            "Pollutants to plot", list(all_polts),
            default=["pm2_5", "pm10"], format_func=lambda p: all_polts[p])
        if not pollutants:
            st.info("Select at least one pollutant.")
        else:
            plot_cols = [p for p in pollutants if p in df.columns]
            if not plot_cols:
                st.info("No values for the selected pollutant — enable live data.")
            else:
                daily = df.set_index("date")[plot_cols].resample("D").mean().reset_index()
                daily = daily.dropna(subset=plot_cols[:1])
                cmap = {c: px.colors.sample_colorscale(
                    "Turbo", (i + 1) / (len(plot_cols) + 1))[0]
                    for i, c in enumerate(plot_cols)}
                fig = px.line(daily, x="date", y=plot_cols,
                              color_discrete_map=cmap)
                fig.update_traces(line=dict(width=2))
                style_figure(fig, height=430, y_title="µg/m³",
                             title="Daily mean pollutant levels — Delhi")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("A sharp step between 2023 and the live period marks the "
                           "change of source (CPCB monitors → CAMS model), not a "
                           "real change in Delhi's air.")

    with tab2:
        c_seas, c_hour = st.columns(2)
        with c_seas:
            monthly = df.groupby("month")[["pm2_5", "pm10"]].mean().reset_index()
            fig = px.bar(monthly, x="month", y=["pm2_5", "pm10"], barmode="group",
                         color_discrete_map={"pm2_5": "#E31A1C", "pm10": "#FF7A00"})
            style_figure(fig, height=400, y_title="µg/m³",
                         title="Mean by month (Jan–Dec)")
            st.plotly_chart(fig, use_container_width=True)
        with c_hour:
            hourly = df.groupby("hour")[["pm2_5", "pm10"]].mean().reset_index()
            fig = px.line(hourly, x="hour", y=["pm2_5", "pm10"],
                          color_discrete_map={"pm2_5": "#E31A1C", "pm10": "#FF7A00"})
            fig.update_traces(line=dict(width=2.4))
            style_figure(fig, height=400, y_title="µg/m³",
                         title="Mean by hour of day")
            st.plotly_chart(fig, use_container_width=True)

        daily_pm = df.set_index("date")["pm2_5"].resample("D").mean().dropna()
        c_w, c_b = st.columns(2)
        with c_w:
            st.markdown("**10 worst days**")
            st.dataframe(daily_pm.nlargest(10).round(0).to_frame("PM2.5"),
                         use_container_width=True)
        with c_b:
            st.markdown("**10 cleanest days**")
            st.dataframe(daily_pm.nsmallest(10).round(0).to_frame("PM2.5"),
                         use_container_width=True)

    with tab3:
        pol = st.selectbox("Pollutant", list(all_polts), format_func=lambda p: all_polts[p])
        if pol not in df.columns:
            st.info("Enable live data to see this pollutant.")
        else:
            val = df[pol].dropna()
            if val.empty:
                st.info("No values for this pollutant.")
            else:
                c_h, c_b = st.columns(2)
                with c_h:
                    fig = px.histogram(val, x=pol, nbins=70,
                                       color_discrete_sequence=["#0ea5e9"])
                    style_figure(fig, height=370, y_title="Hours",
                                 title=f"{all_polts[pol]} distribution")
                    st.plotly_chart(fig, use_container_width=True)
                with c_b:
                    fig = px.box(df, y=pol, color_discrete_sequence=["#22c55e"])
                    style_figure(fig, height=370, y_title=all_polts[pol],
                                 title=f"{all_polts[pol]} spread",
                                 showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                if pol in ("pm2_5", "pm10"):
                    limit = WHO_24H.get(pol)
                    tiles = []
                    if limit:
                        tiles.append((f"{(val <= limit).mean() * 100:.1f}%",
                                      f"Hours within WHO (≤{limit:g} µg/m³)", ACCENT))
                    if pol == "pm2_5":
                        tiles.append((f"{(val <= 60).mean() * 100:.1f}%",
                                      "Hours within CPCB (≤60 µg/m³)", ACCENT_2))
                    tiles.append((f"{val.mean():,.0f}", "Mean observed (µg/m³)", TEXT))
                    tiles.append((f"{val.median():,.0f}", "Median (µg/m³)", TEXT))
                    tile_row(tiles)

    with tab4:
        corr_cols = [c for c in ["pm2_5", "pm10", "co", "no2", "so2", "o3", "nh3",
                                 "dust", "uv_index"] if c in df.columns]
        cdf = df[corr_cols].dropna()
        if cdf.empty:
            st.info("Not enough overlapping data for correlations.")
        else:
            cdf = cdf.rename(columns=POLLUTANTS)
            fig = px.imshow(cdf.corr(), text_auto=".2f", aspect="auto",
                            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                            zmid=0)
            fig.update_traces(textfont=dict(color=TEXT, size=11))
            style_figure(fig, height=max(380, 60 + 42 * len(corr_cols)),
                         title="Pollutant correlation")
            fig.update_layout(xaxis=dict(showgrid=False, linecolor="rgba(0,0,0,0)",
                                         tickfont=dict(color=TEXT)),
                              yaxis=dict(showgrid=False, linecolor="rgba(0,0,0,0)",
                                         tickfont=dict(color=TEXT)))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("PM2.5 and PM10 move together because they share emission "
                       "sources and the same meteorology.")

    with tab5:
        city_view = cached_cities()
        obs_c, _ = split_observed(city_view, AQI_POLLUTANTS)
        if obs_c.empty:
            st.info("No city readings available.")
        else:
            pick = st.selectbox("Pollutant", AQI_POLLUTANTS,
                                format_func=lambda c: POLLUTANTS[c])
            selected = st.multiselect(
                "Cities", sorted(CITIES),
                default=[c for c in ("Delhi", "Mumbai", "Kolkata", "Chennai")
                         if c in set(obs_c["city"])])
            sub = obs_c[obs_c["city"].isin(selected)]
            if sub.empty or pick not in sub.columns:
                st.info("Select at least one city with data.")
            else:
                daily = (sub.set_index("date").groupby("city")[pick]
                         .resample("D").mean().reset_index())
                if pick in df.columns and "Delhi" in selected:
                    ref = df.set_index("date")[pick].resample("D").mean()
                    ref = ref[ref.index >= daily["date"].min()].reset_index()
                    if not ref.empty:
                        ref["city"] = "Delhi · recorded 2020–23"
                        daily = pd.concat([daily, ref], ignore_index=True)
                fig = px.line(daily, x="date", y=pick, color="city")
                fig.update_traces(line=dict(width=2.2))
                style_figure(fig, height=430, y_title="µg/m³",
                             title=f"{POLLUTANTS[pick]} — live CAMS vs recorded")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("The 'recorded' line is CPCB monitor data on a much "
                           "higher scale; the CAMS lines are city-centre estimates.")

    with tab6:
        st.markdown(f"**{len(df):,} hourly observations** · "
                    f"{df['date'].min().date()} → {df['date'].max().date()}")
        stats_cols = [c for c in ["pm2_5", "pm10", "co", "no2", "so2", "o3", "nh3"]
                      if c in df.columns]
        stats = (df[stats_cols].describe().T
                 [["count", "mean", "std", "min", "25%", "50%", "75%", "max"]])
        stats.index = [POLLUTANTS[c] for c in stats.index]
        stats.columns = ["Count", "Mean", "Std", "Min", "p25", "Median", "p75", "Max"]
        st.dataframe(stats.round(1).astype(object), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "⬇️  Download combined dataset (CSV)",
                df.to_csv(index=False).encode("utf-8"),
                file_name="delhi_air_quality_combined.csv",
                mime="text/csv", use_container_width=True)
        with c2:
            st.download_button(
                "⬇️  Download Delhi summary (CSV)",
                df.groupby(df["date"].dt.date)[["pm2_5", "pm10"]].mean()
                  .round(1).reset_index().to_csv(index=False).encode("utf-8"),
                file_name="delhi_daily_summary.csv",
                mime="text/csv", use_container_width=True)


def page_about():
    hero("ℹ️ About AirSense",
         "Where every number comes from, how the model was trained, and how to "
         "read the AQI correctly.")

    m = cached_metrics() or {}
    met = m.get("metrics") or {}

    section("Model performance", "🎯")
    if met:
        tiles = [
            (f"{met.get('mae', '—')}", "MAE (µg/m³)", ACCENT),
            (f"{met.get('rmse', '—')}", "RMSE (µg/m³)", "#c084fc"),
            (f"{met.get('r2', '—')}", "R²", ACCENT_2),
            (f"{met.get('mape', '—')}%", "MAPE", "#fbbf24"),
            (f"{m.get('test_rows', 0):,}", "Held-out hours", TEXT),
        ]
        tile_row(tiles)
        st.caption(
            f"Measured on the newest {m.get('test_rows', 0):,} hours "
            f"({m.get('test_range', ['?', '?'])[0]} → {m.get('test_range', ['?', '?'])[1]}), "
            "which the model never saw during training — a chronological split, so "
            "it reflects genuine future-time accuracy.")
    else:
        st.info("No metrics recorded yet. Run `python train_model.py` to generate them.")

    section("Data sources", "🗄️")
    st.markdown(f"""
| Source | What it provides |
|---|---|
| **Open-Meteo Air Quality (CAMS)** | The live dashboard, the 18-city view, the 7-day forecast, and **{m.get('train_rows', 0):,} training hours** for the model |
| **ERA5 reanalysis** (Open-Meteo archive) | Hourly temperature, humidity, wind and rainfall used as model inputs |
| **CPCB Delhi monitors** (Nov 2020 – Jan 2023) | The recorded dataset in the Explorer, and seasonal context |

> **Why this matters.** CPCB monitors and CAMS estimates are *not* on the same
> scale — monitor PM2.5 in Delhi averages ~238 µg/m³, the CAMS city estimate
> ~65 µg/m³. The model is therefore trained **only** on CAMS data, the same
> source the dashboard displays and the same basis Google uses, so the
> prediction and the reading beside it are directly comparable.
""")

    section("How the prediction works", "🧠")
    st.markdown("""
1. **Inputs** — six pollutant concentrations (PM10, CO, NO2, O3, SO2, dust) for the
   hour of interest, plus temperature, humidity, wind and rainfall for that hour
   (pulled automatically), and calendar features.
2. **Model** — an XGBoost regressor (`model.pkl`) returns a single PM2.5 value in µg/m³.
3. **AQI** — the six pollutants are combined into 24-hour mean concentrations and run
   through the official CPCB breakpoint table. The overall AQI is the **worst**
   sub-index, exactly as CPCB defines it.

> **A note on 24-hour averages.** The CPCB breakpoints are defined against 24-hour
> average concentrations. Feeding a single hourly reading straight into them does
> not produce a valid AQI — for Delhi the two differ by roughly 70 points on
> average. This app computes the index the compliant way and labels it as such.
""")

    section("Reading the numbers", "📖")
    st.markdown("""
- **µg/m³** = micrograms per cubic metre of air.
- **CPCB 24-h standards**: PM2.5 ≤ **60**, PM10 ≤ **100**, CO ≤ **2 mg/m³**,
  NO2 ≤ **80**, SO2 ≤ **80**, O3 ≤ **168**.
- **WHO 2021 24-h guidelines**: PM2.5 ≤ **15**, PM10 ≤ **45**, NO2 ≤ **25**,
  SO2 ≤ **40**, CO ≤ **4 mg/m³**, O3 ≤ **100**.

| Scale | Range | Meaning |
|---|---|---|
| **CPCB AQI** (this app) | 0–500 | India's national scale; good ≤ 100, severe > 400 |
| **US AQI** | 0–500 | US EPA scale, generally stricter breakpoints |
| **European AQI** | 0–100+ | EU scale, relative to air-quality targets |
""")

    section("Disclaimer", "⚠️")
    st.markdown(
        "Forecasts and model estimates may differ from final measured values. "
        "This tool is for education and awareness only — it is not a substitute "
        "for official CPCB bulletins or health advice.")


# ----------------------------------------------------------------------------
# Navigation
# ----------------------------------------------------------------------------
def sidebar():
    with st.sidebar:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.1rem">'
            '<span style="font-size:1.7rem">🌿</span>'
            '<span style="font-size:1.35rem;font-weight:800;color:#fff;'
            'letter-spacing:-.02em">AirSense</span></div>'
            '<div style="color:#98a2b8;font-size:.76rem;margin-bottom:.6rem">'
            'India Air Quality · ML-powered</div>', unsafe_allow_html=True)

        page = st.radio(
            "Navigate", ["Predict PM2.5", "Live Dashboard",
                         "Data Explorer", "About"],
            label_visibility="collapsed")

        st.divider()
        obs, _ = observed_frame()
        row, last_ts = latest_observation(obs)
        if row is not None:
            val = float(row["pm2_5"])
            cat = pm25_category(val)
            aqi_now, _, _, _ = cpcb_aqi_24h(obs, at=last_ts)
            st.markdown(
                f'<div class="panel" style="padding:.85rem 1rem">'
                f'<div style="font-size:.68rem;letter-spacing:.07em;'
                f'text-transform:uppercase;color:{MUTED}">Delhi now</div>'
                f'<div style="font-size:2rem;font-weight:800;line-height:1.15;'
                f'color:{cat["color"] if cat else TEXT}">{val:,.0f}'
                f'<span style="font-size:.8rem;color:{MUTED};font-weight:500"> µg/m³</span></div>'
                f'<div style="margin:.35rem 0">{badge(cat["label"], cat["color"]) if cat else ""}'
                f'<span style="font-size:.7rem;color:{MUTED};margin-left:.4rem">'
                f'AQI {aqi_now if aqi_now is not None else "—"}</span></div>'
                f'<div style="font-size:.7rem;color:{MUTED}">'
                f'{pd.Timestamp(last_ts).strftime("%d %b · %H:%M")} IST · CAMS</div>'
                f'</div>', unsafe_allow_html=True)

        st.caption("Readings update hourly; the cache refreshes every 30 minutes.")
    return page


if __name__ == "__main__":
    page = sidebar()
    if page == "Predict PM2.5":
        page_predict()
    elif page == "Live Dashboard":
        page_live()
    elif page == "Data Explorer":
        page_explorer()
    elif page == "About":
        page_about()
"""Prepares the project-local air quality datasets (periodic refresh).

Downloads LIVE real air-quality data from the free Open-Meteo API:

  * ``delhi_historical.csv`` — recorded Delhi readings (pre-built, 2020–23).
  * ``live_delhi.csv``       — ~92 days of real Delhi readings + 7-day forecast
                               with extra real variables (NO, dust, UV, EU/US AQI).
  * ``live_cities.csv``      — 10 days of real readings for 18 Indian cities.

Run:  python data/prepare_data.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import refresh_live_cities, refresh_live_delhi  # noqa: E402


if __name__ == "__main__":
    print("[1/2] Refreshing Delhi live data (92 days + forecast) ...")
    df = refresh_live_delhi(past_days=92, forecast_days=7)
    print(f"[ok] live_delhi.csv -> {len(df)} hourly rows")
    print("[2/2] Refreshing 18-city live data (10 days + forecast) ...")
    cities = refresh_live_cities(past_days=10, forecast_days=3)
    print(f"[ok] live_cities.csv -> {len(cities)} hourly rows ({cities['city'].nunique()} cities)")
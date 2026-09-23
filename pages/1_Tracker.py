# streamlit run visualize_tracker.py

from __future__ import annotations
import csv

from datetime import date
from pathlib import Path
import pandas as pd
import streamlit as st

# for hPa datatype
from typing import Optional, Tuple

# now import functionalities from weather.py and raise exception if import does not work
try:
    from weather import resolve_location, fetch_barometric_pressure_hpa
except Exception:  
    resolve_location = None 
    fetch_barometric_pressure_hpa = None 

# local paths to migraine_tracker folder with correct csv files for user and daily log overview
APP_DIR = Path.home() / "migraine_tracker"
DAILY_LOG_PATH = APP_DIR / "daily_log_v1_v2.csv"
USER_IDS_PATH = APP_DIR / "user_ids_v2.csv"

# set columns of daily_log_v1_v2.csv DO NOT CHANGE !
COLUMNS = [
    "user_id",
    "day",
    "DV_migraine",
    "sleep_hours",
    "stress",
    "hydration",
    "barometric_pressure_hpa",
    "direct_sun_hours",
    "alcohol_level",
    "caffeine_level",
]

# set columns of user_ids_v2.csv DO NOT CHANGE !
USER_COLUMNS = [
    "user_id",
    "city",
    "country_code",
    "latitude",
    "longitude",
    "password",
]


def today_str() -> str:
    return date.today().strftime("%Y-%m-%d") # always staying consistent !


def _safe_float(x: str) -> Optional[float]:
    try:
        return float(str(x).strip())
    except Exception:
        return None

# translate city and country code to lat and long (two floats)
@st.cache_data(show_spinner=False, ttl=24 * 3600)
def _resolve_latlon_cached(city: str, country_code: str) -> Optional[Tuple[float, float]]:
    """Resolve (lat, lon) from city/country via Open-Meteo geocoding"""
    if resolve_location is None:
        return None
    city = str(city or "").strip()
    cc = str(country_code or "").strip().upper()
    if not city:
        return None
    try:
        loc = resolve_location(city, cc if cc else None)
    except Exception:
        return None
    if loc is None:
        return None
    return (float(loc.latitude), float(loc.longitude))

# extract hpa pressure by using lat and long (two floats)
@st.cache_data(show_spinner=False, ttl=6 * 3600)
def _fetch_hpa_cached(latitude: float, longitude: float, day: str) -> Optional[float]:
    """Fetch mean daily pressure in hPa"""
    if fetch_barometric_pressure_hpa is None:
        return None
    try:
        return fetch_barometric_pressure_hpa(latitude, longitude, day)
    except Exception:
        return None

# important for robustness: check if CSV files exist in migraine_tracker folder and IF csv files do not exist, then create them new 
def ensure_files() -> None:
    """Ensure app folder + CSV files exist and have the canonical headers (upgrade in place if needed)."""
    APP_DIR.mkdir(parents=True, exist_ok=True)

    # daily log - IF NEW then use columns from above !
    if not DAILY_LOG_PATH.exists():
        with DAILY_LOG_PATH.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(COLUMNS)

    # user ids - IF NEW then use columns from above !
    if not USER_IDS_PATH.exists():
        with USER_IDS_PATH.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(USER_COLUMNS)
        return

    df_users = pd.read_csv(USER_IDS_PATH, dtype=str).fillna("")

    # upgrade legacy files by adding missing columns, but never dropping existing columns
    for col in USER_COLUMNS:
        if col not in df_users.columns:
            df_users[col] = ""

    # enforce canonical order/schema on disk
    df_users = df_users[USER_COLUMNS].copy()
    df_users.to_csv(USER_IDS_PATH, index=False)


def load_users() -> pd.DataFrame:
    df_users = pd.read_csv(USER_IDS_PATH, dtype=str).fillna("")

    for col in USER_COLUMNS:
        if col not in df_users.columns:
            df_users[col] = ""

    df_users["user_id"] = df_users["user_id"].astype(str).str.strip()
    df_users["password"] = df_users["password"].astype(str)

    df_users = df_users[df_users["user_id"] != ""].drop_duplicates("user_id", keep="last")
    df_users = df_users[USER_COLUMNS].copy()
    return df_users.sort_values("user_id").reset_index(drop=True)


def save_users(df_users: pd.DataFrame) -> None:
    df_users = df_users.copy().fillna("")

    for col in USER_COLUMNS:
        if col not in df_users.columns:
            df_users[col] = ""

    df_users["user_id"] = df_users["user_id"].astype(str).str.strip()
    df_users["password"] = df_users["password"].astype(str)

    df_users = df_users[df_users["user_id"] != ""].drop_duplicates("user_id", keep="last")
    df_users = df_users[USER_COLUMNS].sort_values("user_id")
    df_users.to_csv(USER_IDS_PATH, index=False)


def upsert_daily_log_row(new_row: dict) -> None:
    """
    a user can overwrite a day:
    - If same user_id + same day already exists, remove old row(s) and write the new one.
    - Otherwise, just append the csv file
    """
    if not DAILY_LOG_PATH.exists():
        with DAILY_LOG_PATH.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(COLUMNS)

    df_entries = pd.read_csv(DAILY_LOG_PATH, dtype=str).fillna("")
    for column_name in COLUMNS:
        if column_name not in df_entries.columns:
            df_entries[column_name] = ""

    new_user_id = str(new_row.get("user_id", "")).strip()
    new_day = str(new_row.get("day", "")).strip()

    keep_mask = ~(
        (df_entries["user_id"].astype(str).str.strip() == new_user_id)
        & (df_entries["day"].astype(str).str.strip() == new_day)
    )
    df_entries = df_entries[keep_mask].copy()

    df_entries = pd.concat([df_entries, pd.DataFrame([new_row])], ignore_index=True)
    df_entries[COLUMNS].to_csv(DAILY_LOG_PATH, index=False)


def load_user_entries(user_id: str) -> pd.DataFrame:
    if not DAILY_LOG_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)

    df_entries = pd.read_csv(DAILY_LOG_PATH)
    for column_name in COLUMNS:
        if column_name not in df_entries.columns:
            df_entries[column_name] = pd.NA

    df_entries["user_id"] = df_entries["user_id"].astype(str).str.strip()
    df_entries = df_entries[df_entries["user_id"] == user_id].copy()

    date_series = df_entries["day"].astype(str).str.strip()
    dayfirst_guess = (date_series.str.contains("/").mean() >= 0.5) if len(date_series) else False
    df_entries["day"] = pd.to_datetime(date_series, errors="coerce", dayfirst=dayfirst_guess)

    return df_entries.dropna(subset=["day"]).sort_values("day", ascending=False)


# USER INTERFACE - UI
st.set_page_config(page_title="Migraine Tracker", layout="centered")
ensure_files()

st.session_state.setdefault("user_id", "")

st.title("Migraine Tracker - Daily Log")

df_users = load_users()
known_users = df_users["user_id"].tolist() # list all the created ("known") users so far 

# LOGIN / CREATE USER - lat and long gets translated from city/ country input into the users_id_v2.csv file
if not st.session_state.user_id:
    left, right = st.columns(2)

    with left:
        st.subheader("Login")
        selected_user_id = st.selectbox("User ID", known_users) if known_users else ""
        entered_password = st.text_input("Password", type="password")

        if st.button("Log in", disabled=not selected_user_id):
            stored_password = df_users.loc[df_users["user_id"] == selected_user_id, "password"].iloc[0]

            # legacy users: set password on first login
            if stored_password == "" and entered_password != "":
                df_users.loc[df_users["user_id"] == selected_user_id, "password"] = entered_password
                save_users(df_users)
                stored_password = entered_password

            if entered_password == stored_password:
                st.session_state.user_id = selected_user_id
                st.rerun()
            else:
                st.error("Wrong password")

    with right:
        st.subheader("Create user")

        # The user provides the identity + location. Lat and Long are written automatically via weather.py
        new_user_id = st.text_input("New user_id").strip()
        new_city = st.text_input("City").strip()
        new_country_code = st.text_input("Country code (ISO-2, e.g., FR, DE, GB)").strip().upper()
        new_password = st.text_input("New password", type="password")

        if st.button("Create user"):
            if not new_user_id or not new_city or not new_country_code or not new_password:
                st.error("user_id, city, country_code, and password are required")
            elif new_user_id in known_users:
                st.error("User already exists")
            else:
                # Auto-resolve coordinates from the provided city/country_code
                lat_str, lon_str = "", ""
                if resolve_location is None:
                    st.warning("Could not import weather.py. Saving user without latitude/longitude.")
                else:
                    try:
                        loc = resolve_location(new_city, new_country_code)
                    except Exception:
                        loc = None

                    if loc is None:
                        st.warning("Could not resolve location. Saving user without latitude/longitude (you can update later).")
                    else:
                        lat_str = f"{float(loc.latitude):.5f}"
                        lon_str = f"{float(loc.longitude):.5f}"

                df_users = pd.concat(
                    [
                        df_users,
                        pd.DataFrame( # really important that this fits the csv file's format !
                            [
                                {
                                    "user_id": new_user_id,
                                    "city": new_city,
                                    "country_code": new_country_code,
                                    "latitude": lat_str,
                                    "longitude": lon_str,
                                    "password": new_password,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
                save_users(df_users)
                st.success("User created. Please log in.")
                st.rerun()

    st.stop()

# MAIN (only "Entry" + "View")
current_user_id = st.session_state.user_id
st.success(f"Hello {current_user_id}")

if st.button("Switch user"):
    st.session_state.user_id = ""
    st.rerun()

tab_entry, tab_view = st.tabs(["Add entry", "View entries"])

# ENTRY - derived from tracker !
with tab_entry:
    with st.form("add_entry_form"):
        day = st.text_input("day", value=today_str())

        DV_migraine = st.selectbox("DV_migraine (0-10)", options=list(range(0, 11)), index=0)

        sleep_hours = st.number_input(
            "sleep_hours (0-24)", min_value=0.0, max_value=24.0, value=7.0, step=0.5
        )

        stress = st.selectbox("stress (0-10)", options=list(range(0, 11)), index=3)

        hydration = st.selectbox(
            "hydration (1 for < 1.5L, 2 for 1.5 - 2.5L, 3 for 2.5L <)",
            options=[1, 2, 3],
            index=1,
        )

        # (hPa) score: auto-fill from user location + entered day --> also: manual fallback if needed (if API does not work)
        barometric_pressure_hpa_label = "barometric_pressure_hpa (in hPa)"
        barometric_pressure_hpa_help = "the hpa level gets added automatically based on your location and the date"
        
        auto_hpa: Optional[float] = None
        auto_reason: str = ""
        
        # 1) Validate day format (YYYY-MM-DD)
        day_str = str(day).strip()
        try:
            _ = date.fromisoformat(day_str)
        except Exception:
            auto_reason = "Invalid date format. Please use YYYY-MM-DD."
        
        # 2) Locate the user's coordinates (prefer stored lat/lon; else resolve from city/country)
        lat: Optional[float] = None
        lon: Optional[float] = None
        if not auto_reason:
            df_users_live = load_users()
            urow = df_users_live[df_users_live["user_id"].astype(str).str.strip() == str(current_user_id).strip()]
            if urow.empty:
                auto_reason = "User profile not found in user_ids_v2.csv."
            else:
                city = str(urow.iloc[0].get("city", "")).strip()
                cc = str(urow.iloc[0].get("country_code", "")).strip().upper()
                lat = _safe_float(str(urow.iloc[0].get("latitude", "")))
                lon = _safe_float(str(urow.iloc[0].get("longitude", "")))
        
                if lat is None or lon is None:
                    # Try resolving from city/country_code if available
                    coords = _resolve_latlon_cached(city, cc)
                    if coords is not None:
                        lat, lon = coords
        
                if lat is None or lon is None:
                    auto_reason = "No resolvable location for this user (city/country or coordinates missing)."
        
        # 3) Fetch mean daily pressure (Forecast → Archive with variable fallbacks)
        if not auto_reason:
            auto_hpa = _fetch_hpa_cached(lat, lon, day_str)
            if auto_hpa is None:
                auto_reason = "Could not fetch pressure automatically (network/API issue or missing data)."
        
        if auto_hpa is not None:
            st.caption(f"Auto-filled: {auto_hpa:.1f} hPa")
            barometric_pressure_hpa = st.number_input(
                barometric_pressure_hpa_label,
                min_value=850.0,
                max_value=1100.0,
                value=float(round(auto_hpa, 1)),
                step=0.1,
                help=barometric_pressure_hpa_help,
                disabled=True,
            )
        else:
            if auto_reason:
                st.warning(auto_reason)
            barometric_pressure_hpa = st.number_input(
                barometric_pressure_hpa_label,
                min_value=850.0,
                max_value=1100.0,
                value=1013.0,
                step=1.0,
                help=barometric_pressure_hpa_help,
            )

        direct_sun_hours = st.number_input(
            "direct_sun_hours (hours 0-24)",
            min_value=0.0,
            max_value=24.0,
            value=1.0,
            step=0.5,
        )

        alcohol_level = st.selectbox(
            "alcohol_level (0=None, 1=Low, 2=Medium, 3=High)",
            options=[0, 1, 2, 3],
            index=0,
        )

        caffeine_level = st.selectbox(
            "caffeine_level (0=None, 1=Low, 2=Medium, 3=High)",
            options=[0, 1, 2, 3],
            index=1,
        )

        if st.form_submit_button("Save entry"):
            upsert_daily_log_row(
                {
                    "user_id": current_user_id,
                    "day": str(day).strip(),
                    "DV_migraine": str(DV_migraine),
                    "sleep_hours": str(sleep_hours),
                    "stress": str(stress),
                    "hydration": str(hydration),
                    "barometric_pressure_hpa": str(barometric_pressure_hpa),
                    "direct_sun_hours": str(direct_sun_hours),
                    "alcohol_level": str(alcohol_level),
                    "caffeine_level": str(caffeine_level),
                }
            )
            st.success("Entry saved (overwrites same day).")
            st.rerun()

#  VIEW --> also download available
with tab_view:
    df_entries = load_user_entries(current_user_id)

    if df_entries.empty:
        st.info("No entries yet")
    else:
        st.dataframe(df_entries, use_container_width=True)

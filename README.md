# Migraine Tracker

A migraine tracking project that evolved across three stages:
- **V0**: CLI prototype for logging and basic viewing
- **V1**: CLI + **ordinal logistic regression** + Streamlit dashboard
- **V2 (final)**: Streamlit multipage app with **user registration/login**, **weather auto-fill (pressure)**, dashboards, and regression

## Overview

Migraines vary widely from day to day and from person to person, and they can be influenced by a mix of lifestyle habits (sleep, stress, hydration, caffeine/alcohol) and environmental conditions (e.g., barometric pressure). This app is a **self-tracking tool** that helps users build a personal history of migraine severity and potential triggers, then run an **interpretable statistical analysis** (ordinal logistic regression) to explore which factors are most associated with migraine severity for that individual over time.

The project was developed iteratively from a simple CLI prototype (V0) to a CLI + analysis + dashboard version (V1), and finally to a Streamlit multipage UI (V2) that supports registration/login, daily logging, regression analysis, interactive visualization, and automatic barometric pressure retrieval via a public weather API (no key required).

---

## Quick start (V2 – Final)

1) Install dependencies:
```bash
pip install -r requirements_v2.txt
```

2) Run the Streamlit app:
```bash
streamlit run home.py
```

Use the sidebar to navigate:
- **Tracker** (register/login + add entries)
- **Daily Log + Regression** (filter logs, plots, run regression)

If Streamlit doesn’t reflect new entries immediately, use Streamlit’s menu (⋮) -> **Clear cache**, then reload.

---

## Stages overview

### V0 — CLI prototype
Focus: a minimal command-line tracker to prove the workflow (create/select user, log daily inputs, view recent logs).

### V1 — CLI + regression + dashboard
Adds:
- Ordinal logistic regression (binned migraine severity)
- Streamlit dashboard for filtering and plotting

### V2 — Streamlit multipage (final)
Adds:
- Registration/login for multiple users (stored locally)
- Weather utilities to auto-fill **barometric pressure (hPa)** from user location + date (Open‑Meteo, no API key)
- A multipage Streamlit UI (Home + `pages/`), combining tracking + visualization + regression

---

## Project structure (V2)

```text
.
├── home.py
├── regression.py
├── weather.py
├── requirements_v2.txt
├── daily_log_v1_v2.csv
├── user_ids_v2.csv
├── pages/
│   ├── 1_Tracker.py
│   └── 2_Daily_Log_Regression.py
├── .gitignore
└── README.md

## Data format

The daily log CSV header is:

`user_id, day, DV_migraine, sleep_hours, stress, hydration, barometric_pressure_hpa, direct_sun_hours, alcohol_level, caffeine_level`

Example encodings:
- `DV_migraine`: 0–10
- `hydration`: 1 = <1.5L, 2 = 1.5–2.5L, 3 = >2.5L
- `caffeine_level`: 0=None, 1=Low, 2=Medium, 3=High

---

## Regression (what it does)

- Bins `DV_migraine` into 3 ordered categories:
  - 0 → none
  - 1–5 → manageable
  - 6–10 → severe
- Fits an **ordinal logistic regression** with predictors:
  `sleep_hours, hydration, caffeine_level, barometric_pressure_hpa`
- Outputs a stats summary and a simplified interpretation

Note: regression needs enough non-zero migraine days to be meaningful (in our project, we target **≥ 40** non-zero entries).

---

## Dependencies

This tracker relies on these libraries. So if you have them installed, you are good to go.
- `numpy`
- `pandas`
- `statsmodels`
- `streamlit`
- `marplotlib`

Or, please refer to the requrements file:
- V2: `requirements_v2.txt`

---

## Notes for demo

- The regression part requires sufficient inputs. Thus, you can find some sample csv files in the repository. All the users in these files have `12345` as their password
- Streamlit caching can hide recent updates; use **Clear cache** if available users, plots and tables don’t refresh. You can do so from the streamlit menue on top right.

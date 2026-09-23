# to run through terminal:
# cd /path/
# streamlit run home.py

from __future__ import annotations

import contextlib
import io
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# import regression.py --> must be in SAME folder
try:
    import regression
except Exception:
    regression = None


# Contract with used csv files (must match headers and titles exactly)
CSV_PATH = Path.home() / "migraine_tracker" / "daily_log_v1_v2.csv"

NUM_COLS = [ # MUST MATCH !
    "DV_migraine",
    "sleep_hours",
    "stress",
    "hydration",
    "barometric_pressure_hpa",
    "direct_sun_hours",
    "alcohol_level",
    "caffeine_level",
]

REQUIRED_COLS = ["user_id", "day", *NUM_COLS]

# "long_names" because input questions are quite long
LONG_NAMES = {
    "sleep_hours": "Sleep duration (in hours)",
    "hydration": "Hydration level (1 = low, 2 = medium, 3 = high)",
    "caffeine_level": "Caffeine consumption level (0 = none, 1 = low, 2 = medium, 3 = high)",
    "barometric_pressure_hpa": "Barometric pressure (hPa)",
}

PREDICTORS = ["sleep_hours", "hydration", "caffeine_level", "barometric_pressure_hpa"] # for regression we use variables that do not not depict a high risk of multicollinearity


@dataclass(frozen=True)
class FilterConfig:
    user_id: str
    start: pd.Timestamp
    end: pd.Timestamp
    last_n: int


@st.cache_data(show_spinner=False)
def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        return df

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df["user_id"] = df["user_id"].astype(str).str.strip()
    df["day"] = pd.to_datetime(df["day"], errors="coerce")

    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["day"]).sort_values("day").reset_index(drop=True)
    return df


def apply_filters(df: pd.DataFrame, cfg: FilterConfig) -> pd.DataFrame:
    out = df[df["user_id"] == cfg.user_id].copy()
    out = out[(out["day"] >= cfg.start) & (out["day"] <= cfg.end)]
    return out


def available_numeric_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in NUM_COLS if c in df.columns and df[c].notna().any()]


def plot_timeseries(df: pd.DataFrame) -> None:
    st.subheader("Your Entries Plotted")

    if df.empty:
        st.info("No rows for the selected filters.")
        return

    cols = available_numeric_cols(df)
    if not cols:
        st.warning("No numeric values available to plot (all missing or non-numeric).")
        return

    label_map = {"barometric_pressure_hpa": "barometric_pressure_hpa (divided by 100)"}
    inv_label_map = {v: k for k, v in label_map.items()}

    display_cols = [label_map.get(c, c) for c in cols]
    selected_display = st.multiselect("Variables", display_cols, default=display_cols)
    if not selected_display:
        st.info("Select at least one variable to plot.")
        return

    selected_cols = [inv_label_map.get(c, c) for c in selected_display]

    tmp = df.set_index("day").copy()
    if "barometric_pressure_hpa" in tmp.columns:
        tmp["barometric_pressure_hpa"] = tmp["barometric_pressure_hpa"] / 100

    tmp = tmp.rename(columns=label_map)
    plot_cols = [label_map.get(c, c) for c in selected_cols]

    fig, ax = plt.subplots()
    tmp[plot_cols].plot(ax=ax)

    # leg = ax.get_legend()
    # if leg is not None:
    #    leg.remove()

    ax.set_title("Tracked Variables")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def render_kpis(df: pd.DataFrame, last_n: int) -> None:
    st.subheader("Average Migraine Score")

    if df.empty:
        st.info("No KPIs (empty selection).")
        return

    dv = df["DV_migraine"].dropna().tail(last_n)
    if dv.empty:
        st.metric("Average DV_migraine (recent)", "N/A")
    else:
        st.metric(f"Last {len(dv)} days", f"{dv.mean():.2f}")


# Regression summary (sd, effect, p-value) --> can be (de)selected, if the user wants

def parse_orderedmodel_table(stdout_text: str) -> Dict[str, Dict[str, float]]:
    """
    Parse the coefficient table lines 
    """
    rows: Dict[str, Dict[str, float]] = {}
    for line in stdout_text.splitlines():
        parts = line.split()
        if len(parts) >= 7 and parts[0] in PREDICTORS:
            try:
                rows[parts[0]] = {
                    "coef": float(parts[1]), # coefficents
                    "std_err": float(parts[2]), # standard error
                    "z": float(parts[3]), # z value
                    "p_value": float(parts[4]), # p value
                    "ci_low": float(parts[5]), # lower bound confidence interval
                    "ci_high": float(parts[6]),  # higher bound confidence interval
                }
            except Exception:
                pass
    return rows


def extract_days_events(stdout_text: str) -> Tuple[int | None, int | None]:
    m = re.search(r"Loaded\s+(\d+)\s+day\(s\)\.\s+Migraine events\s+\(DV>0\):\s+(\d+)", stdout_text)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def extract_simplified_findings(stdout_text: str) -> List[str]:
    lines = stdout_text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip() == "=== SIMPLIFIED FINDINGS ===":
            start = i + 1
            break
    if start is None:
        return []
    out: List[str] = []
    for line in lines[start:]:
        s = line.strip()
        if not s:
            continue
        if s.startswith("==="):
            break
        out.append(s)
    return out


def compute_user_stddev(df_all: pd.DataFrame, user_id: str, cols: List[str]) -> Dict[str, float]:
    df_u = df_all[df_all["user_id"] == user_id].copy()
    stds: Dict[str, float] = {}
    for c in cols:
        if c in df_u.columns:
            series = pd.to_numeric(df_u[c], errors="coerce")
            stds[c] = float(series.std(ddof=1)) if series.notna().sum() >= 2 else float("nan")
        else:
            stds[c] = float("nan")
    return stds


def pretty_p(p: float) -> str:
    if math.isnan(p):
        return "—"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def effect_direction(coef: float) -> str:
    return "REDUCES risk" if coef < 0 else "INCREASES risk"


def effect_strength(p: float) -> str:
    if math.isnan(p):
        return "Unclear"
    return "Strong" if p < 0.05 else "Weak / Unclear"


def render_regression_results(user_id: str, df_all: pd.DataFrame) -> None:
    st.header("Regression insights (easy to understand)")
    st.caption("Simple summary: effect size, p-value, and your real-world standard deviation for each factor.")

    if regression is None:
        st.error("Could not import regression.py. Make sure regression.py is in the same folder as home.py and this Streamlit file.")
        return

    show_tech = st.checkbox("Show technical output", value=False)

    if not st.button("Run regression"):
        return

    # Run regression.py and capture its output
    try:
        regression.CSV_PATH = CSV_PATH
    except Exception:
        pass

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            regression.run_analysis(user_id=user_id, min_events=40, scale=True)
    except Exception as e:
        st.error(str(e))
        return

    output = buf.getvalue().strip()
    if not output:
        st.warning("Regression ran but produced no output...")
        return

    # Top KPIs (from regression output if present)
    days, events = extract_days_events(output)
    c1, c2 = st.columns(2)
    c1.metric("Days recorded", days if days is not None else "—")
    c2.metric("Migraine days (DV > 0)", events if events is not None else "—")

    # coefficient table
    coef_rows = parse_orderedmodel_table(output)
    if not coef_rows:
        st.warning("Could not parse coefficient table from regression output.")
    else:
        # User-specific standard deviations (from raw CSV)
        user_stds = compute_user_stddev(df_all, user_id, PREDICTORS)

        # summary table
        summary_rows = []
        for var in PREDICTORS:
            if var not in coef_rows:
                continue
            r = coef_rows[var]
            coef = r["coef"]
            p = r["p_value"]
            summary_rows.append(
                {
                    "Factor": LONG_NAMES.get(var, var),
                    "Effect (per 1 SD)": effect_direction(coef),
                    "Coefficient": round(coef, 2),
                    "p-value": pretty_p(p),
                    "Strength": effect_strength(p),
                    "Your SD (raw units)": (round(user_stds.get(var, float("nan")), 2) if not math.isnan(user_stds.get(var, float("nan"))) else "—"),
                }
            )

        # Sort by lowest p, then absolute coefficient
        summary_rows.sort(key=lambda x: (float(x["p-value"].replace("< 0.001", "0.0005")) if x["p-value"] != "—" else 1.0, -abs(x["Coefficient"])))
        st.subheader("Most important factors")
        st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

    findings = extract_simplified_findings(output)
    if findings:
        st.subheader("Simplified findings (from regression.py)")
        for f in findings:
            st.write(f"- {f}")

    # Technical output only if checkbox is ticked
    if show_tech:
        st.divider()
        st.subheader("Technical output (from regression.py)")
        st.code(output, language="text")


def main() -> None:
    st.set_page_config(page_title="Migraine Tracker", layout="wide")
    st.title("Migraine Tracker")
    st.caption(f"Data source: {CSV_PATH}")

    try:
        df = load_data(CSV_PATH)
    except Exception as e:
        st.error(str(e))
        st.stop()

    if df.empty:
        st.warning("CSV exists but contains no rows yet.")
        st.stop()

    user_ids = sorted(df["user_id"].dropna().unique().tolist())
    if not user_ids:
        st.warning("No user_id values found in the CSV.")
        st.stop()

    # ---- Sidebar filters ----
    st.sidebar.header("Filters")
    user_id = st.sidebar.selectbox("User ID", user_ids)

    min_day = df["day"].min().date()
    max_day = df["day"].max().date()
    picked = st.sidebar.date_input("Date range", value=(min_day, max_day), min_value=min_day, max_value=max_day)

    if isinstance(picked, tuple) and len(picked) == 2:
        start = pd.Timestamp(picked[0])
        end = pd.Timestamp(picked[1])
    else:
        start = pd.Timestamp(min_day)
        end = pd.Timestamp(max_day)

    last_n = st.sidebar.slider("Choose number of recent days for your migraine average", 1, 100, 7)

    cfg = FilterConfig(user_id=user_id, start=start, end=end, last_n=last_n)
    dff = apply_filters(df, cfg)

    left, right = st.columns([1, 2])

    with left:
        render_kpis(dff, cfg.last_n)

        st.subheader("Filtered rows")
        st.dataframe(dff.reset_index(drop=True), hide_index=True, use_container_width=True)

        st.download_button(
            label="Download filtered CSV",
            data=dff.to_csv(index=False).encode("utf-8"),
            file_name=f"daily_log_v1_v2_{cfg.user_id}_{cfg.start.date()}_{cfg.end.date()}.csv",
            mime="text/csv",
        )

    with right:
        plot_timeseries(dff)

    # Full-width regression section (wide tables)
    st.divider()
    render_regression_results(user_id=cfg.user_id, df_all=df)


if __name__ == "__main__":
    main()

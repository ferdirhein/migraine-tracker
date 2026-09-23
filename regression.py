import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.miscmodels.ordinal_model import OrderedModel
from typing import Optional, List, Tuple, Dict

# again, as in the other files: set the location of csv file the regression works on
CSV_PATH = Path.home() / "migraine_tracker" / "daily_log_v1_v2.csv"

#set the predictor variables
V2_PREDICTORS = ["sleep_hours", "hydration", "caffeine_level", "barometric_pressure_hpa"]

#binning the severity to avoid levels that are not recorded to 3 ordinal levels
def bin_severity(val):
    '''      
    0      -> 0 (None)
    1 - 5  -> 1 (Manageable)
    6 - 10 -> 2 (Severe)
    '''
    if pd.isna(val):
        return np.nan
    val = float(val)
    if val == 0:
        return 0
    if val <= 5:
        return 1
    return 2

# use zscore to identify the constant columns(predictor variables) for later deleteing the column
def zscore_safe(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return s * 0.0
    return (s - s.mean()) / sd

# drop the constant columns identified above, using zscore and SD
def drop_constant_columns(X: pd.DataFrame, eps: float = 1e-12) -> Tuple[pd.DataFrame, List[str]]:
    
    #it take pd.DataFrame and delete any columns with variance smaller epsiron
    #epsiron is used for the threshold instead of 0 due to the float rounding constraint
    #returns the new DataFrame with only kept columns and also list of names of constant columns dropped 
    
    variances = X.var(ddof=0)
    keep = variances[variances > eps].index.tolist()
    dropped = [c for c in X.columns if c not in keep]
    return X[keep], dropped

#the main regression function for a specified user

#40 is used as we have 4 predictor variables
#we need roughly 10 events per one predictor

def run_analysis(
    user_id: Optional[str] = None,
    min_events: int = 40,
    scale: bool = True,
):
    
    #Runs Ordinal Logistic Regression (OLR) on all logged days up to today.
    # uses sleep_hour, hydration, caffeine and barometric_pressure_hpa
    # only works for users with more than 40 nonzero entries in migaine severity
    # scale is used for optional standardisation of the kept columns for regression stability, but I kept this condision if we in the future decide to use unit instead of SD for interpretation of the test result
    # you can set scale to false for unit result when this function is called
    

    #safety check to check whether the csv file exists
    if not CSV_PATH.exists():
        print("No data found. Run home.py (1_Tracker) first and enter dates.")
        return

    df = pd.read_csv(CSV_PATH)

    #Filters df for the specified user_id
    if user_id is not None and "user_id" in df.columns:
        df = df[df["user_id"].astype(str) == str(user_id)]

    # set the date range to up to today
    if "day" in df.columns:
        df["day"] = pd.to_datetime(df["day"], errors="coerce")
        df = df.dropna(subset=["day"]).sort_values("day")
        today = pd.Timestamp.today().normalize()
        df = df[df["day"] <= today]

    # safety check again to check whether the dependent variable exits
    if "DV_migraine" not in df.columns:
        print("Missing required column: DV_migraine")
        print("Columns found:", list(df.columns))
        return

    # same safety caheck for predictor variables
    missing = [c for c in V2_PREDICTORS if c not in df.columns]
    if missing:
        print("Missing required predictor columns:", missing)
        print("Current columns:", list(df.columns))
        return

    print("Using predictors:", V2_PREDICTORS)

    # convert all the variables to numeric for OLR
    df["DV_migraine"] = pd.to_numeric(df["DV_migraine"], errors="coerce")
    for c in V2_PREDICTORS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # drop rows where DV is missing (na)
    df = df.dropna(subset=["DV_migraine"]).copy()
    if len(df) == 0:
        print("No usable rows after cleaning DV_migraine.")
        return

    # sets events so that the min_events constraint set in the obejects can be enforeced
    events = int((df["DV_migraine"] > 0).sum())
    print(f"Loaded {len(df)} day(s). Migraine events (DV>0): {events}")
    if events < min_events:
        print(f"Not enough events to fit reliably: {events}/{min_events}. Keep logging.")
        return

    # apply the binning to the migraine column, drop any with and set them to integer
    df["severity_binned"] = df["DV_migraine"].apply(bin_severity)
    df = df.dropna(subset=["severity_binned"]).copy()
    df["severity_binned"] = df["severity_binned"].astype(int)

    # prep for distribution display in the result summary
    label_map = {0: "None", 1: "Manageable", 2: "Severe"}
    counts = df["severity_binned"].value_counts().reindex([0, 1, 2], fill_value=0)
    print("Binned DV distribution:", dict(counts.rename(index=label_map))) # translate the index into the human language

    # safety check for binned severity for OLR to work
    if df["severity_binned"].nunique() < 2:
        print("Not enough variation in binned DV (only one category present).")
        return

    # build X to be used for the next three functions
    X = df[V2_PREDICTORS].copy()

    # impute missing predictors (median is used for na entries)
    for c in V2_PREDICTORS:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())

    # drop constant predictors
    X, dropped = drop_constant_columns(X)
    if dropped:
        print("Dropped 0-variance predictors:", dropped)
    if X.shape[1] == 0:
        print("All predictors were constant; cannot fit regression.")
        return

    # scaling for standardisation
    if scale:
        for c in X.columns:
            X[c] = zscore_safe(X[c])

    # set the dependent variable for OLR
    y = df["severity_binned"]

    # OLR
    try:
        model = OrderedModel(y, X, distr="logit")
        result = model.fit(method="bfgs", disp=False)

        print("\n=== ORDINAL LOGISTIC REGRESSION ===")
        print(result.summary())

        print("\n=== SIMPLIFIED FINDINGS ===")
        params = result.params
        for col in X.columns:
            coef = float(params[col])
            direction = "INCREASES risk" if coef > 0 else "REDUCES risk"
            strength = abs(coef)

            if strength < 0.1:
                magnitude = "Negligible"
            elif strength < 0.5:
                magnitude = "Weak"
            else:
                magnitude = "STRONG"

            unit = "per 1 SD" if scale else "per 1 unit"
            print(f"{col.upper()}: {magnitude} link, {direction} ({unit}, coef={coef:.2f})")

    # suggestions for the user if an error occurs
    except Exception as e:
        print("\nCould not run ordinal regression.")
        print("Error:", e)
        print("Common fixes:")
        print("- Ensure you have enough entries with non-zero migrane severity days (minimum 40 days for this analysis)") # really important to have reliable regression
        print("- If a predictor does not variate for the user, it will be dropped automatically.")


if __name__ == "__main__":
    run_analysis()

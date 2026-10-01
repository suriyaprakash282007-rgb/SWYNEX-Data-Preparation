"""
Data preparation pipeline: Telco Customer Churn (IBM sample dataset).

Steps: load -> profile -> fix types -> handle missing values -> standardise
categories -> remove duplicates -> validate -> save cleaned data + report.

Usage:
    python src/prepare_data.py                      # downloads the dataset
    python src/prepare_data.py --input my_file.csv  # use a local copy
"""
import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

URL = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
       "master/data/Telco-Customer-Churn.csv")
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
OUT = ROOT / "data" / "processed" / "telco_clean.csv"
REPORT = ROOT / "reports" / "cleaning_report.json"

report = {}  # collects everything we change, for documentation


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading dataset to {path} ...")
        urllib.request.urlretrieve(URL, path)
    # Read everything as-is first; we handle types explicitly below.
    return pd.read_csv(path)


def profile(df: pd.DataFrame, label: str) -> None:
    report[label] = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_per_column": {c: int(n) for c, n in df.isna().sum().items() if n},
        "duplicate_rows": int(df.duplicated().sum()),
    }


def fix_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Strip stray whitespace in all text columns; treat empty strings as NaN.
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    for c in text_cols:
        df[c] = df[c].str.strip().replace("", np.nan)

    # TotalCharges is text because of blank entries -> convert to float.
    before = df["TotalCharges"].isna().sum()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    report["totalcharges_new_nans_from_coercion"] = int(
        df["TotalCharges"].isna().sum() - before
    )

    # SeniorCitizen is coded 0/1 while all other flags are Yes/No -> align.
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # Target to binary integer.
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype("Int64")

    # Low-cardinality text columns -> category dtype (saves memory, clearer intent).
    for c in df.select_dtypes(include=["object", "string"]).columns:
        if c != "customerID":
            df[c] = df[c].astype("category")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    actions = {}

    # Assumption: blank TotalCharges occurs for customers with tenure == 0
    # (signed up, not yet billed) -> true value is 0, not "unknown".
    mask = df["TotalCharges"].isna()
    new_customers = mask & (df["tenure"] == 0)
    df.loc[new_customers, "TotalCharges"] = 0.0
    actions["TotalCharges_filled_with_0_tenure0"] = int(new_customers.sum())

    # Generic fallback for anything else still missing (logged, not silent).
    remaining = {}
    for c in df.columns:
        n = int(df[c].isna().sum())
        if n == 0:
            continue
        if c == "Churn":
            df = df.dropna(subset=["Churn"])  # never impute the label
            remaining[c] = f"{n} rows dropped (target missing)"
        elif pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(df[c].median())
            remaining[c] = f"{n} filled with median"
        else:
            df[c] = df[c].fillna(df[c].mode()[0])
            remaining[c] = f"{n} filled with mode"
    actions["fallback_imputation"] = remaining
    report["missing_value_actions"] = actions
    return df


def standardise_categories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Assumption: "No internet service" / "No phone service" carry the same
    # information as "No" for the add-on columns (the dependency is already
    # captured by InternetService / PhoneService).
    cols = ["MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"]
    for c in cols:
        df[c] = (df[c].astype(str)
                 .replace({"No internet service": "No", "No phone service": "No"})
                 .astype("category"))
    # Shorten payment method labels like "Bank transfer (automatic)".
    df["PaymentMethod"] = (df["PaymentMethod"].astype(str)
                           .str.replace(r"\s*\(automatic\)", "", regex=True)
                           .astype("category"))
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    df = df.drop_duplicates(subset="customerID", keep="first")
    report["duplicate_customers_removed"] = int(n - len(df))
    return df


def outlier_report(df: pd.DataFrame) -> None:
    """Flag (but do NOT remove) IQR outliers; values are plausible billing data."""
    out = {}
    for c in ["tenure", "MonthlyCharges", "TotalCharges"]:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        n = int(((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum())
        out[c] = n
    report["iqr_outliers_flagged_not_removed"] = out


def validate(df: pd.DataFrame) -> None:
    assert df["customerID"].is_unique, "customerID must be unique"
    assert df.isna().sum().sum() == 0, "no missing values should remain"
    assert (df["tenure"] >= 0).all()
    assert (df["MonthlyCharges"] > 0).all()
    assert (df["TotalCharges"] >= 0).all()
    assert set(df["Churn"].unique()) <= {0, 1}
    print("All validation checks passed.")


def main(input_path: Path) -> None:
    df = load(input_path)
    profile(df, "before")
    df = fix_types(df)
    df = handle_missing(df)
    df = standardise_categories(df)
    df = drop_duplicates(df)
    outlier_report(df)
    validate(df)
    profile(df, "after")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    REPORT.write_text(json.dumps(report, indent=2))
    print(f"Saved {len(df)} rows -> {OUT}")
    print(f"Report -> {REPORT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=RAW)
    main(ap.parse_args().input)

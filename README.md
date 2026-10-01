# Telco Customer Churn – Data Preparation

Cleaning pipeline that turns the raw IBM Telco Customer Churn dataset into an
analysis-ready table for churn prediction / EDA.

## Dataset
- **Name:** Telco Customer Churn (IBM sample data)
- **Source:** https://github.com/IBM/telco-customer-churn-on-icp4d (also on Kaggle)
- **Size:** ~7,043 customers x 21 columns (demographics, services, billing, `Churn` target)

## Software used
| Tool | Purpose |
|---|---|
| Python 3.10+ | Language |
| pandas, NumPy | Cleaning and type handling |
| Jupyter / VS Code | Exploration and editing |
| Git + GitHub | Version control, submission |

## How to run
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/prepare_data.py
```
Outputs: `data/processed/telco_clean.csv` and `reports/cleaning_report.json`
(before/after profile and every change made).

## What the pipeline does
1. **Load** the CSV (auto-downloads if missing).
2. **Profile** shape, dtypes, missing values, duplicates (saved to the report).
3. **Fix types**
   - Trim whitespace; empty strings become `NaN`.
   - `TotalCharges` (text) to float. Blank entries become `NaN`.
   - `SeniorCitizen` 0/1 to `No`/`Yes` (consistent with other flags).
   - `Churn` Yes/No to 1/0 (nullable integer).
   - Low-cardinality text columns to `category`.
4. **Handle missing values** (details below).
5. **Standardise categories**, remove duplicate customers.
6. **Validate** with assertions (unique IDs, no NaNs, non-negative charges, binary target).

## Assumptions (documented decisions)
1. **Blank `TotalCharges` = 0 for `tenure == 0`.** The blanks occur only for customers who just signed up and haven't been billed. This is a true zero, not an unknown, so it is not imputed with a mean/median.
2. **Fallback imputation** (not triggered on the standard file, but logged if it is): median for numeric, mode for categorical, and rows with missing `Churn` are dropped. The target is never imputed.
3. **"No internet service" / "No phone service" are merged into "No"** for add-on columns, since `InternetService` / `PhoneService` already encode that dependency. This reduces redundant categories.
4. **`customerID` is a unique key**; duplicates keep the first occurrence.
5. **Outliers are flagged, not removed** (IQR rule). High charges and long tenure are plausible business values.
6. No scaling or encoding is applied here; that belongs in the modelling step to avoid leakage.

## Repository structure
```
├── README.md
├── requirements.txt
├── src/prepare_data.py
├── data/raw/            # downloaded file (git-ignored)
├── data/processed/      # cleaned output
└── reports/             # cleaning_report.json
```

## Submit to GitHub
```bash
git init && git add . && git commit -m "Data preparation: Telco churn"
git branch -M main
git remote add origin https://github.com/<your-username>/telco-data-prep.git
git push -u origin main
```

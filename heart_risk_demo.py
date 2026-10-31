# ==============================================================
# Heart Attack Risk Prediction - DEMO VERSION
# --------------------------------------------------------------
# Trains a simple model on a (synthetic) patient dataset and
# reports a heart-attack risk probability for each patient in
# a held-out sample.
#
# Sex-aware features:
#   - `sex` is kept as a predictor (risk differs by sex)
#   - `menopause_status` is included for women (men = not_applicable)
#   - Performance is reported separately for men and women
#
# DEMO LIMITATIONS (full version removes these):
#   - Single baseline model, no model comparison / tuning
#   - Only a held-out sample (20%) of patients is scored
#   - No cross-validation, calibration or per-patient explanations
#   - Outcome is not a time-bound (e.g. 10-year) risk and predictors
#     are not aligned with standard risk calculators (PCE / PREVENT)
#   - No saved model, API, dashboard or PDF reports
#
# NOTE: Data is synthetic. This is NOT a medical device and must
# not be used for real clinical decisions.
# ==============================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

# ---------- 1. Load data ----------
df = pd.read_csv("patients_demo.csv")
print(f"Dataset shape: {df.shape}")
print(f"Total missing cells: {int(df.isna().sum().sum())}")
print("\nMissing values per column:")
print(df.isna().sum()[df.isna().sum() > 0])

# ---------- 2. Menopause status handling ----------
# Menopause only applies to women. Rules (deterministic, no fitting,
# so applying them before the split causes no data leakage):
#   - men are always "not_applicable"
#   - women with a missing value are inferred from age
#     (age >= 51 ~ typical natural menopause -> postmenopausal)
MENOPAUSE_AGE = 51

def fix_menopause(data):
    data = data.copy()
    is_male = data["sex"] == "M"
    data.loc[is_male, "menopause_status"] = "not_applicable"
    missing_f = (~is_male) & data["menopause_status"].isna()
    data.loc[missing_f, "menopause_status"] = np.where(
        data.loc[missing_f, "age"] >= MENOPAUSE_AGE,
        "postmenopausal", "premenopausal")
    return data

n_inferred = int(((df["sex"] == "F") & df["menopause_status"].isna()).sum())
df = fix_menopause(df)
print(f"\nMenopause status inferred from age for {n_inferred} women")

# ---------- 3. Define features and target ----------
TARGET = "heart_attack"
ID_COL = "patient_id"
categorical = ["sex", "menopause_status", "chest_pain_type", "resting_ecg"]
numeric = [c for c in df.columns if c not in categorical + [TARGET, ID_COL]]

X = df.drop(columns=[TARGET])
y = df[TARGET]

# ---------- 4. Train / test split ----------
# Stratify by target AND sex so both sexes appear in train and test
strata = y.astype(str) + "_" + df["sex"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=strata)

# ---------- 5. Preprocessing (median/mode imputation + scaling) ----------
preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())]), numeric),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
])

# ---------- 6. Baseline model ----------
model = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=1000)),
])
# Drop the ID column before fitting so it is never used as a feature
model.fit(X_train.drop(columns=[ID_COL]), y_train)

# ---------- 7. Predict risk probabilities ----------
proba = model.predict_proba(X_test.drop(columns=[ID_COL]))[:, 1]
pred = (proba >= 0.5).astype(int)

# ---------- 8. Evaluate (overall) ----------
print("\n=== Demo model performance (held-out sample) ===")
print(f"ROC-AUC : {roc_auc_score(y_test, proba):.3f}")
print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
print("Confusion matrix:")
print(confusion_matrix(y_test, pred))

# ---------- 9. Evaluate by sex ----------
# Risk models can perform differently in men and women, so we check
# discrimination (AUC) and average predicted vs observed risk per sex.
# NOTE: subgroup samples are small in this demo, so treat as indicative.
print("\n=== Performance by sex (held-out sample) ===")
rows = []
for s, label in [("M", "Men"), ("F", "Women")]:
    mask = (X_test["sex"] == s).to_numpy()
    yt, pp = y_test.to_numpy()[mask], proba[mask]
    auc = roc_auc_score(yt, pp) if len(np.unique(yt)) == 2 else np.nan
    rows.append({
        "group": label, "n": int(mask.sum()),
        "observed_rate_%": round(yt.mean() * 100, 1),
        "mean_predicted_%": round(pp.mean() * 100, 1),
        "roc_auc": round(auc, 3),
    })
print(pd.DataFrame(rows).to_string(index=False))

# ---------- 10. Per-patient risk report ----------
def risk_level(p):
    """Map a probability to a simple risk category."""
    if p < 0.30:
        return "Low"
    if p < 0.60:
        return "Medium"
    return "High"

report = pd.DataFrame({
    "patient_id": X_test[ID_COL].values,
    "age": X_test["age"].values,
    "sex": X_test["sex"].values,
    "menopause_status": X_test["menopause_status"].values,
    "risk_probability_pct": (proba * 100).round(1),
    "risk_level": [risk_level(p) for p in proba],
}).sort_values("risk_probability_pct", ascending=False)

report.to_csv("risk_report_demo.csv", index=False)
print("\n=== Top 10 highest-risk patients (demo sample) ===")
print(report.head(10).to_string(index=False))
print("\nRisk level counts:")
print(report["risk_level"].value_counts())
print("\nSaved: risk_report_demo.csv")

# Synthetic patient data generator (DEMO ONLY - no real patients)
import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)
N = 1000

# ---- Base clinical features ----
age = np.clip(rng.normal(54, 10, N), 29, 80).round().astype(int)
sex = rng.choice(["M", "F"], N, p=[0.62, 0.38])
chest_pain = rng.choice(
    ["typical_angina", "atypical_angina", "non_anginal", "asymptomatic"],
    N, p=[0.12, 0.2, 0.28, 0.40])
resting_bp = np.clip(rng.normal(131 + (age - 54) * 0.4, 17, N), 90, 200).round()
cholesterol = np.clip(rng.normal(215 + (age - 54) * 0.6, 45, N), 110, 450).round()
fasting_bs = (rng.random(N) < (0.10 + (age - 29) / 400)).astype(int)
resting_ecg = rng.choice(["normal", "st_t_abnormality", "lv_hypertrophy"],
                         N, p=[0.60, 0.22, 0.18])
max_hr = np.clip(rng.normal(205 - age * 0.9, 17, N), 70, 202).round()
exercise_angina = (rng.random(N) < 0.33).astype(int)
st_depression = np.clip(rng.gamma(1.4, 0.7, N), 0, 6).round(1)
smoker = (rng.random(N) < 0.30).astype(int)
bmi = np.clip(rng.normal(27.5, 4.5, N), 17, 46).round(1)
family_history = (rng.random(N) < 0.25).astype(int)

# ---- Menopause status (women only) ----
# Probability of being postmenopausal rises around age ~51 (typical
# natural menopause). Men get "not_applicable".
p_post = 1 / (1 + np.exp(-(age - 51) / 2.5))
is_post = rng.random(N) < p_post
menopause = np.where(sex == "M", "not_applicable",
                     np.where(is_post, "postmenopausal", "premenopausal"))

# ---- Hidden risk model used only to create realistic labels ----
# (assumption for the synthetic data: postmenopausal status adds risk
#  on top of age)
cp_eff = pd.Series(chest_pain).map({
    "typical_angina": 0.2, "atypical_angina": -0.2,
    "non_anginal": -0.4, "asymptomatic": 0.9}).values
z = (-4.6
     + 0.045 * age + 0.5 * (sex == "M")
     + 0.5 * (menopause == "postmenopausal")
     + cp_eff
     + 0.012 * (resting_bp - 130)
     + 0.004 * (cholesterol - 215)
     + 0.35 * fasting_bs
     + 0.30 * (resting_ecg != "normal")
     - 0.020 * (max_hr - 150)
     + 0.9 * exercise_angina
     + 0.55 * st_depression
     + 0.45 * smoker
     + 0.03 * (bmi - 27)
     + 0.40 * family_history
     + rng.normal(0, 0.6, N))
p = 1 / (1 + np.exp(-z))
target = (rng.random(N) < p).astype(int)

df = pd.DataFrame({
    "patient_id": [f"P{str(i).zfill(4)}" for i in range(1, N + 1)],
    "age": age, "sex": sex, "menopause_status": menopause,
    "chest_pain_type": chest_pain,
    "resting_bp": resting_bp, "cholesterol": cholesterol,
    "fasting_blood_sugar": fasting_bs, "resting_ecg": resting_ecg,
    "max_heart_rate": max_hr, "exercise_angina": exercise_angina,
    "st_depression": st_depression, "smoker": smoker,
    "bmi": bmi, "family_history": family_history,
    "heart_attack": target,
})

# ---- Inject exactly 180 missing cells (never in id/target) ----
miss_plan = {"cholesterol": 40, "resting_bp": 25, "max_heart_rate": 30,
             "bmi": 30, "st_depression": 20, "smoker": 20}  # = 165
for col, n_miss in miss_plan.items():
    idx = rng.choice(N, n_miss, replace=False)
    if col != "smoker" and df[col].dtype != object:
        df[col] = df[col].astype(float)
    df[col] = df[col].astype(float)
    df.loc[idx, col] = np.nan

# Menopause status can only be missing for women (15 cells -> total 180)
women_idx = df.index[df["sex"] == "F"].to_numpy()
idx = rng.choice(women_idx, 15, replace=False)
df.loc[idx, "menopause_status"] = np.nan

df.to_csv("patients_demo.csv", index=False)
print("Rows:", len(df), "| Missing cells:", int(df.isna().sum().sum()))
print("Target:", df["heart_attack"].value_counts().to_dict())
print("Women:", int((df['sex'] == 'F').sum()),
      "| Menopause:", df["menopause_status"].value_counts(dropna=False).to_dict())

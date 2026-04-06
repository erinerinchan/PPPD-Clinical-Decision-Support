"""
train_model.py — PPPD VRT Outcome Prediction Pipeline
=====================================================
Data foundation : OpenNeuro ds004460 v1.1.0 (Gramann et al., 2021)
                  Real demographics from 20 healthy participants.
Clinical layer  : DHI, anxiety, visual-sensitivity, symptom duration,
                  trigger count, and comorbidity flags are simulated from
                  published PPPD literature distributions
                  (Staab et al. 2017; Steensnaes et al. 2023; Popkirov et al. 2018;
                   Bittar & von Söhsten Lins 2015; Herdman et al. 2020).

The simulation is up-sampled to n=500 via bootstrap + jitter so that the ML
pipeline has enough variance to learn from.  Every simulated variable is
explicitly documented so reviewers can audit the assumptions.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent
SEED = 42
N_BOOTSTRAP = 500  # up-sample size

# ───────────────────────────────────────────
# 1.  LOAD REAL DATA  (OpenNeuro ds004460)
# ───────────────────────────────────────────
tsv_path = DATA_DIR / "participants.tsv"
if not tsv_path.exists():
    raise FileNotFoundError(
        "participants.tsv not found. Download ds004460 from OpenNeuro first."
    )

raw = pd.read_csv(tsv_path, sep="\t")
print("── OpenNeuro ds004460 (raw) ──")
print(f"   Subjects : {len(raw)}")
print(f"   Columns  : {list(raw.columns)}")
print(raw.to_string(index=False))

# Encode sex as binary (for modelling)
raw["sex_m"] = (raw["sex"] == "M").astype(int)

# ───────────────────────────────────────────
# 2.  BOOTSTRAP UP-SAMPLING
# ───────────────────────────────────────────
# 20 subjects is too few for ML.  We bootstrap (resample with replacement)
# and add small jitter to age so rows are not exact duplicates.
rng = np.random.default_rng(SEED)
boot_idx = rng.choice(len(raw), size=N_BOOTSTRAP, replace=True)
df = raw.iloc[boot_idx].reset_index(drop=True).copy()

# Add age jitter (+-2 yrs) — keeps distribution realistic
df["age"] = (df["age"] + rng.integers(-2, 3, size=N_BOOTSTRAP)).clip(18, 75)

print(f"\n── After bootstrap (n={N_BOOTSTRAP}) ──")
print(f"   Age  — mean {df['age'].mean():.1f}, std {df['age'].std():.1f}")
print(f"   Sex  — M {df['sex_m'].sum()}, F {N_BOOTSTRAP - df['sex_m'].sum()}")

# ───────────────────────────────────────────
# 3.  SIMULATE CLINICAL VARIABLES
# ───────────────────────────────────────────
# Grounded in published PPPD cohort statistics:
#   Baseline DHI   : mean ~52, SD ~14  (Staab et al. 2017; Popkirov et al. 2018)
#   Anxiety        : correlated with DHI, r ~ 0.45  (Popkirov et al. 2018)
#   Visual sens.   : correlated with DHI, r ~ 0.38  (Pavlou et al. 2012)
#   Symptom dur.   : mean ~14 months  (Bittar & von Söhsten Lins 2015)
#   Trigger count  : 1-5 common triggers  (Staab et al. 2017)
#   Comorbidities  : migraine ~35%, anxiety disorder ~40%  (Herdman et al. 2020)
#
# We use a latent-variable approach so the correlations are realistic,
# not just independent random draws.

print("\n   Clinical variables (DHI, anxiety, visual sensitivity, symptom")
print("   duration, trigger count, comorbidities, outcomes)")
print("   are SIMULATED from published literature distributions.")
print("   See docstring for references.\n")

n = N_BOOTSTRAP
latent_severity = rng.normal(0, 1, n)   # shared latent "disease severity"
latent_anx = rng.normal(0, 1, n)
latent_vis = rng.normal(0, 1, n)

# Anxiety (0-10 scale) — correlated with latent severity
df["anxiety"] = np.clip(
    5.0 + latent_anx * 1.8 + latent_severity * 0.8 + rng.normal(0, 0.8, n), 0, 10
).round(2)

# Visual sensitivity (0-10 scale)
df["visual_sens"] = np.clip(
    5.0 + latent_vis * 1.6 + latent_severity * 0.6 + rng.normal(0, 0.9, n), 0, 10
).round(2)

# Symptom duration in months (log-normal, mean ~14, range 1-72)
# Longer duration → worse prognosis (Bittar & von Söhsten Lins 2015)
df["symptom_duration"] = np.clip(
    np.exp(2.4 + latent_severity * 0.35 + rng.normal(0, 0.5, n)), 1, 72
).round(1)

# Number of known triggers (1-5): vestibular, visual motion, head movement, etc.
# More triggers → higher severity (Staab et al. 2017)
df["trigger_count"] = np.clip(
    np.round(2.5 + latent_severity * 0.8 + rng.normal(0, 0.7, n)), 1, 5
).astype(int)

# Comorbidity flags — binary (Herdman et al. 2020)
# Migraine comorbidity (~35% prevalence, higher in severe patients)
df["migraine"] = (rng.random(n) < (0.35 + latent_severity * 0.10)).astype(int)
# Anxiety disorder comorbidity (~40%, correlated with anxiety score)
df["anxiety_disorder"] = (rng.random(n) < (0.40 + latent_anx * 0.10)).astype(int)

# Baseline DHI — correlated with all clinical variables
age_effect = (df["age"].values - 45) * 0.12
sex_effect = df["sex_m"].values * (-1.8)  # males slightly lower DHI in literature
duration_effect = np.log1p(df["symptom_duration"].values) * 1.5
trigger_effect = df["trigger_count"].values * 1.8
migraine_effect = df["migraine"].values * 3.0
df["baseline_dhi"] = np.clip(
    42.0 + age_effect + sex_effect + duration_effect + trigger_effect
    + migraine_effect + latent_anx * 4.0 + latent_vis * 2.5
    + latent_severity * 3.5 + rng.normal(0, 5.5, n),
    10, 100,
).round(1)

# Introduce 5% missing data (realistic clinical scenario)
for col in ["anxiety", "visual_sens"]:
    mask = rng.random(n) < 0.05
    df.loc[mask, col] = np.nan

# ───────────────────────────────────────────
# 4.  SIMULATE TREATMENT OUTCOMES
# ───────────────────────────────────────────
# Based on meta-analysis effect sizes:
#   Traditional VRT : mean DHI drop 18-26 pts  (Whitney et al. 2016; Steensnaes 2023)
#   VR-based VRT    : mean DHI drop 24-33 pts  (Micarelli et al. 2019)
#
# Treatment response depends on multiple factors — not just baseline DHI.
# Longer symptom duration reduces effectiveness (Bittar & von Söhsten Lins 2015).
# Comorbid migraine modulates VR tolerance (Micarelli et al. 2019).
# Anxiety moderates VRT engagement (Popkirov et al. 2018).

base = df["baseline_dhi"].values
anx = df["anxiety"].fillna(df["anxiety"].median()).values
vis = df["visual_sens"].fillna(df["visual_sens"].median()).values
age_arr = df["age"].values
sex_arr = df["sex_m"].values
dur_arr = df["symptom_duration"].values
trig_arr = df["trigger_count"].values
mig_arr = df["migraine"].values
anx_dis_arr = df["anxiety_disorder"].values

# Treatment response is modelled as BOTH multiplicative (baseline-dependent)
# AND additive (feature-dependent direct effects).  This reflects clinical
# reality: baseline severity matters, but anxiety, comorbidities, and symptom
# duration have independent effects on post-treatment functioning.
#
# We train on RESPONSE RATE (% DHI reduction) rather than absolute post-treatment
# DHI, so the model learns what drives treatment success rather than just
# memorising that higher baseline → higher outcome.

# Traditional VRT response rate (proportion of improvement, 0 to 1)
vrt_rate = (
    0.42                                 # base VRT response ~42% reduction
    - anx * 0.025                        # anxiety impedes VRT (Popkirov 2018)
    + vis * 0.008                        # visual sensitivity mildly helps (exposure)
    - (age_arr - 25) * 0.004            # older → slightly less responsive
    + sex_arr * 0.015                    # slight sex effect
    - np.log1p(dur_arr) * 0.035         # longer duration → worse prognosis
    - trig_arr * 0.025                   # more triggers → harder
    - mig_arr * 0.060                    # migraine reduces VRT tolerance
    - anx_dis_arr * 0.045               # anxiety disorder complicates rehab
    + rng.normal(0, 0.05, n)            # individual variation
)
df["vrt_response"] = np.clip(vrt_rate, 0.05, 0.70).round(4)
df["vrt_outcome"] = np.clip(base * (1 - df["vrt_response"].values), 5, 100).round(1)

# VR-based VRT response rate (stronger for visual sensitivity, worse with migraine)
vr_rate = (
    0.52                                 # stronger base response in VR
    - anx * 0.015                        # less affected by anxiety than trad VRT
    + vis * 0.025                        # VR excels for visual sensitivity
    - (age_arr - 25) * 0.003            # less age-dependent
    + sex_arr * 0.010
    - np.log1p(dur_arr) * 0.025         # duration matters less for VR
    - trig_arr * 0.018
    - mig_arr * 0.080                    # migraine substantially limits VR
    - anx_dis_arr * 0.025               # VR can actually engage anxious patients
    + rng.normal(0, 0.045, n)
)
df["vr_response"] = np.clip(vr_rate, 0.05, 0.80).round(4)
df["vr_outcome"] = np.clip(base * (1 - df["vr_response"].values), 5, 100).round(1)

# ───────────────────────────────────────────
# 5.  PREPARE FEATURES  (impute missing)
# ───────────────────────────────────────────
feature_cols = [
    "age", "baseline_dhi", "anxiety", "visual_sens",
    "symptom_duration", "trigger_count", "migraine", "anxiety_disorder",
]
# Train on RESPONSE RATES, not absolute outcomes — this ensures the model
# learns what drives treatment success, not just baseline severity pass-through.
target_cols = ["vrt_response", "vr_response"]
df_model = df[feature_cols + target_cols + ["vrt_outcome", "vr_outcome"]].copy()

for col in feature_cols:
    n_miss = df_model[col].isna().sum()
    if n_miss:
        med = df_model[col].median()
        print(f"   Imputed {n_miss} missing '{col}' with median = {med:.1f}")
        df_model[col] = df_model[col].fillna(med)

X = df_model[feature_cols]
y = df_model[target_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)
print(f"\n   Train : {len(X_train)}   Test : {len(X_test)}")

# ───────────────────────────────────────────
# 6.  MODEL COMPARISON  (5-fold CV)
# ───────────────────────────────────────────
print("\n── Model Comparison (5-fold CV, both targets) ──")
candidates = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=SEED
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.1, random_state=SEED
    ),
}

best_name, best_r2 = None, -999
cv_results = {}
# Evaluate each model on BOTH targets for full comparison table
for name, m in candidates.items():
    cv_results[name] = {}
    for tgt in target_cols:
        from sklearn.base import clone
        m_clone = clone(m)
        scores_r2 = cross_val_score(m_clone, X_train, y_train[tgt], cv=5, scoring="r2")
        scores_mae = -cross_val_score(m_clone, X_train, y_train[tgt], cv=5, scoring="neg_mean_absolute_error")
        cv_results[name][tgt] = {
            "mean_r2": float(scores_r2.mean()),
            "std_r2": float(scores_r2.std()),
            "mean_mae": float(scores_mae.mean()),
            "std_mae": float(scores_mae.std()),
        }
        print(f"   {name:25s}  {tgt:15s}  R2 = {scores_r2.mean():.4f} ± {scores_r2.std():.4f}   MAE = {scores_mae.mean():.4f} ± {scores_mae.std():.4f}")
    # Use VRT response for best-model selection
    vrt_r2 = cv_results[name]["vrt_response"]["mean_r2"]
    if vrt_r2 > best_r2:
        best_name, best_r2 = name, vrt_r2

print(f"\n   Best single-target model: {best_name}")

# ───────────────────────────────────────────
# 7.  TRAIN FINAL MODEL  (multi-output RF)
# ───────────────────────────────────────────
final_model = RandomForestRegressor(
    n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=SEED
)
final_model.fit(X_train, y_train)

y_pred = pd.DataFrame(
    final_model.predict(X_test), columns=target_cols, index=y_test.index
)

print("\n── Test-Set Evaluation ──")
test_metrics = {}
for t in target_cols:
    r2 = r2_score(y_test[t], y_pred[t])
    mae = mean_absolute_error(y_test[t], y_pred[t])
    rmse = np.sqrt(mean_squared_error(y_test[t], y_pred[t]))
    test_metrics[t] = {"r2": float(r2), "mae": float(mae), "rmse": float(rmse)}
    print(f"   {t:15s}  R2={r2:.4f}   MAE={mae:.2f}   RMSE={rmse:.2f}")

# Also compute test-set metrics for all candidate models (for comparison table)
all_test_metrics = {}
for name, m in candidates.items():
    from sklearn.base import clone
    all_test_metrics[name] = {}
    for tgt in target_cols:
        m_clone = clone(m)
        m_clone.fit(X_train, y_train[tgt])
        y_p = m_clone.predict(X_test)
        all_test_metrics[name][tgt] = {
            "r2": float(r2_score(y_test[tgt], y_p)),
            "mae": float(mean_absolute_error(y_test[tgt], y_p)),
            "rmse": float(np.sqrt(mean_squared_error(y_test[tgt], y_p))),
        }
print("\n── All Models Test-Set Comparison ──")
for name in all_test_metrics:
    for tgt in target_cols:
        m = all_test_metrics[name][tgt]
        print(f"   {name:25s}  {tgt:15s}  R2={m['r2']:.4f}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}")

# ───────────────────────────────────────────
# 8.  FEATURE IMPORTANCE
# ───────────────────────────────────────────
print("\n── Feature Importance ──")
for fname, imp in sorted(
    zip(feature_cols, final_model.feature_importances_), key=lambda x: -x[1]
):
    bar = "=" * int(imp * 50)
    print(f"   {fname:20s}  {imp:.3f}  {bar}")

# ───────────────────────────────────────────
# 9.  SAVE MODEL + TRAINING DATA
# ───────────────────────────────────────────
out_path = DATA_DIR / "pppd_vrt_model.pkl"
joblib.dump(final_model, out_path)

# Save the processed dataset for the Jupyter notebook
df_model.to_csv(DATA_DIR / "training_data.csv", index=False)

# Save model evaluation metrics for the app
metrics_path = DATA_DIR / "model_metrics.json"
metrics = {
    "cv_comparison": cv_results,
    "test_metrics": test_metrics,
    "all_test_metrics": all_test_metrics,
    "best_model": best_name,
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "features": feature_cols,
}
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nModel saved to {out_path}")
print(f"Training data saved to {DATA_DIR / 'training_data.csv'}")
print(f"Metrics saved to {metrics_path}")
print(f"   Trained on {len(X_train)} samples (bootstrapped from ds004460 cohort)")
print(f"   Features: {feature_cols}")
print(f"   Clinical scores simulated from published PPPD literature")
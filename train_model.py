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

Data generation is handled by simulate.py — see generate_training_data().
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

from simulate import generate_training_data

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent
SEED = 42

# ───────────────────────────────────────────
# 1-4.  GENERATE TRAINING DATA
# ───────────────────────────────────────────
# Loads OpenNeuro ds004460, bootstraps to n=500, simulates clinical
# variables and treatment outcomes.  See simulate.py for full details.
df_model = generate_training_data(data_dir=DATA_DIR, seed=SEED)

feature_cols = [
    "age", "baseline_dhi", "anxiety", "visual_sens",
    "symptom_duration", "trigger_count", "migraine", "anxiety_disorder",
]
target_cols = ["vrt_response", "vr_response"]

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
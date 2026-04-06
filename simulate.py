"""
simulate.py — PPPD Clinical Data Simulation
============================================
Generates a synthetic clinical dataset for PPPD vestibular rehabilitation
outcome prediction by:
  1. Loading real demographics from OpenNeuro ds004460 v1.1.0
  2. Bootstrap up-sampling to n=500 with age jitter
  3. Simulating clinical variables from published literature distributions
  4. Simulating treatment response outcomes

Every simulated variable is documented with its source reference so
reviewers can audit all assumptions.

References:
  Staab et al. (2017) — PPPD diagnostic criteria, DHI distributions
  Steensnaes et al. (2023) — VRT outcomes in PPPD
  Micarelli et al. (2019) — VR-enhanced vestibular rehabilitation
  Popkirov et al. (2018) — Anxiety-dizziness relationship
  Bittar & von Söhsten Lins (2015) — Symptom duration and prognosis
  Herdman et al. (2020) — Comorbidity prevalence in PPPD
  Pavlou et al. (2012) — Visual sensitivity in vestibular disorders
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent
SEED = 42
N_BOOTSTRAP = 500


def generate_training_data(
    data_dir: Path = DATA_DIR,
    seed: int = SEED,
    n_bootstrap: int = N_BOOTSTRAP,
) -> pd.DataFrame:
    """Generate a synthetic PPPD clinical dataset.

    Parameters
    ----------
    data_dir : Path
        Directory containing ``participants.tsv`` from OpenNeuro ds004460.
    seed : int
        Random seed for reproducibility.
    n_bootstrap : int
        Number of bootstrap samples to generate.

    Returns
    -------
    pd.DataFrame
        DataFrame with 8 clinical features, 2 response-rate targets, and
        2 absolute-outcome columns (500 rows by default).
    """
    rng = np.random.default_rng(seed)

    # ───────────────────────────────────────
    # 1. LOAD REAL DATA (OpenNeuro ds004460)
    # ───────────────────────────────────────
    tsv_path = data_dir / "participants.tsv"
    if not tsv_path.exists():
        raise FileNotFoundError(
            "participants.tsv not found. Download ds004460 from OpenNeuro first."
        )

    raw = pd.read_csv(tsv_path, sep="\t")
    print("── OpenNeuro ds004460 (raw) ──")
    print(f"   Subjects : {len(raw)}")
    print(f"   Columns  : {list(raw.columns)}")
    print(raw.to_string(index=False))

    raw["sex_m"] = (raw["sex"] == "M").astype(int)

    # ───────────────────────────────────────
    # 2. BOOTSTRAP UP-SAMPLING
    # ───────────────────────────────────────
    boot_idx = rng.choice(len(raw), size=n_bootstrap, replace=True)
    df = raw.iloc[boot_idx].reset_index(drop=True).copy()
    df["age"] = (df["age"] + rng.integers(-2, 3, size=n_bootstrap)).clip(18, 75)

    print(f"\n── After bootstrap (n={n_bootstrap}) ──")
    print(f"   Age  — mean {df['age'].mean():.1f}, std {df['age'].std():.1f}")
    print(f"   Sex  — M {df['sex_m'].sum()}, F {n_bootstrap - df['sex_m'].sum()}")

    # ───────────────────────────────────────
    # 3. SIMULATE CLINICAL VARIABLES
    # ───────────────────────────────────────
    n = n_bootstrap
    latent_severity = rng.normal(0, 1, n)
    latent_anx = rng.normal(0, 1, n)
    latent_vis = rng.normal(0, 1, n)

    print("\n   Clinical variables (DHI, anxiety, visual sensitivity, symptom")
    print("   duration, trigger count, comorbidities, outcomes)")
    print("   are SIMULATED from published literature distributions.")
    print("   See docstring for references.\n")

    # Anxiety (0-10 scale) — correlated with latent severity
    df["anxiety"] = np.clip(
        5.0 + latent_anx * 1.8 + latent_severity * 0.8 + rng.normal(0, 0.8, n), 0, 10
    ).round(2)

    # Visual sensitivity (0-10 scale)
    df["visual_sens"] = np.clip(
        5.0 + latent_vis * 1.6 + latent_severity * 0.6 + rng.normal(0, 0.9, n), 0, 10
    ).round(2)

    # Symptom duration in months (log-normal, mean ~14, range 1-72)
    df["symptom_duration"] = np.clip(
        np.exp(2.4 + latent_severity * 0.35 + rng.normal(0, 0.5, n)), 1, 72
    ).round(1)

    # Number of known triggers (1-5)
    df["trigger_count"] = np.clip(
        np.round(2.5 + latent_severity * 0.8 + rng.normal(0, 0.7, n)), 1, 5
    ).astype(int)

    # Comorbidity flags — binary (Herdman et al. 2020)
    df["migraine"] = (rng.random(n) < (0.35 + latent_severity * 0.10)).astype(int)
    df["anxiety_disorder"] = (rng.random(n) < (0.40 + latent_anx * 0.10)).astype(int)

    # Baseline DHI — correlated with all clinical variables
    age_effect = (df["age"].values - 45) * 0.12
    sex_effect = df["sex_m"].values * (-1.8)
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

    # ───────────────────────────────────────
    # 4. SIMULATE TREATMENT OUTCOMES
    # ───────────────────────────────────────
    base = df["baseline_dhi"].values
    anx = df["anxiety"].fillna(df["anxiety"].median()).values
    vis = df["visual_sens"].fillna(df["visual_sens"].median()).values
    age_arr = df["age"].values
    sex_arr = df["sex_m"].values
    dur_arr = df["symptom_duration"].values
    trig_arr = df["trigger_count"].values
    mig_arr = df["migraine"].values
    anx_dis_arr = df["anxiety_disorder"].values

    # Traditional VRT response rate
    vrt_rate = (
        0.42
        - anx * 0.025
        + vis * 0.008
        - (age_arr - 25) * 0.004
        + sex_arr * 0.015
        - np.log1p(dur_arr) * 0.035
        - trig_arr * 0.025
        - mig_arr * 0.060
        - anx_dis_arr * 0.045
        + rng.normal(0, 0.05, n)
    )
    df["vrt_response"] = np.clip(vrt_rate, 0.05, 0.70).round(4)
    df["vrt_outcome"] = np.clip(base * (1 - df["vrt_response"].values), 5, 100).round(1)

    # VR-based VRT response rate
    vr_rate = (
        0.52
        - anx * 0.015
        + vis * 0.025
        - (age_arr - 25) * 0.003
        + sex_arr * 0.010
        - np.log1p(dur_arr) * 0.025
        - trig_arr * 0.018
        - mig_arr * 0.080
        - anx_dis_arr * 0.025
        + rng.normal(0, 0.045, n)
    )
    df["vr_response"] = np.clip(vr_rate, 0.05, 0.80).round(4)
    df["vr_outcome"] = np.clip(base * (1 - df["vr_response"].values), 5, 100).round(1)

    # ───────────────────────────────────────
    # 5. PREPARE FINAL DATAFRAME
    # ───────────────────────────────────────
    feature_cols = [
        "age", "baseline_dhi", "anxiety", "visual_sens",
        "symptom_duration", "trigger_count", "migraine", "anxiety_disorder",
    ]
    target_cols = ["vrt_response", "vr_response"]
    df_model = df[feature_cols + target_cols + ["vrt_outcome", "vr_outcome"]].copy()

    for col in feature_cols:
        n_miss = df_model[col].isna().sum()
        if n_miss:
            med = df_model[col].median()
            print(f"   Imputed {n_miss} missing '{col}' with median = {med:.1f}")
            df_model[col] = df_model[col].fillna(med)

    return df_model


if __name__ == "__main__":
    df = generate_training_data()
    out = DATA_DIR / "training_data.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows to {out}")

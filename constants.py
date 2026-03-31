"""
constants.py — Literature-derived constants for the PPPD CDSS
References:
  - Staab et al. (2017) — Barany Society PPPD diagnostic criteria
  - Steensnaes et al. (2023) — VRT outcomes in PPPD
  - Whitney et al. (2016) — Vestibular rehabilitation meta-analysis
  - Micarelli et al. (2019) — VR-enhanced vestibular rehabilitation
  - Popkirov et al. (2018) — Anxiety-dizziness relationship
  - Jacobson & Newman (1990) — DHI development and scoring
  - Bittar & von Söhsten Lins (2015) — Symptom duration and prognosis
  - Herdman et al. (2020) — Comorbidity prevalence in vestibular disorders
"""

# Severity thresholds (Jacobson & Newman, 1990)
# Mild: 0-30 | Moderate: 31-60 | Severe: 61-100
DHI_THRESHOLD_MILD = 30
DHI_THRESHOLD_MODERATE = 60
DHI_THRESHOLD_SEVERE = 61

# DHI sub-scale question indices (0-based, matching the 25-item DHI)
DHI_PHYSICAL_QS = [0, 3, 4, 7, 10, 12, 13, 16, 18, 24]
DHI_EMOTIONAL_QS = [1, 5, 9, 14, 15, 19, 20, 21, 22]
DHI_FUNCTIONAL_QS = [2, 6, 8, 11, 17, 23]

# Data source
DATASET_DOI = "doi:10.18112/openneuro.ds004460.v1.1.0"
DATASET_REF = (
    "Gramann, K., Hohlefeld, F.U., Gehrke, L. & Klug, M. (2021). "
    "Human cortical dynamics during full-body heading changes. "
    "Sci Rep 11, 18186."
)
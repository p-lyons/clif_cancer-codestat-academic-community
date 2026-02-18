# CLIF Code Status Analysis

## Research Question

**Among hospitalized adult patients admitted through the ED, does the association between active cancer and initial full code status differ between academic medical centers and community hospitals?**

**Hypothesis:** The higher rate of full code status among cancer patients at AMCs reflects selection bias, where patients presenting to AMCs for advanced treatment or clinical trials are systematically different from cancer patients admitted to community hospitals.

## Requirements

### Python Dependencies
```bash
pip install -r requirements.txt
```

### CLIF Tables Required
- `clif_hospitalization.parquet` (or .csv)
- `clif_adt.parquet`
- `clif_patient.parquet`
- `clif_hospital_diagnosis.parquet`
- `clif_vitals.parquet`
- `clif_code_status.parquet`

### File Structure
```
project/
├── config/
│   └── config.yaml          # Site configuration (not committed to git)
├── 01_cohort_code_status.py  # Main analysis script
├── upload_to_box/            # Created automatically — poolable outputs
├── proj_tables/              # Created automatically — intermediate data
└── README.md
```

## Setup

1. **Create config file:**
   ```bash
   mkdir -p config
   cp config_template.yaml config/config.yaml
   ```

2. **Edit config.yaml with your site details:**
   ```yaml
   site_lowercase: "your_site"
   tables_location: "/path/to/clif/tables"
   file_type: "parquet"
   start_date: "2018-01-01"
   end_date: "2024-12-31"
   ```

3. **Verify your CLIF tables are present:**
   ```bash
   ls /path/to/your/clif/data/clif_*.parquet
   ```

4. **Run the analysis:**
   ```bash
   python 01_cohort_code_status.py
   ```

   For batch/automated execution (exits on test failure instead of prompting):
   ```bash
   python 01_cohort_code_status.py --no-interactive
   ```

**Expected runtime:** 5–20 minutes depending on data size.

## Analysis Workflow

### Cohort Definition

1. **Base cohort:**
   - Adult patients (≥18 years)
   - Hospitalized during study period
   - Admitted through ED
   - Had ward stay
   - Had complete vital signs on wards (HR, RR, SBP, SpO2, temp)
   - Excludes: OB/psych/rehab, LTACH

2. **Encounter linkage:**
   - Hospitalizations within 6 hours (including overlapping encounters) are linked into single episodes

3. **Cancer classification (7-group hierarchy):**
   - **Active cancer:** ICD-10 C00–C96 (excludes Z85 history codes and C44 non-melanoma skin cancer)
   - **Hierarchy:** Metastatic > Hematologic > Lung > GI/HPB > Breast > GU > Other solid
   - **Stratification:** Heme vs solid, metastatic vs non-metastatic

4. **Code status:**
   - Extracted from first 12 hours after admission
   - **Binary outcome:** Full code (includes "presume_full" if no code status documented) vs other
   - DNR/DNI coded as "other"
   - Imputation rate tracked and reported in outputs

5. **Hospital type:**
   - From `adt.hospital_type` field
   - Academic vs Community (LTACH excluded)

6. **Comorbidity:**
   - van Walraven Elixhauser score
   - Cancer-related categories (lymphoma, metastatic, tumor) are zeroed because cancer is the primary exposure

7. **Race/ethnicity:**
   - Collapsed to non-Hispanic White (1) vs everyone else (0)
   - Missing/unknown coded as 0

### Statistical Analysis

**Primary Model:**
```
logit(full_code) ~ cancer + hosp_type + cancer:hosp_type +
                   age + female + nhw + elixhauser_score
```

- Clustered standard errors at `hospital_id` level when ≥10 hospitals; HC1 robust SEs otherwise
- Falls back to main-effects model (no interaction) when only one hospital type present
- Outputs coefficient vector and variance-covariance matrix for random-effects meta-analysis pooling

**Expected Pattern (if hypothesis correct):**
- At community hospitals: Cancer → lower odds of full code
- At AMCs: Cancer → higher/similar odds of full code
- Significant interaction term

### Defensive Unit Tests

The script runs 3 validation tests:

1. **Cancer Classification Test:** Checks cancer prevalence (5–50% expected), validates heme vs solid classification
2. **Data Completeness Test:** Checks missingness for critical variables, validates outcome distribution (30–95% full code expected), verifies multiple hospital types, reports code status imputation rate
3. **ED Admission Filter Test:** Verifies 100% of cohort are ED admissions

Tests pause execution if failures are detected (or exit in `--no-interactive` mode).

## Outputs

All outputs saved to `upload_to_box/`:

### Required for Pooling
| File | Contents |
|------|----------|
| `regression_results_[site].csv` | Coefficients, SEs, ORs, 95% CIs, n_clusters, SE method |
| `vcov_matrix_[site].csv` | Full variance-covariance matrix for REMA |
| `site_summary_[site].csv` | Sample sizes, dates, cancer/full-code/hospital-type counts |
| `table1_continuous_[site].csv` | Age, comorbidity, LOS by cancer × hospital type |
| `table1_categorical_[site].csv` | Demographics, outcomes by cancer × hospital type |
| `cancer_groups_[site].csv` | Cancer group distribution by hospital type |
| `analysis_log_[site].txt` | Full log of the analysis run |

### Intermediate Files (in `proj_tables/`)
- `cohort_[site].parquet` — Full analysis dataset (not shared)

## Interpreting Your Results

### Key Variables in `regression_results_[site].csv`:

| Variable | Interpretation |
|----------|----------------|
| `ca_01` | Effect of cancer at **community** hospitals (reference group) |
| `hosp_type` | Effect of academic hospital among **non-cancer** patients |
| `ca_01:hosp_type` | **INTERACTION** — does the cancer effect differ by hospital type? |

### The Interaction Term

- **Positive coefficient** = Cancer has a weaker negative (or more positive) effect on full code at AMCs
- **p < 0.05** = Evidence that the relationship differs by hospital type
- **This supports the selection bias hypothesis**

### Example:

```
ca_01:              OR = 0.75  → Cancer → lower odds of full code at community
ca_01:hosp_type:    OR = 1.45  → DIFFERENT relationship at academic centers
Combined at AMC:    0.75 × 1.45 = 1.09 → similar/higher odds of full code
```

## Quality Control Checklist

Before submitting data, verify:

- [ ] All unit tests passed
- [ ] Cancer rate: 5–50% of cohort
- [ ] Full code rate: 30–95% of cohort
- [ ] Both hospital types present
- [ ] ED admission rate: 100%
- [ ] Age missingness: <10%
- [ ] Code status imputation rate documented
- [ ] Site summary statistics reasonable
- [ ] Review `analysis_log_[site].txt` for warnings

## Troubleshooting

### "config.yaml not found"
Create `config/config.yaml` with required fields (see Setup section).

### "Required table not found"
Check that `tables_location` path is correct and contains files like `clif_hospitalization.parquet`.

### "Table validation failed"
The script checks that required columns and expected values exist in each table. Review the validation error messages — they will identify the specific table and missing column or value.

### "statsmodels not installed"
```bash
pip install statsmodels
```

### Unit test warnings
Review the specific warning and validate your data. Type "yes" when prompted if warnings are minor. Use `--no-interactive` for batch runs (script exits on failure).

### "Only one hospital type"
The script will fit a main-effects model without the interaction term. Report this to the coordinating center.

### Low sample size
Minimum recommended: ~500 encounters with cancer, ideally split across hospital types.

## After Running

1. **Review outputs locally** before sharing
2. **Check `site_summary_[site].csv`** for reasonableness (N, dates, cancer rate)
3. **Review `analysis_log_[site].txt`** for any warnings
4. **Upload all files from `upload_to_box/`** to the shared location
5. **Notify the coordinating center** that you've uploaded

## Contact

For questions about analysis methods or data issues:
- Check CLIF documentation: https://github.com/kaveri-s/clif
- Contact the study coordinator with your site name, the specific error message, and your analysis log file

## Version History

- v1.1 (2025-02-18): Revised
  - Fixed code status join (hospitalization_id path, not patient_id)
  - Fixed encounter linkage for overlapping encounters
  - 7-group cancer hierarchy (metastatic, heme, lung, GI/HPB, breast, GU, other solid)
  - Vectorized cancer classification
  - Race/ethnicity collapsed to non-Hispanic White vs other
  - Zeroed cancer-related Elixhauser categories (collinearity with exposure)
  - Adaptive SE method: clustered (≥10 hospitals) or HC1 robust (<10)
  - Falls back to main-effects model when single hospital type
  - Outputs VCV matrix for REMA pooling
  - Table validation after loading (column and value checks)
  - Datetime parsing for CSV files
  - Code status imputation rate tracked
  - `full_code_01` created before unit tests run
  - LTACH exclusion consolidated to single location
  - Logging to file and console
  - `--no-interactive` flag for batch execution
- v1.0 (2025-02-18): Initial release
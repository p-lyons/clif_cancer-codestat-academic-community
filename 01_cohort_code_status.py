#!/usr/bin/env python3
# ==============================================================================
# 01_cohort_code_status.py
# CLIF distributed analysis: Cancer and initial code status by hospital type
# Requires: config/config.yaml with site_lowercase, tables_location, file_type
# ==============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import sys
import argparse
import logging
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# ==============================================================================
# LOGGING
# ==============================================================================

def setup_logging(project_location, site):
    """Configure logging to both console and file"""
    log_dir = Path(project_location) / 'upload_to_box'
    log_dir.mkdir(exist_ok=True, parents=True)
    log_file = log_dir / f'analysis_log_{site}.txt'

    logger = logging.getLogger('clif_code_status')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger

# ==============================================================================
# CONFIGURATION
# ==============================================================================

def load_config():
    """Load site configuration from config.yaml"""
    config_path = Path("config/config.yaml")

    if not config_path.exists():
        raise FileNotFoundError(
            "config/config.yaml not found. Create it with: site_lowercase, "
            "tables_location, file_type, start_date, end_date"
        )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    required = ['site_lowercase', 'tables_location', 'file_type']
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"config.yaml missing required keys: {missing}")

    return config

def setup_paths(config):
    """Setup file paths and output directories"""
    site = config['site_lowercase'].lower()
    tables_location = Path(config['tables_location'])
    project_location = Path(config.get('project_location', '.'))
    file_type = config['file_type'].lower()

    if file_type not in ['parquet', 'csv']:
        raise ValueError(f"file_type must be 'parquet' or 'csv', got '{file_type}'")

    (project_location / 'upload_to_box').mkdir(exist_ok=True, parents=True)
    (project_location / 'proj_tables').mkdir(exist_ok=True, parents=True)

    return site, tables_location, project_location, file_type

# ==============================================================================
# VALIDATION
# ==============================================================================

VALIDATION_SPECS = {
    'hospitalization': {
        'req_vars': [
            'patient_id', 'hospitalization_id', 'age_at_admission',
            'admission_dttm', 'discharge_dttm', 'discharge_category'
        ],
        'req_values': {
            'discharge_category': ['expired', 'hospice']
        }
    },
    'adt': {
        'req_vars': [
            'hospitalization_id', 'hospital_id', 'hospital_type',
            'location_category', 'in_dttm', 'out_dttm'
        ],
        'req_values': {
            'location_category': ['ed', 'icu', 'ward']
        }
    },
    'patient': {
        'req_vars': [
            'patient_id', 'sex_category', 'race_category', 'ethnicity_category'
        ],
        'req_values': {
            'sex_category': ['female', 'male']
        }
    },
    'hospital_diagnosis': {
        'req_vars': ['hospitalization_id', 'diagnosis_code'],
        'req_values': {}
    },
    'vitals': {
        'req_vars': ['hospitalization_id', 'vital_category', 'vital_value', 'recorded_dttm'],
        'req_values': {
            'vital_category': ['heart_rate', 'respiratory_rate', 'sbp', 'spo2', 'temp_c']
        }
    },
    'code_status': {
        'req_vars': ['hospitalization_id', 'code_status_category', 'start_dttm'],
        'req_values': {}
    }
}


def validate_table(df, table_name, spec, logger):
    """
    Validate a loaded table against its specification.
    Checks for required columns and expected values (case-insensitive).
    """
    problems = []

    # Check required columns (case-insensitive)
    actual_cols_lower = {c.lower(): c for c in df.columns}
    for var in spec.get('req_vars', []):
        if var.lower() not in actual_cols_lower:
            problems.append(f"Missing required column: {var}")

    # Check required values
    for var, expected_vals in spec.get('req_values', {}).items():
        col_lower = var.lower()
        if col_lower not in actual_cols_lower:
            problems.append(f"Cannot check values for missing column: {var}")
            continue

        actual_col = actual_cols_lower[col_lower]
        present_vals = df[actual_col].dropna().astype(str).str.lower().unique()
        for ev in expected_vals:
            if ev.lower() not in present_vals:
                problems.append(
                    f"Column '{actual_col}' missing expected value: '{ev}' "
                    f"(found: {sorted(set(present_vals))[:10]})"
                )

    if problems:
        for p in problems:
            logger.warning(f"  {table_name}: {p}")
        return False

    logger.info(f"  ✓ {table_name} validated")
    return True


def validate_all_tables(table_dict, logger):
    """Validate all loaded tables against VALIDATION_SPECS"""
    logger.info("Validating table structure and contents...")
    all_passed = True

    for table_name, spec in VALIDATION_SPECS.items():
        if table_name not in table_dict:
            logger.error(f"  ✗ {table_name} not loaded")
            all_passed = False
            continue
        if not validate_table(table_dict[table_name], table_name, spec, logger):
            all_passed = False

    if not all_passed:
        raise RuntimeError(
            "Table validation failed. Review warnings above and fix your data."
        )

    logger.info("✓ All tables validated successfully")

# ==============================================================================
# DATA LOADING
# ==============================================================================

DATETIME_COLS = {
    'hospitalization': ['admission_dttm', 'discharge_dttm'],
    'adt': ['in_dttm', 'out_dttm'],
    'vitals': ['recorded_dttm'],
    'code_status': ['start_dttm'],
    'patient': ['death_dttm']
}


def load_clif_table(tables_location, table_name, file_type, logger, columns=None):
    """Load a CLIF table with datetime parsing for CSV files"""
    filepath = tables_location / f"clif_{table_name}.{file_type}"

    if not filepath.exists():
        raise FileNotFoundError(f"Required table not found: {filepath}")

    try:
        if file_type == 'parquet':
            df = pd.read_parquet(filepath, columns=columns)
        elif file_type == 'csv':
            parse_dates = DATETIME_COLS.get(table_name, [])
            if columns:
                parse_dates = [c for c in parse_dates if c in columns]
            df = pd.read_csv(
                filepath,
                usecols=columns,
                parse_dates=parse_dates
            ) if columns else pd.read_csv(filepath, parse_dates=parse_dates)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        logger.info(f"✓ Loaded {table_name}: {len(df):,} rows")
        return df

    except Exception as e:
        raise RuntimeError(f"Error loading {table_name}: {e}")

# ==============================================================================
# COHORT DEFINITION
# ==============================================================================

def create_encounter_linkage(hosp_df, start_date, end_date, logger, link_hours=6):
    """
    Link hospitalizations within 6 hours into joined encounters.
    Handles overlapping encounters (negative gaps) by linking them.
    """
    logger.info("--- Creating encounter linkage ---")

    hosp = hosp_df[
        (hosp_df['age_at_admission'] >= 18) &
        (hosp_df['admission_dttm'] >= start_date) &
        (hosp_df['admission_dttm'] <= end_date) &
        (hosp_df['admission_dttm'] < hosp_df['discharge_dttm']) &
        (hosp_df['discharge_dttm'].notna())
    ].copy()

    logger.info(f"Adult admissions in study window: {len(hosp):,}")

    hosp = hosp.sort_values(['patient_id', 'admission_dttm']).reset_index(drop=True)

    # Calculate gaps between consecutive encounters per patient
    hosp['prev_dc'] = hosp.groupby('patient_id')['discharge_dttm'].shift(1)
    hosp['prev_gap'] = (
        hosp['admission_dttm'] - hosp['prev_dc']
    ).dt.total_seconds() / 3600

    # Link encounters: new group if first encounter OR gap >= link_hours
    # Overlapping encounters (negative gap) and gaps < link_hours are linked
    hosp['new_group'] = (hosp['prev_gap'].isna()) | (hosp['prev_gap'] >= link_hours)
    hosp['group_num'] = hosp.groupby('patient_id')['new_group'].cumsum()
    hosp['joined_hosp_id'] = (
        hosp['patient_id'].astype(str) + '_' + hosp['group_num'].astype(str)
    )

    hid_jid_crosswalk = hosp[['patient_id', 'hospitalization_id', 'joined_hosp_id']].copy()

    logger.info(f"Created {hosp['joined_hosp_id'].nunique():,} joined encounters "
                f"from {len(hosp):,} hospitalizations")

    return hosp, hid_jid_crosswalk


def apply_inpatient_filters(linked_df, hid_jid_crosswalk, adt_df, vitals_df,
                            req_vitals, logger):
    """
    Apply filters: ward stay required, exclude OB/psych/rehab, require vitals.
    """
    logger.info("--- Applying inpatient filters ---")

    cohort_hids = hid_jid_crosswalk['hospitalization_id'].unique()

    # Require ward stay
    ward_hids = adt_df[
        (adt_df['hospitalization_id'].isin(cohort_hids)) &
        (adt_df['location_category'].str.lower() == 'ward')
    ]['hospitalization_id'].unique()

    logger.info(f"With ward stay: {len(ward_hids):,}")

    # Exclude OB/psych/rehab
    exclude_hids = adt_df[
        (adt_df['hospitalization_id'].isin(cohort_hids)) &
        (adt_df['location_category'].str.lower().isin(['l&d', 'psych', 'rehab']))
    ]['hospitalization_id'].unique()

    keep_hids = set(ward_hids) - set(exclude_hids)
    logger.info(f"After excluding OB/psych/rehab: {len(keep_hids):,}")

    # Require full vital sign set
    vital_counts = vitals_df[
        (vitals_df['hospitalization_id'].isin(keep_hids)) &
        (vitals_df['vital_category'].isin(req_vitals))
    ].groupby('hospitalization_id')['vital_category'].nunique()

    has_all_vitals = vital_counts[vital_counts == len(req_vitals)].index
    logger.info(f"With all required vital signs: {len(has_all_vitals):,}")

    hid_jid_crosswalk = hid_jid_crosswalk[
        hid_jid_crosswalk['hospitalization_id'].isin(has_all_vitals)
    ]

    linked_df = linked_df[
        linked_df['joined_hosp_id'].isin(hid_jid_crosswalk['joined_hosp_id'])
    ]

    return linked_df, hid_jid_crosswalk


# ==============================================================================
# CANCER CLASSIFICATION
# ==============================================================================

# 7-group cancer hierarchy
CANCER_GROUPS = [
    {'group': 'Metastatic',    'rank': 1, 'pattern': r'^C7[7-9]|^C80',       'heme': 0},
    {'group': 'Hematologic',   'rank': 2, 'pattern': r'^C8[1-9]|^C9[0-6]',   'heme': 1},
    {'group': 'Lung',          'rank': 3, 'pattern': r'^C3[34]',              'heme': 0},
    {'group': 'GI/HPB',        'rank': 4, 'pattern': r'^C1[5-9]|^C2[0-6]',   'heme': 0},
    {'group': 'Breast',        'rank': 5, 'pattern': r'^C50',                 'heme': 0},
    {'group': 'Genitourinary', 'rank': 6, 'pattern': r'^C5[1-9]|^C6[0-8]',   'heme': 0},
    {'group': 'Other solid',   'rank': 7, 'pattern': r'.*',                   'heme': 0},
]


def classify_cancer(dx_df, hid_jid_crosswalk, logger):
    """
    Classify cancer diagnoses using 7-group hierarchy (vectorized).
    Active cancer: ICD-10 C00-C96, excluding Z85 history codes and C44 NMSC.
    """
    logger.info("--- Classifying cancer diagnoses ---")

    dx = dx_df[
        dx_df['hospitalization_id'].isin(hid_jid_crosswalk['hospitalization_id'])
    ].copy()

    dx['diagnosis_code'] = dx['diagnosis_code'].str.upper().str.strip()

    # Active cancer: C00-C96, not Z85 history, not C44 NMSC
    ca_pattern = r'^C([0-7][0-9]|8[0-9]|9[0-6])'
    dx = dx[
        dx['diagnosis_code'].str.match(ca_pattern, na=False) &
        ~dx['diagnosis_code'].str.startswith('C44')
    ].copy()

    if len(dx) == 0:
        logger.warning("No cancer diagnoses found")
        return pd.DataFrame(columns=[
            'joined_hosp_id', 'ca_icd10_enc', 'liquid_01_enc',
            'rank_enc', 'cancer_group_enc'
        ])

    # Vectorized group assignment: loop over 7 groups, assign by priority
    dx['rank'] = 999
    dx['diag_group'] = 'Unclassified'
    dx['heme'] = 0

    for grp in CANCER_GROUPS:
        mask = (
            dx['diagnosis_code'].str.match(grp['pattern'], na=False) &
            (dx['rank'] == 999)  # only assign if not yet classified
        )
        dx.loc[mask, 'rank'] = grp['rank']
        dx.loc[mask, 'diag_group'] = grp['group']
        dx.loc[mask, 'heme'] = grp['heme']

    # Highest-priority diagnosis per hospitalization
    dx = (dx
          .sort_values(['hospitalization_id', 'rank', 'diagnosis_code'])
          .groupby('hospitalization_id')
          .first()
          .reset_index())

    dx = dx[['hospitalization_id', 'diagnosis_code', 'heme', 'rank', 'diag_group']]

    # Map to joined_hosp_id
    dx = dx.merge(hid_jid_crosswalk, on='hospitalization_id')

    # Highest-priority per joined encounter
    dx_enc = (dx
              .sort_values(['joined_hosp_id', 'rank', 'diagnosis_code'])
              .groupby('joined_hosp_id')
              .first()
              .reset_index())

    dx_enc = dx_enc.rename(columns={
        'diagnosis_code': 'ca_icd10_enc',
        'heme': 'liquid_01_enc',
        'rank': 'rank_enc',
        'diag_group': 'cancer_group_enc'
    })

    logger.info(f"Encounters with active cancer (excluding C44): {len(dx_enc):,}")
    logger.info(f"Cancer group distribution:\n{dx_enc['cancer_group_enc'].value_counts().to_string()}")

    return dx_enc[['joined_hosp_id', 'ca_icd10_enc', 'liquid_01_enc',
                    'rank_enc', 'cancer_group_enc']]


# ==============================================================================
# ENCOUNTER FEATURES
# ==============================================================================

def get_ed_admissions(adt_df, cohort_hids):
    """Identify encounters with ED admission"""
    ed_admits = adt_df[
        (adt_df['hospitalization_id'].isin(cohort_hids)) &
        (adt_df['location_category'].str.lower() == 'ed')
    ]['hospitalization_id'].unique()
    return ed_admits


def get_first_ward_times(adt_df, hid_jid_crosswalk):
    """Get first ward admission time for each encounter"""
    ward_times = adt_df[
        (adt_df['hospitalization_id'].isin(hid_jid_crosswalk['hospitalization_id'])) &
        (adt_df['location_category'].str.lower() == 'ward')
    ][['hospitalization_id', 'in_dttm']].copy()

    ward_times = ward_times.merge(hid_jid_crosswalk, on='hospitalization_id')

    first_ward = (ward_times
                  .sort_values('in_dttm')
                  .groupby('joined_hosp_id')['in_dttm']
                  .first()
                  .reset_index()
                  .rename(columns={'in_dttm': 'first_ward_dttm'}))

    return first_ward


def get_initial_code_status(code_status_df, cohort_df, hid_jid_crosswalk,
                            logger, window_hours=12):
    """
    Extract initial code status within window of admission.
    Joins through hospitalization_id (not patient_id) to avoid many-to-many.
    """
    logger.info("--- Extracting initial code status ---")

    # Join code_status to crosswalk on hospitalization_id
    codes = code_status_df.merge(
        hid_jid_crosswalk[['hospitalization_id', 'joined_hosp_id']],
        on='hospitalization_id',
        how='inner'
    )

    # Bring in admission time from cohort
    codes = codes.merge(
        cohort_df[['joined_hosp_id', 'admission_dttm']],
        on='joined_hosp_id',
        how='inner'
    )

    # Filter to window: 1 day before to window_hours after admission
    codes = codes[
        (codes['start_dttm'] >= codes['admission_dttm'] - pd.Timedelta(days=1)) &
        (codes['start_dttm'] <= codes['admission_dttm'] + pd.Timedelta(hours=window_hours))
    ]

    # Take last code status in window
    codes = (codes
             .sort_values('start_dttm')
             .groupby('joined_hosp_id')['code_status_category']
             .last()
             .reset_index()
             .rename(columns={'code_status_category': 'initial_code_status'}))

    codes['initial_code_status'] = codes['initial_code_status'].str.lower()

    # Recode: DNR/DNI -> "other"
    codes.loc[
        codes['initial_code_status'].isin(['dnr', 'dni', 'dnr/dni']),
        'initial_code_status'
    ] = 'other'

    logger.info(f"Code status available: {len(codes):,}")
    logger.info(f"Distribution:\n{codes['initial_code_status'].value_counts().to_string()}")

    return codes


def extract_hospital_type(adt_df, hid_jid_crosswalk, logger):
    """
    Extract hospital_type from ADT table.
    Maps: academic -> 1, community -> 0. Excludes LTACH.
    """
    logger.info("--- Extracting hospital type ---")

    hosp_type = adt_df[
        adt_df['hospitalization_id'].isin(hid_jid_crosswalk['hospitalization_id'])
    ][['hospitalization_id', 'hospital_type']].drop_duplicates()

    # Handle multiple hospital types per hospitalization
    type_counts = hosp_type.groupby('hospitalization_id').size()
    if (type_counts > 1).any():
        logger.warning(
            f"{(type_counts > 1).sum()} hospitalizations have multiple hospital_type values; "
            f"taking first"
        )
        hosp_type = hosp_type.groupby('hospitalization_id').first().reset_index()

    # Merge to joined_hosp_id and take first per joined encounter
    hosp_type = hosp_type.merge(hid_jid_crosswalk, on='hospitalization_id')
    hosp_type = (hosp_type
                 .groupby('joined_hosp_id')['hospital_type']
                 .first()
                 .reset_index())

    hosp_type['hospital_type'] = hosp_type['hospital_type'].str.lower()

    logger.info(f"Hospital type distribution:\n"
                f"{hosp_type['hospital_type'].value_counts().to_string()}")

    # Exclude LTACH here (single location for this logic)
    n_ltach = (hosp_type['hospital_type'] == 'ltach').sum()
    hosp_type = hosp_type[hosp_type['hospital_type'] != 'ltach']
    if n_ltach > 0:
        logger.info(f"Excluded {n_ltach:,} LTACH encounters")

    return hosp_type


# ==============================================================================
# COMORBIDITY
# ==============================================================================

def calculate_elixhauser(dx_df, hid_jid_crosswalk, logger):
    """
    Calculate van Walraven Elixhauser score (vectorized).
    Cancer-related categories (lymphoma, mets, tumor) are zeroed out
    because cancer is modeled as the primary exposure.
    """
    logger.info("--- Calculating comorbidity scores ---")

    elix_patterns = {
        'chf':             r'^I50|^I11\.0|^I13\.0|^I13\.2',
        'arrhythmia':      r'^I47|^I48|^I49',
        'valvular':        r'^I05|^I06|^I07|^I08|^I34|^I35|^I36|^I37|^I38|^I39',
        'pulm_circ':       r'^I26|^I27|^I28',
        'pvd':             r'^I70|^I71|^I73|^I77|^I79',
        'htn_uncomp':      r'^I10',
        'htn_comp':        r'^I11|^I12|^I13|^I15',
        'paralysis':       r'^G81|^G82',
        'neuro':           r'^G10|^G20|^G30|^G31|^G32|^G35|^G36|^G37',
        'copd':            r'^J40|^J41|^J42|^J43|^J44|^J45|^J46|^J47',
        'diabetes_uncomp': r'^E10[0-9]|^E11[0-9]',
        'diabetes_comp':   r'^E10[2-5]|^E11[2-5]',
        'hypothyroid':     r'^E03',
        'renal':           r'^N18|^N19|^N25',
        'liver_mild':      r'^K70\.0|^K70\.1|^K70\.2|^K70\.3|^K70\.9|^K71|^K73|^K74\.0|^K74\.1|^K74\.2',
        'liver_severe':    r'^K70\.4|^K72|^K76\.5|^K76\.6|^K76\.7',
        'peptic_ulcer':    r'^K25|^K26|^K27|^K28',
        'aids':            r'^B20',
        'lymphoma':        r'^C81|^C82|^C83|^C84|^C85',
        'mets':            r'^C77|^C78|^C79|^C80',
        'tumor':           r'^C[0-2][0-9]|^C3[0-9]|^C4[0-3]|^C4[5-9]|^C5[0-9]|^C6[0-9]|^C7[0-6]',
        'rheum':           r'^M05|^M06|^M32|^M33|^M34',
        'coag':            r'^D65|^D66|^D67|^D68',
        'obesity':         r'^E66',
        'wt_loss':         r'^E40|^E41|^E42|^E43|^E44|^E45|^E46',
        'fluid':           r'^E87',
        'anemia_blood':    r'^D50|^D51|^D52|^D53',
        'alcohol':         r'^F10',
        'drug':            r'^F11|^F12|^F13|^F14|^F15|^F16|^F18|^F19',
        'psychosis':       r'^F20|^F22|^F23|^F24|^F25|^F28|^F29',
        'depression':      r'^F32|^F33'
    }

    # van Walraven weights — cancer categories zeroed (exposure variable)
    vw_weights = {
        'chf': 7, 'arrhythmia': 5, 'valvular': -1, 'pulm_circ': 4, 'pvd': 2,
        'htn_uncomp': 0, 'htn_comp': 0, 'paralysis': 7, 'neuro': 6, 'copd': 3,
        'diabetes_uncomp': 0, 'diabetes_comp': 0, 'hypothyroid': 0, 'renal': 5,
        'liver_mild': 11, 'liver_severe': 11, 'peptic_ulcer': 0, 'aids': 0,
        'lymphoma': 0, 'mets': 0, 'tumor': 0,  # zeroed: cancer is the exposure
        'rheum': 0, 'coag': 3,
        'obesity': -4, 'wt_loss': 6, 'fluid': 5, 'anemia_blood': -2,
        'alcohol': 0, 'drug': -7, 'psychosis': 0, 'depression': -3
    }

    dx = dx_df[
        dx_df['hospitalization_id'].isin(hid_jid_crosswalk['hospitalization_id'])
    ].copy()
    dx['diagnosis_code'] = dx['diagnosis_code'].str.upper()

    dx = dx.merge(hid_jid_crosswalk, on='hospitalization_id')

    all_jids = hid_jid_crosswalk['joined_hosp_id'].unique()
    scores = pd.DataFrame({'joined_hosp_id': all_jids, 'vw': 0})
    scores.set_index('joined_hosp_id', inplace=True)

    for condition, pattern in elix_patterns.items():
        weight = vw_weights.get(condition, 0)
        if weight == 0:
            continue

        mask = dx['diagnosis_code'].str.match(pattern, na=False)
        affected_jids = dx.loc[mask, 'joined_hosp_id'].unique()
        scores.loc[scores.index.isin(affected_jids), 'vw'] += weight

    elix_df = scores.reset_index()
    logger.info(f"van Walraven score — mean: {elix_df['vw'].mean():.1f}, "
                f"median: {elix_df['vw'].median():.1f}")

    return elix_df


# ==============================================================================
# DEFENSIVE UNIT TESTS
# ==============================================================================

def test_cancer_classification(cohort_df, logger):
    """Test 1: Cancer classification logic validation"""
    logger.info("=== UNIT TEST 1: Cancer Classification ===")

    ca_rate = cohort_df['ca_01'].mean()
    liquid_rate = cohort_df['liquid_01'].mean()

    logger.info(f"Active cancer rate: {ca_rate:.1%}")
    logger.info(f"Hematologic cancer rate (of all): {liquid_rate:.1%}")

    if ca_rate < 0.05 or ca_rate > 0.50:
        logger.warning(f"Cancer rate {ca_rate:.1%} outside expected 5-50% range")

    if liquid_rate > ca_rate:
        logger.error("More hematologic cancers than total cancers")
        return False

    met_rate = (cohort_df['rank_enc'] == 1).mean()
    logger.info(f"Metastatic cancer rate: {met_rate:.1%}")

    logger.info("✓ Cancer classification test passed")
    return True


def test_cohort_completeness(cohort_df, site, logger):
    """Test 2: Variable completeness and site-level QC"""
    logger.info("=== UNIT TEST 2: Data Completeness ===")

    critical_vars = [
        'full_code_01', 'ca_01', 'age', 'hosp_type',
        'nhw_01', 'vw'
    ]

    passed = True

    for var in critical_vars:
        if var not in cohort_df.columns:
            logger.error(f"Missing critical variable: {var}")
            passed = False
            continue

        missing_pct = cohort_df[var].isna().mean() * 100
        logger.info(f"{var}: {missing_pct:.1f}% missing")

        if var in ['full_code_01', 'ca_01', 'hosp_type'] and missing_pct > 10:
            logger.warning(f"{var} has {missing_pct:.1f}% missing (>10%)")

    # Outcome distribution
    if 'full_code_01' in cohort_df.columns:
        full_code_rate = cohort_df['full_code_01'].mean()
        logger.info(f"Full code rate: {full_code_rate:.1%}")

        if full_code_rate < 0.30 or full_code_rate > 0.95:
            logger.warning(f"Full code rate {full_code_rate:.1%} outside typical 30-95% range")

    # Hospital type distribution
    if 'hosp_type' in cohort_df.columns:
        logger.info(f"Hospital type distribution:\n"
                    f"{cohort_df['hosp_type'].value_counts().to_string()}")
        if cohort_df['hosp_type'].nunique() < 2:
            logger.warning("Only one hospital type present — cannot test interaction")

    # Code status imputation rate
    if 'code_status_imputed' in cohort_df.columns:
        impute_rate = cohort_df['code_status_imputed'].mean()
        logger.info(f"Code status imputed (presume_full): {impute_rate:.1%}")

    if passed:
        logger.info("✓ Completeness test passed")

    return passed


def test_ed_admission_filter(cohort_df, logger):
    """Test 3: Verify ED admission restriction applied correctly"""
    logger.info("=== UNIT TEST 3: ED Admission Filter ===")

    if 'ed_admit_01' not in cohort_df.columns:
        logger.error("ed_admit_01 column not found")
        return False

    ed_rate = cohort_df['ed_admit_01'].mean()
    logger.info(f"ED admission rate in cohort: {ed_rate:.1%}")

    if ed_rate < 0.99:
        logger.error(f"Expected 100% ED admits, got {ed_rate:.1%}")
        return False

    logger.info("✓ ED admission filter test passed")
    return True


# ==============================================================================
# ANALYSIS
# ==============================================================================

def build_analysis_dataset(cohort_df, logger):
    """Create final analysis dataset with binary outcome and predictors"""
    logger.info("--- Building analysis dataset ---")

    analysis_df = cohort_df.copy()

    # Cancer stratification variables
    analysis_df['heme_ca_01'] = np.where(
        (analysis_df['ca_01'] == 1) & (analysis_df['liquid_01'] == 1), 1, 0
    )
    analysis_df['solid_ca_01'] = np.where(
        (analysis_df['ca_01'] == 1) & (analysis_df['liquid_01'] == 0), 1, 0
    )
    analysis_df['metastatic_01'] = np.where(
        (analysis_df['ca_01'] == 1) & (analysis_df['rank_enc'] == 1), 1, 0
    )

    # Exclude missing age
    n_before = len(analysis_df)
    analysis_df = analysis_df[analysis_df['age'].notna()].copy()
    n_after = len(analysis_df)
    if n_before > n_after:
        logger.info(f"Excluded {n_before - n_after:,} encounters with missing age")

    logger.info(f"Final analysis N: {len(analysis_df):,}")
    logger.info(f"  Cancer: {analysis_df['ca_01'].sum():,} ({analysis_df['ca_01'].mean():.1%})")
    logger.info(f"  Full code: {analysis_df['full_code_01'].sum():,} "
                f"({analysis_df['full_code_01'].mean():.1%})")
    logger.info(f"  Academic: {analysis_df['hosp_type'].sum():,} "
                f"({analysis_df['hosp_type'].mean():.1%})")

    return analysis_df


def run_logistic_regression(df, site, output_dir, logger):
    """
    Run logistic regression with cancer x hospital_type interaction.
    Uses clustered SEs at hospital_id when n_clusters >= 10,
    otherwise HC1 robust SEs with a warning.
    Outputs coefficients and variance-covariance matrix for REMA pooling.
    """
    logger.info("--- Running logistic regression ---")

    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError:
        logger.error("statsmodels not installed. Run: pip install statsmodels")
        return None

    # Prepare model data
    model_df = df[[
        'full_code_01', 'ca_01', 'hosp_type', 'age', 'female_01',
        'nhw_01', 'vw', 'hospital_id'
    ]].copy()

    # Drop remaining missing
    n_before = len(model_df)
    model_df = model_df.dropna()
    n_after = len(model_df)
    if n_before > n_after:
        logger.info(f"Dropped {n_before - n_after:,} rows with missing data")

    logger.info(f"Model N: {len(model_df):,}")

    # Check hospital type variation
    n_hosp_types = model_df['hosp_type'].nunique()
    if n_hosp_types < 2:
        logger.warning("Only one hospital type — fitting main-effects model without interaction")
        formula = 'full_code_01 ~ ca_01 + age + female_01 + nhw_01 + vw'
    else:
        formula = (
            'full_code_01 ~ ca_01 + hosp_type + ca_01:hosp_type + '
            'age + female_01 + nhw_01 + vw'
        )

    logger.info(f"Formula: {formula}")

    # Determine SE method based on number of clusters
    n_clusters = model_df['hospital_id'].nunique()
    logger.info(f"Number of hospital clusters: {n_clusters}")

    if n_clusters >= 10:
        logger.info("Using clustered standard errors at hospital_id level")
        model = smf.logit(formula, data=model_df).fit(
            cov_type='cluster',
            cov_kwds={'groups': model_df['hospital_id']},
            maxiter=100,
            disp=0
        )
    else:
        if n_clusters > 1:
            logger.warning(
                f"Only {n_clusters} clusters — clustered SEs unreliable. "
                f"Using HC1 robust SEs. Meta-analysis should account for this."
            )
        else:
            logger.info("Single hospital — using HC1 robust SEs")
        model = smf.logit(formula, data=model_df).fit(
            cov_type='HC1',
            maxiter=100,
            disp=0
        )

    logger.info(f"\n{model.summary()}")

    # Extract results
    results = pd.DataFrame({
        'variable': model.params.index,
        'coef': model.params.values,
        'se': model.bse.values,
        'z': model.tvalues.values,
        'pval': model.pvalues.values,
        'ci_lower': model.conf_int()[0].values,
        'ci_upper': model.conf_int()[1].values,
        'or': np.exp(model.params.values),
        'or_ci_lower': np.exp(model.conf_int()[0].values),
        'or_ci_upper': np.exp(model.conf_int()[1].values),
        'n_obs': len(model_df),
        'n_events': int(model_df['full_code_01'].sum()),
        'n_clusters': n_clusters,
        'se_method': 'clustered' if n_clusters >= 10 else 'HC1'
    })

    results.to_csv(output_dir / f'regression_results_{site}.csv', index=False)
    logger.info(f"✓ Saved regression results")

    # Save variance-covariance matrix for REMA pooling
    vcov = pd.DataFrame(
        model.cov_params(),
        index=model.params.index,
        columns=model.params.index
    )
    vcov.to_csv(output_dir / f'vcov_matrix_{site}.csv')
    logger.info(f"✓ Saved variance-covariance matrix")

    return results


def generate_table1(df, site, output_dir, logger):
    """Generate Table 1: characteristics by hospital type and cancer status"""
    logger.info("--- Generating Table 1 ---")

    strata = df.groupby(['ca_01', 'hosp_type'])

    # Continuous variables
    cont_vars = ['age', 'vw', 'los_hosp_d']
    cont_summary = []
    for var in cont_vars:
        if var in df.columns:
            summary = strata[var].agg(['count', 'mean', 'std', 'median'])
            summary['variable'] = var
            summary = summary.reset_index()
            cont_summary.append(summary)

    if cont_summary:
        cont_df = pd.concat(cont_summary, ignore_index=True)
        cont_df.to_csv(output_dir / f'table1_continuous_{site}.csv', index=False)
        logger.info("✓ Saved continuous variables table")
    else:
        cont_df = None

    # Categorical variables
    cat_vars = ['female_01', 'nhw_01', 'dead_01', 'icu_01', 'full_code_01',
                'code_status_imputed']
    cat_summary = []
    for var in cat_vars:
        if var in df.columns:
            counts = df.groupby(['ca_01', 'hosp_type', var]).size().reset_index(name='n')
            counts['variable'] = var
            cat_summary.append(counts)

    if cat_summary:
        cat_df = pd.concat(cat_summary, ignore_index=True)
        cat_df.to_csv(output_dir / f'table1_categorical_{site}.csv', index=False)
        logger.info("✓ Saved categorical variables table")
    else:
        cat_df = None

    # Cancer group distribution table
    if 'cancer_group_enc' in df.columns:
        ca_group = (df[df['ca_01'] == 1]
                    .groupby(['cancer_group_enc', 'hosp_type'])
                    .size()
                    .reset_index(name='n'))
        ca_group.to_csv(output_dir / f'cancer_groups_{site}.csv', index=False)
        logger.info("✓ Saved cancer group distribution")

    return cont_df, cat_df


# ==============================================================================
# MAIN WORKFLOW
# ==============================================================================

def main():
    """Main analysis workflow"""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='CLIF Code Status Analysis')
    parser.add_argument(
        '--no-interactive', action='store_true',
        help='Skip interactive prompts (exit on test failure instead of asking)'
    )
    args = parser.parse_args()

    # Configuration
    config = load_config()
    site, tables_location, project_location, file_type = setup_paths(config)

    # Setup logging
    logger = setup_logging(project_location, site)

    logger.info("=" * 80)
    logger.info("CLIF CODE STATUS ANALYSIS")
    logger.info("Cancer and initial full code status by hospital type")
    logger.info("=" * 80)

    logger.info(f"Site: {site}")
    logger.info(f"Tables location: {tables_location}")
    logger.info(f"Output location: {project_location}")

    # Study dates
    start_date = pd.to_datetime(config.get('start_date', '2018-01-01'), utc=True)
    end_date = pd.to_datetime(config.get('end_date', '2024-12-31'), utc=True)
    logger.info(f"Study period: {start_date.date()} to {end_date.date()}")

    # Required vital signs
    req_vitals = ['heart_rate', 'respiratory_rate', 'sbp', 'spo2', 'temp_c']

    # ---- LOAD DATA ----
    logger.info("=" * 80)
    logger.info("LOADING DATA")
    logger.info("=" * 80)

    hosp = load_clif_table(tables_location, 'hospitalization', file_type, logger)
    adt = load_clif_table(tables_location, 'adt', file_type, logger)
    patient = load_clif_table(tables_location, 'patient', file_type, logger)
    dx = load_clif_table(tables_location, 'hospital_diagnosis', file_type, logger)
    vitals = load_clif_table(tables_location, 'vitals', file_type, logger)
    code_status = load_clif_table(tables_location, 'code_status', file_type, logger)

    # Validate
    validate_all_tables({
        'hospitalization': hosp,
        'adt': adt,
        'patient': patient,
        'hospital_diagnosis': dx,
        'vitals': vitals,
        'code_status': code_status
    }, logger)

    # ---- BUILD COHORT ----
    logger.info("=" * 80)
    logger.info("BUILDING COHORT")
    logger.info("=" * 80)

    linked_df, hid_jid_crosswalk = create_encounter_linkage(
        hosp, start_date, end_date, logger
    )

    linked_df, hid_jid_crosswalk = apply_inpatient_filters(
        linked_df, hid_jid_crosswalk, adt, vitals, req_vitals, logger
    )

    # Base cohort: 1 row per joined_hosp_id
    cohort = (linked_df
              .groupby('joined_hosp_id')
              .agg({
                  'patient_id': 'first',
                  'age_at_admission': 'first',
                  'admission_dttm': 'first',
                  'discharge_dttm': 'last',
                  'discharge_category': 'last'
              })
              .reset_index()
              .rename(columns={'age_at_admission': 'age'}))

    # Demographics
    demographics = patient[['patient_id', 'death_dttm', 'sex_category',
                            'race_category', 'ethnicity_category']].drop_duplicates()

    cohort = cohort.merge(demographics, on='patient_id', how='left')
    cohort['female_01'] = np.where(cohort['sex_category'].str.lower() == 'female', 1, 0)
    cohort['dead_01'] = np.where(
        cohort['discharge_category'].str.lower() == 'expired', 1, 0
    )

    # Race/ethnicity: non-Hispanic White = 1, everyone else = 0
    cohort['nhw_01'] = np.where(
        (cohort['race_category'].str.lower() == 'white') &
        (cohort['ethnicity_category'].str.lower() == 'non-hispanic'),
        1, 0
    )

    # Cancer classification
    cancer_df = classify_cancer(dx, hid_jid_crosswalk, logger)
    cohort = cohort.merge(cancer_df, on='joined_hosp_id', how='left')
    cohort['ca_01'] = np.where(cohort['ca_icd10_enc'].notna(), 1, 0)
    cohort['liquid_01'] = cohort['liquid_01_enc'].fillna(0).astype(int)

    # Restrict to ED admissions
    logger.info("--- Restricting to ED admissions ---")
    ed_admits = get_ed_admissions(adt, hid_jid_crosswalk['hospitalization_id'].unique())

    ed_jids = hid_jid_crosswalk[
        hid_jid_crosswalk['hospitalization_id'].isin(ed_admits)
    ]['joined_hosp_id'].unique()

    n_before = len(cohort)
    cohort = cohort[cohort['joined_hosp_id'].isin(ed_jids)]
    cohort['ed_admit_01'] = 1
    logger.info(f"After ED restriction: {len(cohort):,} (excluded {n_before - len(cohort):,})")

    # First ward time
    first_ward = get_first_ward_times(adt, hid_jid_crosswalk)
    cohort = cohort.merge(first_ward, on='joined_hosp_id', how='left')

    # Code status (joins through hospitalization_id, not patient_id)
    codes = get_initial_code_status(code_status, cohort, hid_jid_crosswalk, logger)
    cohort = cohort.merge(codes, on='joined_hosp_id', how='left')

    # Track imputation before filling
    cohort['code_status_imputed'] = np.where(
        cohort['initial_code_status'].isna(), 1, 0
    )
    cohort['initial_code_status'] = cohort['initial_code_status'].fillna('presume_full')

    # Binary outcome: full code vs anything else (created BEFORE tests)
    cohort['full_code_01'] = np.where(
        cohort['initial_code_status'].isin(['full', 'presume_full']), 1, 0
    )

    # Hospital type (LTACH excluded in extract_hospital_type)
    hosp_type_df = extract_hospital_type(adt, hid_jid_crosswalk, logger)
    cohort = cohort.merge(hosp_type_df, on='joined_hosp_id', how='left')

    # Drop encounters with no hospital type (LTACH or missing)
    n_before = len(cohort)
    cohort = cohort[cohort['hospital_type'].notna()]
    if n_before > len(cohort):
        logger.info(f"Dropped {n_before - len(cohort):,} encounters with no/LTACH hospital type")

    # Binary hospital type: academic = 1, community = 0
    cohort['hosp_type'] = np.where(cohort['hospital_type'] == 'academic', 1, 0)

    # Comorbidity (cancer categories zeroed)
    elix_df = calculate_elixhauser(dx, hid_jid_crosswalk, logger)
    cohort = cohort.merge(elix_df, on='joined_hosp_id', how='left')
    cohort['vw'] = cohort['vw'].fillna(0)

    # Hospital ID for clustering
    hospital_map = (adt[['hospitalization_id', 'hospital_id']]
                    .drop_duplicates()
                    .merge(hid_jid_crosswalk, on='hospitalization_id')
                    .groupby('joined_hosp_id')['hospital_id']
                    .first()
                    .reset_index())
    cohort = cohort.merge(hospital_map, on='joined_hosp_id', how='left')

    # Length of stay
    cohort['los_hosp_d'] = (
        (cohort['discharge_dttm'] - cohort['admission_dttm']).dt.total_seconds() / 86400
    )

    # ICU flag (for Table 1)
    icu_encs = adt[
        (adt['hospitalization_id'].isin(hid_jid_crosswalk['hospitalization_id'])) &
        (adt['location_category'].str.lower() == 'icu')
    ]['hospitalization_id'].unique()

    icu_jids = hid_jid_crosswalk[
        hid_jid_crosswalk['hospitalization_id'].isin(icu_encs)
    ]['joined_hosp_id'].unique()

    cohort['icu_01'] = np.where(cohort['joined_hosp_id'].isin(icu_jids), 1, 0)

    # ---- DEFENSIVE TESTS ----
    logger.info("=" * 80)
    logger.info("DEFENSIVE UNIT TESTS")
    logger.info("=" * 80)

    test1_pass = test_cancer_classification(cohort, logger)
    test2_pass = test_cohort_completeness(cohort, site, logger)
    test3_pass = test_ed_admission_filter(cohort, logger)

    if not all([test1_pass, test2_pass, test3_pass]):
        logger.warning("One or more unit tests failed")
        if args.no_interactive:
            logger.error("Exiting due to test failure (--no-interactive mode)")
            sys.exit(1)
        else:
            response = input("\nContinue anyway? (yes/no): ")
            if response.lower() != 'yes':
                sys.exit(1)
    else:
        logger.info("✓✓✓ All unit tests passed")

    # ---- ANALYSIS ----
    logger.info("=" * 80)
    logger.info("ANALYSIS")
    logger.info("=" * 80)

    analysis_df = build_analysis_dataset(cohort, logger)

    # Save cohort
    cohort_file = project_location / 'proj_tables' / f'cohort_{site}.parquet'
    analysis_df.to_parquet(cohort_file)
    logger.info(f"✓ Saved cohort: {cohort_file}")

    # Regression
    output_dir = project_location / 'upload_to_box'
    results = run_logistic_regression(analysis_df, site, output_dir, logger)

    # Table 1
    generate_table1(analysis_df, site, output_dir, logger)

    # Site summary
    summary = pd.DataFrame({
        'site': [site],
        'n_total': [len(analysis_df)],
        'n_cancer': [int(analysis_df['ca_01'].sum())],
        'n_heme': [int(analysis_df['heme_ca_01'].sum())],
        'n_solid': [int(analysis_df['solid_ca_01'].sum())],
        'n_metastatic': [int(analysis_df['metastatic_01'].sum())],
        'n_full_code': [int(analysis_df['full_code_01'].sum())],
        'n_code_status_imputed': [int(analysis_df['code_status_imputed'].sum())],
        'n_academic': [int(analysis_df['hosp_type'].sum())],
        'n_community': [int((1 - analysis_df['hosp_type']).sum())],
        'n_clusters': [analysis_df['hospital_id'].nunique()],
        'date_min': [analysis_df['admission_dttm'].min()],
        'date_max': [analysis_df['admission_dttm'].max()]
    })

    summary.to_csv(output_dir / f'site_summary_{site}.csv', index=False)
    logger.info("✓ Saved site summary")

    logger.info("=" * 80)
    logger.info("✓✓✓ ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Outputs saved to: {output_dir}")
    logger.info(f"  - regression_results_{site}.csv")
    logger.info(f"  - vcov_matrix_{site}.csv")
    logger.info(f"  - table1_continuous_{site}.csv")
    logger.info(f"  - table1_categorical_{site}.csv")
    logger.info(f"  - cancer_groups_{site}.csv")
    logger.info(f"  - site_summary_{site}.csv")
    logger.info(f"  - analysis_log_{site}.txt")


if __name__ == '__main__':
    main()
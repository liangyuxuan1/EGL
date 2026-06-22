# Data Preprocessing

This README documents the current preprocessing pipeline implemented under `data_processing/`.

The main executable pipeline is:

1. `S00_convert_raw_to_feather.py`
2. `S01_split_COPD_NLST.py`
3. `S02_keep_optimal_CT_series_only.py`

`S03_merge_clinical_data.py` and the two check scripts under Step 03 are currently used for validation and analysis, not for the final diagnostic label used in the main modeling pipeline.

## Step 00: Convert raw tables to Feather format

Script:
- `S00_convert_raw_to_feather.py`

Inputs:
- COPDGene clinical table: `data/raw_data/COPDGene_P1P2P3_SM_NS_Long_SEP24.xlsx`, 21794 rows x 1032 columns
- Combined COPDGene + NLST CT table: `data/raw_data/full_label_copd_nlst.csv`, 147385 rows x 136 columns

Outputs:
- `data/raw_data/COPDGene_P1P2P3_SM_NS_Long_SEP24.feather`
- `data/raw_data/full_label_copd_nlst.feather`

What the code actually does:
- Reads the COPDGene table from Excel and the CT table from CSV
- Converts the CT table sentinel values `-1`, `-1.0`, `"-1"`, `"-1.0"` to `pd.NA`
- Does not apply the `-1 -> NA` conversion to the COPDGene clinical table
- Normalizes problematic `object` columns before Feather export for Arrow compatibility
- Validates Feather round-trip consistency after writing
- Skips regeneration if the target Feather file already exists

## Step 01: Split COPD and NLST CT data, and generate summaries

Script:
- `S01_split_COPD_NLST.py`

Inputs:
- Clinical Feather from Step 00: `data/raw_data/COPDGene_P1P2P3_SM_NS_Long_SEP24.feather`
- Combined CT Feather from Step 00: `data/raw_data/full_label_copd_nlst.feather`

Outputs:
- `data/splitted_data/COPD_ct.feather`, `data/splitted_data/COPD_ct.csv`, 103906 rows x 136 columns
- `data/splitted_data/NLST_ct.feather`, `data/splitted_data/NLST_ct.csv`, 43479 rows x 136 columns
- `data/splitted_data/ct_raw_data_summary.json`
- `data/splitted_data/ct_variable_summary.csv`
- `data/splitted_data/clinical_variable_summary.csv`

What the code actually does:
- Splits `full_label_copd_nlst.feather` into COPD and NLST subsets using the `source` column
- Saves both subsets to Feather and CSV
- Writes participant-level summary counts to `ct_raw_data_summary.json`
- Builds `ct_variable_summary.csv` from the full combined CT table using `COPD_CT_METADATA_for_CVD_Risk_Prediction`
- Builds `clinical_variable_summary.csv` from the full COPDGene clinical table by appending `missing_rate` and summary notes to `COPDGene_P1P2P3_visitlevel_DataDict_SEP24.xlsx`

Notes:
- The CT variable summary is generated from the combined CT table, not separately from COPD and NLST

## Step 02: Keep optimal CT series only

Script:
- `S02_keep_optimal_CT_series_only.py`

Inputs:
- `data/splitted_data/COPD_ct.feather`
- `data/splitted_data/NLST_ct.feather`

Outputs:
- `data/splitted_data/COPD_ct_optimal_series_only.feather`, `data/splitted_data/COPD_ct_optimal_series_only.csv`, 19323 rows x 140 columns
- `data/splitted_data/NLST_ct_optimal_series_only.feather`, `data/splitted_data/NLST_ct_optimal_series_only.csv`, 26961 rows x 140 columns
- `data/splitted_data/optimal_series_only_summary.json`

What the code actually does:
- Adds `sid = case`
- Recomputes `Phase_study` from `(source, phase)` for both COPD and NLST
- Uses the following phase mapping:

```python
COPD_PHASE_TO_PHASE_STUDY = {
    "COPDGene": 1,
    "COPDGene-2": 2,
    "COPDGene-3": 3,
    "COPDGene-3B": 3,
}

NLST_PHASE_TO_PHASE_STUDY = {
    "T0": 1,
    "T1": 2,
    "T2": 3,
}
```

- Keeps one CT row per `(sid, Phase_study)`
- Selection rule:
  - first prefer exact `series == optimal_series`
  - otherwise try `series.str.contains(optimal_series)`
  - otherwise fall back deterministically to the lexicographically first series
- Adds audit columns:
  - `series_count`
  - `series_list`
- Writes participant-level summary counts to `optimal_series_only_summary.json`

## Step 03: Merge COPD CT with COPDGene clinical table

Script:
- `S03_merge_clinical_data.py`

Inputs:
- `data/splitted_data/COPD_ct_optimal_series_only.feather`
- `data/raw_data/COPDGene_P1P2P3_SM_NS_Long_SEP24.feather`

Outputs:
- `data/ct_clin_integration/COPD_ct_clinical_merged.feather`, 19201 rows x 1170 columns
- `data/ct_clin_integration/COPD_ct_clinical_merged.csv`
- `data/ct_clin_integration/COPD_merge_report.json`

What the code actually does:
- Processes COPD only
- Aggregates the COPDGene clinical table to unique `(sid, Phase_study)` rows using `build_clinical_table(...)`
- Uses `config_cvd.CVD_EVENT_COLS` to aggregate event variables within duplicated `(sid, Phase_study)` rows
- Merges CT and clinical tables on `(sid, Phase_study)` with CT as the left table
- Drops unmatched CT rows after the left merge to avoid treating missing clinical labels as negatives
- Preserves overlapping CT variables and appends the clinical versions with the suffix `_clin`

Current status:
- This step is still useful for clinical validation, event semantics checks, and possible future risk-prediction work
- It is not the main source of the final diagnostic label currently used downstream
- NLST clinical integration is not implemented in this script

## Validation scripts related to Step 03

### Sanity check of clinician-provided CVD event columns

Script:
- `S03_sanity_check_cvd_events.py`

Input:
- `data/ct_clin_integration/COPD_ct_clinical_merged.feather`

Output:
- `data/ct_clin_integration/COPD_ct_clinical_merged.cvd_event_sanity.json`

Purpose:
- Verifies that the clinician-provided COPDGene event columns exist in the merged COPD table
- Checks the relationship between each base event variable and its `*_slv` counterpart across study phases
- Supports the interpretation that base fields encode cumulative/ever history and `*_slv` fields encode interval events since the last visit

### Check `diag_cvd` definition in the CT table

Script:
- `S02_check_diag_cvd.py`

Input:
- `data/splitted_data/COPD_ct_optimal_series_only.feather`

Output:
- `data/splitted_data/COPD_ct_optimal_series_only.diag_cvd_check.json`

What the code actually checks:
- Confirms that `diag_cvd` exactly matches the OR of:
  - `diag_coronary_artery`
  - `diag_angina`
  - `diag_heart_attack`
  - `diag_stroke`
  - `diag_tia`
  - `diag_periph_vascular`
- The current saved report shows perfect agreement on all evaluable rows:
  - `n_total = 19323`
  - `n_valid = 19173`
  - `agree = 19173`
  - `disagree = 0`

## Current practical conclusion for downstream labeling

For the current diagnostic discovery pipeline, the final diagnosis label is taken directly from the CT tables after Step 02:

- `data/splitted_data/COPD_ct_optimal_series_only.feather`
- `data/splitted_data/NLST_ct_optimal_series_only.feather`

Specifically, the downstream diagnostic label uses the existing `diag_cvd` column in those CT tables rather than rebuilding the label from the merged clinical table in Step 03.

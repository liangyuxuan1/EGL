#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CT + Clinical integration (COPD only) — v0.3

User constraints:
  - CT table has multiple sources: COPD, NLST
  - For now: ONLY process COPD (source == 'COPD')
  - CT.case corresponds to clinical.sid
  - CT.phase values: COPDGene / COPDGene-2 / COPDGene-3 / COPDGene-3B
    map to clinical Phase_study: 1 / 2 / 3 / 3

Outputs (COPD only):
  outputs/ct_clin_integration_copd_v03/
    - ct_schema_groups.json
    - ct_info_table.csv + .feather
    - clinical_phase_table.csv + .feather
    - ct_clin_merged.csv + .feather
    - merge_report.json  (basic merge stats)

NLST:
  - placeholder functions exist, but NOT implemented / NOT called.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parent.parent  # 上一级目录
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_cvd import CVD_EVENT_COLS

# =========================================================
# I/O helpers
# =========================================================
def load_table(path: Path) -> pd.DataFrame:
    """Load a table from .feather / .csv / .xlsx (auto-detect)."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suf = path.suffix.lower()
    if suf == ".feather":
        return pd.read_feather(path)
    if suf in [".csv", ".tsv"]:
        sep = "\t" if suf == ".tsv" else ","
        return pd.read_csv(path, sep=sep, low_memory=False)
    if suf in [".xlsx", ".xls"]:
        return pd.read_excel(path, engine="openpyxl" if suf == ".xlsx" else None)

    raise ValueError(f"Unsupported file type: {path}")


def save_table(df: pd.DataFrame, path: Path) -> None:
    """Save dataframe to feather/csv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suf = path.suffix.lower()
    if suf == ".feather":
        df.reset_index(drop=True).to_feather(path)
    elif suf == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output type: {path}")

# =========================================================
# Step B: Clinical phase table (per sid, Phase_study) — COPD only
# =========================================================
def build_clinical_table_old(clin_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce clinical to (sid, Phase_study) unique table.

    Requirements:
      - clinical has columns: sid, Phase_study
    """
    df = clin_df.copy()

    if "sid" not in df.columns:
        raise ValueError("Clinical table must contain 'sid' column.")
    if "Phase_study" not in df.columns:
        raise ValueError("Clinical table must contain 'Phase_study' column (1/2/3).")

    df["sid"] = df["sid"].astype(str)
    df["Phase_study"] = pd.to_numeric(df["Phase_study"], errors="coerce")

    bad = int(df["Phase_study"].isna().sum())
    if bad > 0:
        raise ValueError(f"Clinical Phase_study has {bad} NaNs; cannot align.")

    df["Phase_study"] = df["Phase_study"].astype(int)

    # one row per sid-phase
    gcols = ["sid", "Phase_study"]
    df = df.sort_values(gcols).drop_duplicates(gcols, keep="first").reset_index(drop=True)

    return df


def build_clinical_table(
    clin_df: pd.DataFrame,
    *,
    sid_col: str = "sid",
    phase_col: str = "Phase_study",
    event_cols: Dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Reduce clinical to a (sid, Phase_study) unique table.

    Rationale
    ---------
    Clinical data may contain multiple rows per (sid, phase). For event-driven labels,
    a naive drop_duplicates(keep="first") can erase positives and introduce label noise.
    Here we:
      - enforce valid (sid, phase)
      - aggregate duplicates so event columns are OR'ed (max) within the same sid-phase.

    Parameters
    ----------
    event_cols:
      Dict[col -> "cat"/"num"]. If provided, we will aggregate these columns robustly:
        - "cat": max() after coercion to numeric (treat as OR for 0/1)
        - "num": median() after coercion
      Other columns (not in event_cols) are kept as the first non-null value.

    Returns
    -------
    DataFrame with one row per (sid, phase).
    """
    df = clin_df.copy()

    if sid_col not in df.columns:
        raise ValueError(f"Clinical table must contain '{sid_col}' column.")
    if phase_col not in df.columns:
        raise ValueError(f"Clinical table must contain '{phase_col}' column (1/2/3).")

    df[sid_col] = df[sid_col].astype(str)
    df[phase_col] = pd.to_numeric(df[phase_col], errors="coerce")

    bad = int(df[phase_col].isna().sum())
    if bad > 0:
        raise ValueError(f"Clinical {phase_col} has {bad} NaNs; cannot align.")

    df[phase_col] = df[phase_col].astype(int)

    # Optional but recommended: validate phase range
    invalid_phase = ~df[phase_col].isin([1, 2, 3])
    if bool(invalid_phase.any()):
        ex = df.loc[invalid_phase, [sid_col, phase_col]].head(10).to_dict(orient="records")
        raise ValueError(f"Unexpected {phase_col} values outside {{1,2,3}}. Examples: {ex}")

    gcols = [sid_col, phase_col]

    # Fast path: already unique
    if df.duplicated(subset=gcols).sum() == 0:
        return df.sort_values(gcols).reset_index(drop=True)

    # ----------------------------
    # Aggregation plan
    # ----------------------------
    event_cols = event_cols or {}
    cat_cols = [c for c, t in event_cols.items() if t == "cat" and c in df.columns]
    num_cols = [c for c, t in event_cols.items() if t == "num" and c in df.columns]

    # For remaining columns, keep first non-null (stable + avoids wiping info)
    other_cols = [c for c in df.columns if c not in gcols and c not in set(cat_cols) and c not in set(num_cols)]

    # Coerce event columns to numeric to make max/median meaningful
    for c in cat_cols + num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    agg: Dict[str, str] = {}
    for c in cat_cols:
        agg[c] = "max"       # OR for 0/1 indicators
    for c in num_cols:
        agg[c] = "median"    # robust summary

    # first non-null for others: implement via custom aggregation
    def first_nonnull(s: pd.Series):
        s2 = s.dropna()
        return s2.iloc[0] if len(s2) else pd.NA

    df_agg = df.groupby(gcols, as_index=False).agg(
        {**agg, **{c: first_nonnull for c in other_cols}}
    )

    # Ensure stable order
    df_agg = df_agg.sort_values(gcols).reset_index(drop=True)
    return df_agg

# =========================================================
# NLST placeholders (NOT implemented)
# =========================================================
def integrate_nlst_placeholder():
    """
    Placeholder:
      - Later you will load NLST clinical table
      - Align NLST phase definition
      - Merge NLST CT rows with NLST clinical rows
    """
    raise NotImplementedError("NLST integration not implemented yet (per user request).")


# =========================================================
# Merge + Report
# =========================================================
def merge_ct_clinical_old(df_ct: pd.DataFrame, df_clin: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Merge on (sid, Phase_study). Return merged df + a small report dict.
    """
    gcols = ["sid", "Phase_study"]

    merged = df_ct.merge(df_clin, on=gcols, how="left", suffixes=("", "_clin"))

    matched = int(merged["sid"].notna().sum())
    miss_clin = int(merged.filter(like="_clin").isna().all(axis=1).sum()) if any(c.endswith("_clin") for c in merged.columns) else int(merged.isna().any(axis=1).sum())

    report = {
        "ct_unique_keys": int(df_ct[gcols].drop_duplicates().shape[0]),
        "clin_unique_keys": int(df_clin[gcols].drop_duplicates().shape[0]),
        "merge_how": "left",
        "matched_rows": matched,
        "missing_clin_rows": miss_clin,
        "note": "Any CT sid-phase without clinical will have NaNs on clinical columns.",
    }
    return merged, report


def merge_ct_clinical_1(df_ct: pd.DataFrame, df_clin: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Merge CT and clinical tables on (sid, Phase_study).

    Key design choices
    ------------------
    - We treat CT as the primary table, but we REQUIRE a matched clinical row because
      CVD event labels come from the clinical table. Unmatched CT rows are DROPPED
      to avoid label noise (treating missing clinical as no-event).
    - We explicitly track overlap (duplicate) column names between CT and clinical.
      Overlapping clinical columns receive the suffix "_clin" after merge.

    Returns
    -------
    (merged_filtered, report)
      - merged_filtered: merged df containing only rows where a clinical match exists
      - report: diagnostics including overlap columns and mismatch counts
    """
    gcols = ["sid", "Phase_study"]
    for c in gcols:
        if c not in df_ct.columns:
            raise ValueError(f"df_ct missing required key column: {c}")
        if c not in df_clin.columns:
            raise ValueError(f"df_clin missing required key column: {c}")

    # -----------------------------
    # 1) Detect overlapping (duplicate) variable names (excluding keys)
    # -----------------------------
    overlap_cols = sorted((set(df_ct.columns) & set(df_clin.columns)) - set(gcols))

    # -----------------------------
    # 2) Check key uniqueness (important: many-to-one merge can explode rows)
    # -----------------------------
    ct_unique_keys = int(df_ct[gcols].drop_duplicates().shape[0])
    clin_unique_keys = int(df_clin[gcols].drop_duplicates().shape[0])

    ct_has_dups = ct_unique_keys != int(df_ct.shape[0])
    clin_has_dups = clin_unique_keys != int(df_clin.shape[0])

    # -----------------------------
    # 3) Merge with indicator to accurately identify unmatched CT rows
    # -----------------------------
    merged = df_ct.merge(
        df_clin,
        on=gcols,
        how="left",
        suffixes=("", "_clin"),
        indicator=True,  # adds column "_merge": "left_only", "both", "right_only"
    )

    # -----------------------------
    # 4) Identify which columns in merged come from clinical
    #    (needed for a robust missing_clin_rows computation)
    # -----------------------------
    clin_nonkey_cols: List[str] = [c for c in df_clin.columns if c not in gcols]

    # After merge:
    # - if a clinical col overlaps with CT col, the clinical version becomes "<col>_clin"
    # - otherwise it remains "<col>"
    clin_merged_cols: List[str] = []
    missing_in_merged: List[str] = []
    for c in clin_nonkey_cols:
        if c in overlap_cols:
            mc = f"{c}_clin"
        else:
            mc = c
        if mc in merged.columns:
            clin_merged_cols.append(mc)
        else:
            # Should not happen unless columns are missing / changed unexpectedly
            missing_in_merged.append(mc)

    # Missing clinical rows:
    # Primary truth: _merge == "left_only"
    miss_by_indicator = (merged["_merge"] == "left_only")

    # Secondary sanity check: all clinical-derived columns are NA
    # (useful to detect unexpected join artifacts)
    if clin_merged_cols:
        miss_by_all_na = merged[clin_merged_cols].isna().all(axis=1)
    else:
        # if clinical table had no non-key columns (unlikely), treat indicator as source of truth
        miss_by_all_na = miss_by_indicator.copy()

    # They should usually agree; we report both.
    missing_clin_rows = int(miss_by_indicator.sum())
    missing_clin_rows_allna = int(miss_by_all_na.sum())

    # -----------------------------
    # 5) DROP unmatched CT rows to avoid label noise
    # -----------------------------
    keep_mask = (merged["_merge"] == "both")
    dropped_rows = int((~keep_mask).sum())

    merged_filtered = merged.loc[keep_mask].copy()

    # You probably don't want to keep the indicator column in downstream modeling
    merged_filtered.drop(columns=["_merge"], inplace=True)

    report: Dict[str, Any] = {
        "merge_keys": gcols,
        "merge_how": "left",
        "ct_rows": int(df_ct.shape[0]),
        "clin_rows": int(df_clin.shape[0]),
        "ct_unique_keys": ct_unique_keys,
        "clin_unique_keys": clin_unique_keys,
        "ct_has_duplicate_keys": bool(ct_has_dups),
        "clin_has_duplicate_keys": bool(clin_has_dups),
        "overlap_columns": overlap_cols,  # (2) show all duplicated variable names
        "n_overlap_columns": int(len(overlap_cols)),
        "clinical_columns_in_merged": clin_merged_cols,
        "n_clinical_columns_in_merged": int(len(clin_merged_cols)),
        "clinical_columns_missing_in_merged": missing_in_merged,
        "missing_clin_rows_by_indicator": missing_clin_rows,
        "missing_clin_rows_by_allna": missing_clin_rows_allna,
        "dropped_unmatched_ct_rows": dropped_rows,  # (3) how many removed
        "kept_rows_after_drop": int(merged_filtered.shape[0]),
        "note": (
            "Unmatched CT rows (no clinical match) are dropped to avoid label noise. "
            "Overlapping clinical columns are suffixed with '_clin'."
        ),
    }

    return merged_filtered, report


from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import pandas as pd


def merge_ct_clinical(
    df_ct: pd.DataFrame,
    df_clin: pd.DataFrame,
    *,
    check_value_consistency: bool = True,
    value_check_max_cols: int = 30,
    value_check_tol_num: float = 1e-6,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Merge CT and clinical tables on (sid, Phase_study).

    Key design choices
    ------------------
    - We treat CT as the primary table, but we REQUIRE a matched clinical row because
      CVD event labels come from the clinical table. Unmatched CT rows are DROPPED
      to avoid label noise (treating missing clinical as no-event).
    - We explicitly track overlap (duplicate) column names between CT and clinical.
      Overlapping clinical columns receive the suffix "_clin" after merge.

    Additional validation (new)
    ---------------------------
    After filtering to matched rows, we validate that for every name in overlap_cols:
      - CT column "<col>" exists in merged_filtered
      - Clinical column "<col>_clin" exists in merged_filtered
    Optionally (disabled by default), we also check value consistency for overlaps.

    Returns
    -------
    (merged_filtered, report)
      - merged_filtered: merged df containing only rows where a clinical match exists
      - report: diagnostics including overlap columns and mismatch counts
    """
    gcols = ["sid", "Phase_study"]
    for c in gcols:
        if c not in df_ct.columns:
            raise ValueError(f"df_ct missing required key column: {c}")
        if c not in df_clin.columns:
            raise ValueError(f"df_clin missing required key column: {c}")

    # 1) Overlapping column names (excluding keys)
    overlap_cols = sorted((set(df_ct.columns) & set(df_clin.columns)) - set(gcols))

    # 2) Check key uniqueness
    ct_unique_keys = int(df_ct[gcols].drop_duplicates().shape[0])
    clin_unique_keys = int(df_clin[gcols].drop_duplicates().shape[0])
    ct_has_dups = ct_unique_keys != int(df_ct.shape[0])
    clin_has_dups = clin_unique_keys != int(df_clin.shape[0])

    # 3) Merge with indicator
    merged = df_ct.merge(
        df_clin,
        on=gcols,
        how="left",
        suffixes=("", "_clin"),
        indicator=True,
    )

    # 4) Identify clinical-derived columns in merged (robust missing_clin_rows)
    clin_nonkey_cols: List[str] = [c for c in df_clin.columns if c not in gcols]
    clin_merged_cols: List[str] = []
    missing_in_merged: List[str] = []
    for c in clin_nonkey_cols:
        mc = f"{c}_clin" if c in overlap_cols else c
        if mc in merged.columns:
            clin_merged_cols.append(mc)
        else:
            missing_in_merged.append(mc)

    miss_by_indicator = (merged["_merge"] == "left_only")
    miss_by_all_na = merged[clin_merged_cols].isna().all(axis=1) if clin_merged_cols else miss_by_indicator.copy()
    missing_clin_rows = int(miss_by_indicator.sum())
    missing_clin_rows_allna = int(miss_by_all_na.sum())

    # 5) DROP unmatched CT rows
    keep_mask = (merged["_merge"] == "both")
    dropped_rows = int((~keep_mask).sum())

    merged_filtered = merged.loc[keep_mask].copy()
    merged_filtered.drop(columns=["_merge"], inplace=True)

    # ---------------------------------------------------------
    # NEW: overlap consistency checks on merged_filtered
    # ---------------------------------------------------------
    # Name-level check: for each overlap col, we expect both <col> and <col>_clin.
    overlap_expected_ct = overlap_cols[:]  # <col>
    overlap_expected_clin = [f"{c}_clin" for c in overlap_cols]  # <col>_clin

    missing_overlap_ct = [c for c in overlap_expected_ct if c not in merged_filtered.columns]
    missing_overlap_clin = [c for c in overlap_expected_clin if c not in merged_filtered.columns]

    overlap_name_check_ok = (len(missing_overlap_ct) == 0) and (len(missing_overlap_clin) == 0)

    # Optional value consistency: compare <col> vs <col>_clin where both exist.
    # WARNING: only enable if you believe these overlaps represent the same semantic variable.
    value_inconsistency: Dict[str, Any] = {}
    if check_value_consistency and overlap_name_check_ok and overlap_cols:
        cols_to_check = overlap_cols[:value_check_max_cols]
        for c in cols_to_check:
            c_ct = c
            c_cl = f"{c}_clin"

            s1 = merged_filtered[c_ct]
            s2 = merged_filtered[c_cl]

            # Compare on rows where both are non-missing
            m = s1.notna() & s2.notna()
            if int(m.sum()) == 0:
                continue

            # If numeric, compare with tolerance; else compare exact equality after string normalization
            if pd.api.types.is_numeric_dtype(s1) and pd.api.types.is_numeric_dtype(s2):
                a = pd.to_numeric(s1[m], errors="coerce")
                b = pd.to_numeric(s2[m], errors="coerce")
                mm = a.notna() & b.notna()
                if int(mm.sum()) == 0:
                    continue
                diff = (a[mm] - b[mm]).abs()
                frac_bad = float((diff > value_check_tol_num).mean())
                if frac_bad > 0.0:
                    value_inconsistency[c] = {
                        "type": "numeric",
                        "n_compared": int(mm.sum()),
                        "frac_mismatch": frac_bad,
                        "tol": value_check_tol_num,
                    }
            else:
                a = s1[m].astype(str).str.strip()
                b = s2[m].astype(str).str.strip()
                frac_bad = float((a != b).mean())
                if frac_bad > 0.0:
                    value_inconsistency[c] = {
                        "type": "non_numeric",
                        "n_compared": int(m.sum()),
                        "frac_mismatch": frac_bad,
                    }

    report: Dict[str, Any] = {
        "merge_keys": gcols,
        "merge_how": "left",
        "ct_rows": int(df_ct.shape[0]),
        "clin_rows": int(df_clin.shape[0]),
        "ct_unique_keys": ct_unique_keys,
        "clin_unique_keys": clin_unique_keys,
        "ct_has_duplicate_keys": bool(ct_has_dups),
        "clin_has_duplicate_keys": bool(clin_has_dups),

        "overlap_columns": overlap_cols,
        "n_overlap_columns": int(len(overlap_cols)),

        "clinical_columns_in_merged": clin_merged_cols,
        "n_clinical_columns_in_merged": int(len(clin_merged_cols)),
        "clinical_columns_missing_in_merged": missing_in_merged,

        "missing_clin_rows_by_indicator": missing_clin_rows,
        "missing_clin_rows_by_allna": missing_clin_rows_allna,

        "dropped_unmatched_ct_rows": dropped_rows,
        "kept_rows_after_drop": int(merged_filtered.shape[0]),

        # NEW: overlap checks
        "overlap_name_check_ok": bool(overlap_name_check_ok),
        "missing_overlap_ct_columns": missing_overlap_ct,
        "missing_overlap_clin_columns": missing_overlap_clin,
        "overlap_value_check_enabled": bool(check_value_consistency),
        "overlap_value_inconsistency": value_inconsistency,

        "note": (
            "Unmatched CT rows (no clinical match) are dropped to avoid label noise. "
            "Overlapping clinical columns are suffixed with '_clin'."
        ),
    }

    return merged_filtered, report


# =========================================================
# Main
# =========================================================
def main():
    # -----------------------------
    # Config (edit paths)
    # -----------------------------
    CT_PATH = Path(__file__).resolve().parent.parent / "data" / "splitted_data" / "COPD_ct_optimal_series_only.feather"      # CT table (COPD optimal CT series only)
    CLINICAL_COPD_PATH = Path(__file__).resolve().parent.parent / "data" / "raw_data" / "COPDGene_P1P2P3_SM_NS_Long_SEP24.feather"  # COPD clinical

    OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "ct_clin_integration"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Load CT table
    # -----------------------------
    ct_copd = load_table(CT_PATH)

    # -----------------------------
    # Load COPD clinical and reduce to sid-phase
    # -----------------------------
    clin_all = load_table(CLINICAL_COPD_PATH)
    clin_phase = build_clinical_table(clin_all, event_cols=CVD_EVENT_COLS)

    # -----------------------------
    # Merge
    # -----------------------------
    merged, report = merge_ct_clinical(ct_copd, clin_phase)
    save_table(merged, OUT_DIR / "COPD_ct_clinical_merged.csv")
    save_table(merged, OUT_DIR / "COPD_ct_clinical_merged.feather")
    (OUT_DIR / "COPD_merge_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[INFO] COPD CT+Clinical merge done.")
    print("[INFO] Outputs:", OUT_DIR)
    print("[INFO] Merge report:", json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

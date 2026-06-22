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
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

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
# Column organization (CT table) — for documentation only
# =========================================================
def build_ct_schema(columns: List[str]) -> Dict[str, List[str]]:
    """
    Create a *documentation schema* grouping CT columns.
    This does NOT change the data; it’s just for you to understand the table.
    """
    cols = list(columns)

    id_cols = [c for c in ["source", "sid", "phase", "Phase_study", "series", "optimal_series"] if c in cols]
    acq_cols = [c for c in cols if c.startswith("size_") or c.startswith("spacing_")]
    roi_cols = [c for c in cols if c.startswith("roi_center_")]
    cac_cols = [c for c in ["gt_agatston_score", "agatston_score"] if c in cols]

    heart_cols = [c for c in cols if any(k in c for k in [
        "la_volume", "lv_volume", "ra_volume", "rv_volume",
        "ascending_aorta", "descending_aorta", "pa_diameter",
        "wall_thickness", "heart_volume", "heart_fat",
    ])]

    derived_cols = [c for c in cols if c.endswith("_indexed") or c.endswith("_ratio") or c.endswith("_index")]

    demo_cols = [c for c in cols if c in [
        "gender", "age_baseline", "age_visit", "race", "ethnic",
        "height", "weight", "waist", "arm_span", "bmi", "bsa",
        "sys_bp", "dias_bp", "heart_rate", "resting_sao2"
    ]]

    diag_cols = [c for c in cols if c.startswith("diag_")]

    misc_cols = [c for c in cols if c in [
        "o2_therapy", "o2_therapy_hours", "o2_therapy_years",
        "smoking_status", "smoking_stop_age", "smoking_start_age",
        "quit_smoking_years", "smoking_duration",
        "alcohol_how_often",
        "cabg", "angioplasty",
        "cvd_cause_death", "all_cause_death", "years_to_death",
    ]]

    used = set(id_cols + acq_cols + roi_cols + cac_cols + heart_cols + derived_cols + demo_cols + diag_cols + misc_cols)
    other_cols = [c for c in cols if c not in used]

    return {
        "id": id_cols,
        "acquisition_geometry": sorted(set(acq_cols)),
        "roi": sorted(set(roi_cols)),
        "cac": cac_cols,
        "ct_heart_measurements": sorted(set(heart_cols)),
        "derived_indices_ratios": sorted(set(derived_cols)),
        "demographics_in_ct_table": demo_cols,
        "diagnosis_flags": sorted(set(diag_cols)),
        "lifestyle_therapy_procedure_outcome": misc_cols,
        "other": other_cols,
    }


# =========================================================
# Phase mapping per your clarification
# =========================================================
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

def ensure_phase_study(ct_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure CT dataframe has:
      - sid (string)        : derived from `case`
      - Phase_study (int)   : use existing column if present, otherwise map from `phase`

    IMPORTANT:
      - We do NOT touch clinical's Phase_study definition.
      - We force CT to match clinical.
    """
    df = ct_df.copy()

    if "case" not in df.columns:
        raise ValueError("CT table must contain column 'case' (maps to clinical.sid).")
    if "source" not in df.columns:
        raise ValueError("CT table must contain column 'source'.")
    if "phase" not in df.columns:
        raise ValueError("CT table must contain column 'phase'.")

    # sid derived from case
    df["sid"] = df["case"].astype(str)

    # map phase (str) to Phase_study (int) according to source
    df["Phase_study"] = pd.NA
    source_to_phase_map = {
        "COPD": COPD_PHASE_TO_PHASE_STUDY,
        "NLST": NLST_PHASE_TO_PHASE_STUDY,
    }
    for source, phase_map in source_to_phase_map.items():
        mask = df["source"].astype(str).eq(source)
        df.loc[mask, "Phase_study"] = df.loc[mask, "phase"].map(phase_map)

    bad = int(df["Phase_study"].isna().sum())
    if bad > 0:
        bad_examples = (
            df.loc[df["Phase_study"].isna(), ["source", "phase"]]
            .astype(str)
            .value_counts()
            .head(10)
            .to_dict()
        )
        raise ValueError(
            f"CT Phase_study has {bad} NaNs. Unmapped (source, phase) values? examples={bad_examples}"
        )

    df["Phase_study"] = df["Phase_study"].astype(int)
    return df


# =========================================================
# Step A: CT info table (per sid, Phase_study) — COPD only
# =========================================================
def pick_optimal_CT_series(ct_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build CT info table at (sid, Phase_study) granularity.

    Strategy:
      - For each (sid, Phase_study), keep ONLY ONE CT row:
          * Prefer row where series == optimal_series
          * Else fallback to first row (deterministic)
      - Add:
          series_count, series_list (audit)
    """
    required = ["sid", "Phase_study", "series", "optimal_series"]
    for c in required:
        if c not in ct_df.columns:
            raise ValueError(f"CT table missing required column: {c}")

    df = ct_df.copy()
    df["sid"] = df["sid"].astype(str)
    df["Phase_study"] = pd.to_numeric(df["Phase_study"], errors="coerce").astype(int)
    df["series"] = df["series"].astype(str)
    df["optimal_series"] = df["optimal_series"].astype(str)

    gcols = ["sid", "Phase_study"]

    # audit columns
    series_agg = (
        df.groupby(gcols)["series"]
          .agg(series_count="count",
               series_list=lambda x: "|".join(sorted(set(map(str, x)))))
          .reset_index()
    )

    # select representative row: optimal if exists
    def _pick_group(g: pd.DataFrame) -> pd.DataFrame:
        opt = g["optimal_series"].iloc[0]
        hit = g[g["series"] == opt]
        if len(hit) >= 1:
            return hit.iloc[[0]]
        # fallback: sometimes a "contains" match
        hit2 = g[g["series"].str.contains(opt, na=False)]
        if len(hit2) >= 1:
            return hit2.iloc[[0]]
        return g.sort_values("series").iloc[[0]]  # deterministic fallback

    gb = df.groupby(gcols, group_keys=False)
    tqdm.pandas(desc="Picking CT series")
    picked = gb.progress_apply(_pick_group).reset_index(drop=True)

    out = picked.merge(series_agg, on=gcols, how="left")

    # tidy column order
    key_cols = ["sid", "Phase_study", "optimal_series", "series_count", "series_list"]
    ref_cols = [c for c in ["source", "case", "phase", "series"] if c in out.columns and c not in key_cols]
    rest_cols = [c for c in out.columns if c not in key_cols + ref_cols]
    out = out[key_cols + ref_cols + rest_cols]

    return out


def keep_optimal_series_only(ct_path: Path, out_dir: Path) -> pd.DataFrame:
    # -----------------------------
    # Load CT table
    # -----------------------------
    df = load_table(ct_path)

    # Ensure sid + Phase_study alignment
    # ct表中增加 sid = case，
    # For COPD: Phase_study {1, 2, 3} = phase {COPDGene, COPDGene-2, COPDGene-3, COPDGene-3B}
    # for NLST: Phase_study {1, 2, 3} = phase {T0, T1, T2}
    df = ensure_phase_study(df)
    print(f"phase counts: {df['phase'].value_counts().to_dict()}")
    print(f"Phase_study counts: {df['Phase_study'].value_counts().to_dict()}")

    # Save schema doc
    # schema = build_ct_schema(df.columns.tolist())
    # (OUT_DIR / "COPD_ct_schema_groups.json").write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")

    # -----------------------------
    # Build CT info table (sid-phase)
    # keep only one optimal series per sid-phase
    # -----------------------------
    df_filtered = pick_optimal_CT_series(df)
    new_path_csv = ct_path.with_name(ct_path.stem + "_optimal_series_only.csv")
    new_path_feather = ct_path.with_name(ct_path.stem + "_optimal_series_only.feather")
    save_table(df_filtered, new_path_csv)
    save_table(df_filtered, new_path_feather)

    return df_filtered


def build_participant_summary(df: pd.DataFrame) -> Dict[str, object]:
    """
    Build participant-level summary for a filtered CT table.
    """
    required = ["sid", "Phase_study"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Filtered CT table missing required column: {c}")

    return {
        "n_rows": int(len(df)),
        "n_sid": int(df["sid"].astype(str).nunique()),
        "sid_by_phase_study": {
            str(int(phase)): int(group["sid"].astype(str).nunique())
            for phase, group in df.groupby("Phase_study")
        },
    }


def save_filtered_summary(summary_by_source: Dict[str, Dict[str, object]], out_dir: Path) -> Path:
    """
    Save summary in JSON for easy manual inspection without extra dependencies.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "optimal_series_only_summary.json"
    summary_path.write_text(
        json.dumps(summary_by_source, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary_path


if __name__ == "__main__":
    # =========================================================
    # Config (edit paths)
    # =========================================================
    copd_path = Path(__file__).resolve().parent.parent / "data" / "splitted_data" / "COPD_ct.feather"      # CT table (COPD only)
    nlst_path = Path(__file__).resolve().parent.parent / "data" / "splitted_data" / "NLST_ct.feather"      # CT table (nlst only)

    out_dir = Path(__file__).resolve().parent.parent / "data" / "splitted_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_nlst = keep_optimal_series_only(nlst_path, out_dir)

    df_copd = keep_optimal_series_only(copd_path, out_dir)

    summary = {
        "NLST": build_participant_summary(df_nlst),
        "COPD": build_participant_summary(df_copd),
    }
    summary_path = save_filtered_summary(summary, out_dir)
    print(f"Saved participant summary to: {summary_path}")

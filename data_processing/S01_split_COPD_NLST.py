#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert the large COPDGene Excel dataset into a fast-loading format (Feather).

Why this is needed:
-------------------
The COPDGene visit-level dataset (Excel) contains ~20k rows × 1000+ columns,
and reading it using pandas.read_excel() is extremely slow because Excel files
must be parsed as XML. Feather (Apache Arrow) loads almost instantly (usually
within a few hundred milliseconds). This conversion step greatly accelerates
all downstream data-processing scripts.

Output:
-------
- COPDGene_P1P2P3_SM_NS_Long_SEP24.feather

Notes:
------
- Feather format preserves column names, dtypes, and supports very fast IO.
- If you prefer Parquet instead, instructions are included below.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict

from COPD_CT_Dict import COPD_CT_METADATA_for_CVD_Risk_Prediction
from utils_data_processing import build_variable_summary, expand_clinical_dict

def build_participant_summary(df: pd.DataFrame) -> Dict[str, object]:
    """
    Build participant-level summary for a filtered CT table.
    """
    required = ["case", "phase"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Filtered CT table missing required column: {c}")

    return {
        "n_rows": int(len(df)),
        "n_case": int(df["case"].astype(str).nunique()),
        "case_by_phase": {
            str(phase): int(group["case"].astype(str).nunique())
            for phase, group in df.groupby("phase")
        },
    }


def save_filtered_summary(summary_by_source: Dict[str, Dict[str, object]], summary_path: Path) -> Path:
    """
    Save summary in JSON for easy manual inspection without extra dependencies.
    """
    summary_path.write_text(
        json.dumps(summary_by_source, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary_path

def split_ct_by_source(
    INPUT_PATH: Path,
    COPD_CSV_OUT: Path,
    COPD_FEATHER_OUT: Path,
    NLST_CSV_OUT: Path,
    NLST_FEATHER_OUT: Path,
    summary_path: Path,
    source_column: str = "source",
):
    """
    Split the combined CT dataset into COPD and NLST subsets based on `source`.
    Supports reading from .feather or .csv input.
    """
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"[ERROR] Combined CT file not found: {INPUT_PATH}")

    print(f"[INFO] Loading CT file for splitting: {INPUT_PATH}")
    suffix = INPUT_PATH.suffix.lower()
    df = pd.read_feather(INPUT_PATH)

    if source_column not in df.columns:
        raise KeyError(f"[ERROR] Column '{source_column}' not found in CT data.")

    src_norm = df[source_column].astype(str).str.strip().str.upper()
    df_copd = df[src_norm == "COPD"]
    df_nlst = df[src_norm == "NLST"]

    print(f"[INFO] COPD subset shape: {df_copd.shape}")
    print(f"[INFO] NLST subset shape: {df_nlst.shape}")

    COPD_CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    NLST_CSV_OUT.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Writing COPD subset → {COPD_CSV_OUT} / {COPD_FEATHER_OUT}")
    df_copd.to_csv(COPD_CSV_OUT, index=False)
    df_copd.to_feather(COPD_FEATHER_OUT)

    print(f"[INFO] Writing NLST subset → {NLST_CSV_OUT} / {NLST_FEATHER_OUT}")
    df_nlst.to_csv(NLST_CSV_OUT, index=False)
    df_nlst.to_feather(NLST_FEATHER_OUT)

    print("[INFO] Split complete.")

    summary = {
        "NLST": build_participant_summary(df_nlst),
        "COPD": build_participant_summary(df_copd),
    }
    summary_path = save_filtered_summary(summary, summary_path)
    print(f"Saved participant summary to: {summary_path}")



if __name__ == "__main__":
    # ---------------------------------------------------------
    # File paths (modify as needed)
    # ---------------------------------------------------------

    # Inputs
    raw_data_dir = Path(__file__).resolve().parent.parent / "data" / "raw_data"
    COPD_FEATHER_PATH   = Path(raw_data_dir / "COPDGene_P1P2P3_SM_NS_Long_SEP24.feather")
    CT_FEATHER_PATH     = Path(raw_data_dir / "full_label_copd_nlst.feather")

    # Outputs
    output_dir = Path(__file__).resolve().parent.parent / "data" / "splitted_data"
    output_dir.mkdir(exist_ok=True, parents=True)
    CT_COPD_CSV_PATH    = Path(output_dir / "COPD_ct.csv")
    CT_COPD_FEATHER_PATH= Path(output_dir / "COPD_ct.feather")
    CT_NLST_CSV_PATH    = Path(output_dir / "NLST_ct.csv")
    CT_NLST_FEATHER_PATH= Path(output_dir / "NLST_ct.feather")
    summary_path = Path(output_dir / "ct_raw_data_summary.json")

    # Load data
    df_copd_clinical = pd.read_feather(COPD_FEATHER_PATH)
    df_copd_CT = pd.read_feather(CT_FEATHER_PATH)

    # Split CT dataset into COPD and NLST subsets based on `source`
    split_ct_by_source(
        CT_FEATHER_PATH,
        CT_COPD_CSV_PATH,
        CT_COPD_FEATHER_PATH,
        CT_NLST_CSV_PATH,
        CT_NLST_FEATHER_PATH,
        summary_path
    )

    # Build variable summary for CT dataset
    CT_summary = build_variable_summary(df_copd_CT, used_dict=COPD_CT_METADATA_for_CVD_Risk_Prediction)
    CT_summary_csv = output_dir / "ct_variable_summary.csv"
    CT_summary.to_csv(CT_summary_csv, index=False)

    # Append missing_rate and notes to COPDGene dictionary
    clinical_summary = build_variable_summary(df_copd_clinical)
    clinical_dict_expanded = expand_clinical_dict(clinical_summary)
    clinical_dict_expanded_csv = output_dir / "clinical_variable_summary.csv"
    clinical_dict_expanded.to_csv(clinical_dict_expanded_csv, index=False)

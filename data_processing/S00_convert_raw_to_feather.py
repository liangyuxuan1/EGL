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
from pandas.testing import assert_frame_equal

from utils_data_processing import build_variable_summary, expand_clinical_dict


def _normalize_object_columns_for_feather(
    df: pd.DataFrame,
    force_string_columns: tuple[str, ...] = ("case",),
) -> pd.DataFrame:
    """
    PyArrow/Feather cannot reliably write pandas 'object' columns that contain
    mixed Python types (e.g., str + int, bytes + int). This normalizes those
    columns to a consistent dtype.

    Strategy:
    - Force selected ID-like columns (default: 'case') to pandas StringDtype.
    - If bytes/bytearray appear, decode them to text first.
    - If an object column is fully numeric, cast to nullable Int/Float.
    - Otherwise, cast to pandas StringDtype for deterministic Arrow conversion.
    """
    changed_cols: list[str] = []

    # Force selected columns to string for stability and to preserve any leading zeros.
    for col in force_string_columns:
        if col not in df.columns:
            continue
        s = df[col]
        if s.dtype == "string":
            continue
        if s.dtype == "object":
            df[col] = s.map(
                lambda x: (
                    x.decode("utf-8", errors="replace")
                    if isinstance(x, (bytes, bytearray))
                    else x
                )
            ).astype("string")
        else:
            df[col] = s.astype("string")
        changed_cols.append(col)

    object_cols = [
        c for c in df.columns
        if df[c].dtype == "object" and c not in force_string_columns
    ]
    if not object_cols:
        if changed_cols:
            preview = ", ".join(changed_cols[:12]) + (" ..." if len(changed_cols) > 12 else "")
            print(f"[WARN] Normalized columns for Feather compatibility: {preview}")
        return df

    for col in object_cols:
        s = df[col]
        non_null = s.dropna()
        if non_null.empty:
            continue

        # If we see any bytes-like values, decode them first so we can safely
        # cast the entire column to string without "Expected bytes" issues.
        has_bytes = non_null.map(lambda x: isinstance(x, (bytes, bytearray))).any()
        if has_bytes:
            s = s.map(
                lambda x: (
                    x.decode("utf-8", errors="replace")
                    if isinstance(x, (bytes, bytearray))
                    else x
                )
            )

        # Many "uniform" CSV columns still show up as dtype=object due to a few
        # non-numeric tokens (e.g., "", "NA"); prefer numeric casting if possible.
        if not has_bytes:
            numeric = pd.to_numeric(s, errors="coerce")
            if numeric.notna().sum() == s.notna().sum():
                numeric_non_null = numeric.dropna()
                is_integer = ((numeric_non_null % 1) == 0).all()
                if is_integer:
                    df[col] = pd.Series(pd.array(numeric, dtype="Int64"), index=s.index)
                else:
                    df[col] = numeric.astype("Float64")
                changed_cols.append(col)
                continue

        # Otherwise normalize to string dtype to make Arrow conversion deterministic.
        non_null2 = s.dropna()
        types = set(non_null2.map(type).unique().tolist())
        if types.issubset({str}):
            if has_bytes:
                df[col] = s.astype("string")
                changed_cols.append(col)
            continue

        df[col] = s.astype("string")
        changed_cols.append(col)

    if changed_cols:
        preview = ", ".join(changed_cols[:12]) + (" ..." if len(changed_cols) > 12 else "")
        print(f"[WARN] Normalized columns for Feather compatibility: {preview}")

    # Final safety check: remaining object columns should not contain mixed types
    # that Arrow cannot handle; surface them for debugging.
    remaining_object_cols = [c for c in df.columns if df[c].dtype == "object"]
    if remaining_object_cols:
        suspicious: list[str] = []
        for col in remaining_object_cols:
            non_null = df[col].dropna()
            if non_null.empty:
                continue
            types = set(non_null.map(type).unique().tolist())
            if not types.issubset({str, bytes, bytearray}):
                suspicious.append(f"{col}({', '.join(t.__name__ for t in sorted(types, key=lambda t: t.__name__))})")
        if suspicious:
            print("[WARN] Remaining object columns may still be incompatible with Feather:")
            print("       " + "; ".join(suspicious[:20]) + (" ..." if len(suspicious) > 20 else ""))

    return df

def _validate_feather_roundtrip(
    df_expected: pd.DataFrame,
    feather_path: Path,
    *,
    check_dtype: bool = False,
    use_fingerprint: bool = True,
) -> None:
    """
    Validate that a Feather file, when read back, matches the DataFrame we wrote.

    This checks values rather than strict dtypes by default, because we may
    normalize columns (e.g., 'case' -> string) for Arrow compatibility.
    """
    if not feather_path.exists():
        raise FileNotFoundError(f"Feather file not found for validation: {feather_path}")

    df_actual = pd.read_feather(feather_path)

    expected_cols = list(df_expected.columns)
    actual_cols = list(df_actual.columns)
    missing = [c for c in expected_cols if c not in df_actual.columns]
    extra = [c for c in actual_cols if c not in df_expected.columns]
    if missing or extra:
        raise AssertionError(
            f"Column mismatch after Feather roundtrip. missing={missing[:20]} extra={extra[:20]}"
        )

    # Align order and ignore index representation differences.
    df_expected_aligned = df_expected[expected_cols].reset_index(drop=True)
    df_actual_aligned = df_actual[expected_cols].reset_index(drop=True)

    if use_fingerprint:
        # Fast-ish probabilistic check before full comparison.
        exp_hash = pd.util.hash_pandas_object(df_expected_aligned, index=False).to_numpy(dtype="uint64")
        act_hash = pd.util.hash_pandas_object(df_actual_aligned, index=False).to_numpy(dtype="uint64")
        exp_sig = (int(exp_hash.sum(dtype=np.uint64)), int(np.bitwise_xor.reduce(exp_hash)))
        act_sig = (int(act_hash.sum(dtype=np.uint64)), int(np.bitwise_xor.reduce(act_hash)))
        if exp_sig == act_sig:
            print("[INFO] Feather validation passed (fingerprint match).")
            return

    # Deterministic check (may be slower for large datasets).
    assert_frame_equal(
        df_expected_aligned,
        df_actual_aligned,
        check_dtype=check_dtype,
        check_exact=True,
    )
    print("[INFO] Feather validation passed (full equality).")


def _replace_neg1_with_na(df: pd.DataFrame) -> pd.DataFrame:
    """
    In the CT CSV, missing values are encoded as -1/-1.0 (and sometimes string forms).
    Normalize them to pandas NA so downstream IO writes real blanks.
    """
    replacements = {-1: pd.NA, -1.0: pd.NA, "-1": pd.NA, "-1.0": pd.NA}
    return df.replace(replacements)


def convert_excel_to_feather(INPUT_PATH: Path, FEATHER_PATH: Path, convert_neg1_with_NA: bool = True)-> pd.DataFrame:
    """
    Convert CSV or Excel file to Feather format.
    Automatically detects file type by suffix.

    Supported input formats:
      - .xlsx / .xls
      - .csv
    """

    # ---------------------------------------------------------
    # 0. Skip if Feather already exists
    # ---------------------------------------------------------
    if FEATHER_PATH.exists():
        print(f"[INFO] Feather file already exists: {FEATHER_PATH}")
        print("[INFO] Delete it manually if you want to regenerate it.")
        return

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    suffix = INPUT_PATH.suffix.lower()

    # ---------------------------------------------------------
    # 1. Load input file (slow step, done only once)
    # ---------------------------------------------------------
    print(f"[INFO] Loading input file: {INPUT_PATH}")
    if suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(INPUT_PATH, engine="openpyxl")
    elif suffix == ".csv":
        df = pd.read_csv(INPUT_PATH)
    else:
        raise ValueError(
            f"Unsupported input format: {suffix}. "
            "Only .csv, .xlsx, .xls are supported."
        )

    print(f"[INFO] Loaded data. Shape = {df.shape}")

    # Memory usage
    mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"[INFO] DataFrame memory usage: {mem_mb:.2f} MB")

    # ---------------------------------------------------------
    # 2. Save as Feather
    # ---------------------------------------------------------
    # For CT CSV, convert sentinel -1/-1.0 to NA before writing.
    if convert_neg1_with_NA:
        df = _replace_neg1_with_na(df)

    # Normalize problematic 'object' columns (e.g., mixed int/bytes) before writing.
    df = _normalize_object_columns_for_feather(df)

    print(f"[INFO] Writing Feather file: {FEATHER_PATH}")
    FEATHER_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_feather(FEATHER_PATH)
    print("[INFO] Feather conversion complete.")

    # ---------------------------------------------------------
    # 3. Validate roundtrip (read Feather back and compare)
    # ---------------------------------------------------------
    _validate_feather_roundtrip(df, FEATHER_PATH)

    return df


if __name__ == "__main__":
    # ---------------------------------------------------------
    # File paths (modify as needed)
    # ---------------------------------------------------------

    # Inputs
    raw_data_dir = Path(__file__).resolve().parent.parent / "data" / "raw_data"
    COPD_EXCEL_PATH     = Path(raw_data_dir / "COPDGene_P1P2P3_SM_NS_Long_SEP24.xlsx")
    CT_CSV_PATH         = Path(raw_data_dir / "full_label_copd_nlst.csv")

    # Outputs
    COPD_FEATHER_PATH   = Path(raw_data_dir / "COPDGene_P1P2P3_SM_NS_Long_SEP24.feather")
    CT_FEATHER_PATH     = Path(raw_data_dir / "full_label_copd_nlst.feather")

    # 原始表格读取很慢，如果feather文件存在，就不再转换
    # 如需重新转换，需要手动删除feather文件
    df_copd_clinical = convert_excel_to_feather(COPD_EXCEL_PATH, COPD_FEATHER_PATH, convert_neg1_with_NA=False)
    df_copd_CT = convert_excel_to_feather(CT_CSV_PATH, CT_FEATHER_PATH, convert_neg1_with_NA=True)

import pandas as pd
from typing import Tuple, List, Dict
from pathlib import Path

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def summarize(series: pd.Series) -> Tuple[str, float]:
    total = len(series)

    # Missing rules:
    # - NaN/None
    # - numeric sentinels: -1 / -1.0
    # - object/text empties: "", "NA", "NaN", "-1", "-1.0"
    is_missing = series.isna()
    if pd.api.types.is_numeric_dtype(series):
        is_missing = is_missing | (series == -1) | (series == -1.0)
    else:
        is_missing = is_missing | series.map(
            lambda x: isinstance(x, str) and x.strip() in {"", "-1", "-1.0", "NA", "NaN"}
        )

    missing_count = int(is_missing.sum())
    missing_rate = missing_count / total
    missing_pct = (missing_count / total * 100) if total else 0.0

    valid = series[~is_missing].dropna()
    if valid.empty:
        return f"missing={missing_count} ({missing_pct:.1f}%)", missing_rate

    dtype = series.dtype
    if pd.api.types.is_numeric_dtype(dtype):
        uniq_valid = pd.unique(valid)
        uniq_set = set(uniq_valid.tolist())
        # Common case for diag codes: {-1,0,1} with -1 as missing.
        if uniq_set.issubset({0, 1}) and len(uniq_set) <= 2:
            cnt0 = int((valid == 0).sum())
            cnt1 = int((valid == 1).sum())
            pct0 = cnt0 / total * 100 if total else 0.0
            pct1 = cnt1 / total * 100 if total else 0.0
            summary_note = (
                f"missing={missing_count} ({missing_pct:.1f}%), "
                f"0={cnt0} ({pct0:.1f}%), 1={cnt1} ({pct1:.1f}%)"
            )
            return summary_note, missing_rate

        if len(uniq_valid) <= 10:
            summary_note = (
                f"missing={missing_count} ({missing_pct:.1f}%), "
                "values={" + ", ".join(str(x) for x in sorted(uniq_valid)) + "}"
            )
            return summary_note, missing_rate

        p01 = valid.quantile(0.01)
        p99 = valid.quantile(0.99)
        summary_note = (
            f"missing={missing_count} ({missing_pct:.1f}%), "
            f"min={valid.min():.3g}, p01={p01:.3g}, mean={valid.mean():.3g}, p99={p99:.3g}, max={valid.max():.3g}"
        )
        return summary_note, missing_rate

    # treat everything else as categorical/text
    uniq = valid.unique()
    if len(uniq) <= 10:
        summary_note = (
            f"missing={missing_count} ({missing_pct:.1f}%), "
            "values={" + ", ".join(str(x) for x in uniq) + "}"
        )
        return summary_note, missing_rate

    top = valid.value_counts().head(5)

    summary_note = f"missing={missing_count} ({missing_pct:.1f}%), top5=" + "; ".join(f"{idx}:{cnt}" for idx, cnt in top.items())
    return  summary_note, missing_rate


# ---------------------------------------------------------
# Build variable summary
# ---------------------------------------------------------
def build_variable_summary(df: pd.DataFrame, used_dict: Dict[str, Dict[str, object]]=None) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    if used_dict is not None:
        missing_meta = [c for c in df.columns if c not in used_dict]
        if missing_meta:
            print(f"[WARN] Missing {len(missing_meta)} entries for columns in given dictionary: {missing_meta}")

        extra_meta = [c for c in used_dict if c not in df.columns]
        if extra_meta:
            print(f"[WARN] {len(extra_meta)} entries in given dictionary are not in data (ignored): {extra_meta}")

        for col, meta in used_dict.items():
            if col not in df.columns:
                continue  # already warned; skip

            summary_note, missing_rate = summarize(df[col])
            base_note = meta.get("notes", "")
            combined_note = summary_note if not base_note else f"{base_note} | {summary_note}"
            rows.append({
                "var_name": col,
                "clinical_domain": meta["clinical_domain"],
                "vtype": meta["vtype"],
                "dtype": meta["dtype"],
                "used_for_prediction": meta["used_for_prediction"],
                "missing_rate": missing_rate,
                "need_filtering": meta["need_filtering"],
                "meaning": meta["meaning"],
                "notes": combined_note,
            })
    else:
        for col in df.columns:
            summary_note, missing_rate = summarize(df[col])
            rows.append({
                "var_name": col,
                "clinical_domain": pd.NA,
                "vtype": pd.NA,
                "dtype": str(df[col].dtype),
                "used_for_prediction": pd.NA,
                "missing_rate": missing_rate,
                "need_filtering": pd.NA,
                "meaning": pd.NA,
                "notes": summary_note,
            })

    return pd.DataFrame(rows, columns=[
        "var_name", "clinical_domain", "vtype", "dtype", "used_for_prediction", "missing_rate", "need_filtering", "meaning", "notes"
    ])


def expand_clinical_dict(clinical_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Load the COPD clinical dictionary (Excel), then append missing_rate and notes
    from clinical_summary by matching variable names.
    """
    dict_path = Path(__file__).resolve().parent.parent / "data" / "raw_data" / "COPDGene_P1P2P3_visitlevel_DataDict_SEP24.xlsx"
    dict_df = pd.read_excel(dict_path)

    if "VariableName" not in dict_df.columns:
        raise ValueError(
            f"Clinical dict missing required column 'VariableName'. "
            f"Columns={list(dict_df.columns)[:20]}"
        )

    # Prepare summary columns
    required = {"var_name", "missing_rate", "notes"}
    missing_cols = required - set(clinical_summary.columns)
    if missing_cols:
        raise ValueError(f"clinical_summary missing required columns: {sorted(missing_cols)}")

    summary = clinical_summary[["var_name", "missing_rate", "notes"]].copy()

    # Normalize matching keys (strip whitespace)
    dict_df["_var_key"] = dict_df["VariableName"].astype(str).str.strip()
    summary["_var_key"] = summary["var_name"].astype(str).str.strip()

    # Avoid clobbering existing dict notes by naming the summary column explicitly
    summary = summary.rename(columns={"notes": "summary_notes"})

    merged = dict_df.merge(summary, on="_var_key", how="left")
    merged = merged.drop(columns=["_var_key", "var_name"])

    # If no "notes" column exists in the dict, promote summary_notes -> notes
    if "notes" not in merged.columns and "summary_notes" in merged.columns:
        merged = merged.rename(columns={"summary_notes": "notes"})

    lead_cols = ["varnum", "VariableName", "missing_rate", "notes"]
    existing_lead = [c for c in lead_cols if c in merged.columns]
    remaining = [c for c in merged.columns if c not in existing_lead]
    merged = merged[existing_lead + remaining]

    return merged

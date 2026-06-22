# check_diag_cvd_or.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd


SID_COL = "sid"
PHASE_COL = "Phase_study"

TARGET = "diag_cvd"
COMPONENTS = [
    "diag_coronary_artery",
    "diag_angina",
    "diag_heart_attack",
    # "diag_afib",
    # "diag_heart_fail",
    "diag_stroke",
    "diag_tia",
    "diag_periph_vascular",
]

# how to treat missing values in components
# "strict"     : only evaluate rows where ALL components and diag_cvd are non-missing
# "na_as_zero" : treat component NA as 0; require diag_cvd non-missing
# MISSING_POLICY = "strict"
MISSING_POLICY = "na_as_zero"


MAX_EXAMPLES = 20


def load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_feather(path)


def to01(s: pd.Series) -> pd.Series:
    """
    Coerce to Int64 {0,1,NA}. Be strict: keep only 0/1.
    """
    sn = pd.to_numeric(s, errors="coerce")
    out = pd.Series(pd.NA, index=s.index, dtype="Int64")
    out.loc[sn == 0] = 0
    out.loc[sn == 1] = 1
    return out


def compute_or(
    df: pd.DataFrame,
    comps: List[str],
    *,
    missing_policy: str,
) -> Tuple[pd.Series, pd.Series]:
    """
    Returns:
      or_value: Int64 {0,1,NA} OR across comps
      valid_mask: bool mask of rows where OR is considered valid under policy
    """
    comp01 = {c: to01(df[c]) for c in comps}

    if missing_policy == "na_as_zero":
        # NA -> 0 for components only
        mat = pd.concat([comp01[c].fillna(0).astype(int) for c in comps], axis=1)
        orv = (mat.max(axis=1) > 0).astype("Int64")
        valid = pd.Series(True, index=df.index)
        return orv, valid

    if missing_policy == "strict":
        mat = pd.concat([comp01[c] for c in comps], axis=1)
        valid = mat.notna().all(axis=1)
        # compute OR only on valid rows
        orv = pd.Series(pd.NA, index=df.index, dtype="Int64")
        if valid.any():
            matv = mat.loc[valid].astype(int)
            orv.loc[valid] = (matv.max(axis=1) > 0).astype("Int64")
        return orv, valid

    raise ValueError(f"Unknown missing_policy: {missing_policy}")


def main():
    DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "splitted_data" / "COPD_ct_optimal_series_only.feather"      # CT table (COPD optimal CT series only)

    df = load_df(DATA_PATH)

    # -------- column existence --------
    missing_cols = [c for c in [TARGET] + COMPONENTS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in table: {missing_cols}")

    y = to01(df[TARGET])

    orv, valid_comp = compute_or(df, COMPONENTS, missing_policy=MISSING_POLICY)

    # validity requires diag_cvd present too
    valid = valid_comp & y.notna()

    # comparisons
    agree = (y == orv) & valid
    disagree = (y != orv) & valid

    n_total = int(len(df))
    n_valid = int(valid.sum())
    n_agree = int(agree.sum())
    n_disagree = int(disagree.sum())

    agree_rate = float(n_agree / n_valid) if n_valid > 0 else np.nan

    # -------- mismatch types --------
    # Type A: diag_cvd=1 but OR=0
    type_a = (y == 1) & (orv == 0) & valid
    # Type B: diag_cvd=0 but OR=1
    type_b = (y == 0) & (orv == 1) & valid

    # also track NA situations (useful to see if strict policy discards too much)
    n_target_na = int(y.isna().sum())
    n_any_comp_na = int(pd.concat([to01(df[c]).isna() for c in COMPONENTS], axis=1).any(axis=1).sum())

    # -------- optional: which component(s) cause OR=1 when diag_cvd=0 --------
    comp01 = {c: to01(df[c]) for c in COMPONENTS}
    comp_mat = pd.concat([comp01[c] for c in COMPONENTS], axis=1)
    comp_mat.columns = COMPONENTS

    # For type_b rows, count which components are 1
    comp_hits_b = {}
    if type_b.any():
        sub = comp_mat.loc[type_b].fillna(0).astype(int)
        comp_hits_b = sub.sum(axis=0).sort_values(ascending=False).to_dict()

    # For type_a rows, show whether all components are 0 (should be by definition) and any NA
    comp_na_a = {}
    if type_a.any():
        sub = comp_mat.loc[type_a]
        comp_na_a = {
            "rows_with_any_component_na": int(sub.isna().any(axis=1).sum()),
            "rows_all_components_zero_among_nonmissing": int((sub.fillna(0).astype(int).sum(axis=1) == 0).sum()),
        }

    # -------- print summary --------
    print("=" * 80)
    print("[CHECK] diag_cvd equals OR of component diag_* variables")
    print("=" * 80)
    print(f"Table: {DATA_PATH}")
    print(f"Missing policy: {MISSING_POLICY}")
    print(f"Rows total: {n_total}")
    print(f"Rows with diag_cvd NA: {n_target_na}")
    print(f"Rows with ANY component NA (before policy): {n_any_comp_na}")
    print(f"Rows evaluated (valid under policy): {n_valid}")
    print(f"Agree: {n_agree}  Disagree: {n_disagree}  AgreeRate: {agree_rate:.6f}")
    print()
    print(f"Mismatch A (diag_cvd=1, OR=0): {int(type_a.sum())}")
    print(f"Mismatch B (diag_cvd=0, OR=1): {int(type_b.sum())}")

    if comp_hits_b:
        print("\n[DETAIL] For mismatch B, component ones counts (which diag_* drive OR=1):")
        for k, v in comp_hits_b.items():
            print(f"  {k}: {int(v)}")

    if comp_na_a:
        print("\n[DETAIL] For mismatch A, NA / zero diagnostics:")
        for k, v in comp_na_a.items():
            print(f"  {k}: {v}")

    # -------- show examples --------
    show_cols = [c for c in [SID_COL, PHASE_COL] if c in df.columns] + [TARGET] + COMPONENTS
    if n_disagree > 0:
        print("\n" + "=" * 80)
        print("[EXAMPLES] Disagreement rows (up to max examples)")
        print("=" * 80)
        ex = df.loc[disagree, show_cols].copy()
        ex[TARGET] = y.loc[disagree]
        ex["_or_components"] = orv.loc[disagree]
        ex = ex.head(MAX_EXAMPLES)
        print(ex.to_string(index=False))

    # -------- return a small machine-readable report --------
    report: Dict[str, Any] = {
        "path": str(DATA_PATH),
        "missing_policy": MISSING_POLICY,
        "n_total": n_total,
        "n_valid": n_valid,
        "agree": n_agree,
        "disagree": n_disagree,
        "agree_rate": agree_rate,
        "mismatch_A_diag1_or0": int(type_a.sum()),
        "mismatch_B_diag0_or1": int(type_b.sum()),
        "component_hits_mismatch_B": comp_hits_b,
    }

    out_json = DATA_PATH.with_suffix(".diag_cvd_check.json")
    with open(out_json, "w") as f:
        import json
        json.dump(report, f, indent=2)
    print(f"\n[WRITE] JSON report -> {out_json}")


if __name__ == "__main__":
    main()
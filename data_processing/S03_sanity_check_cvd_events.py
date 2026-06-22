# sanity_check_cvd_events.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# User config: paths
# ------------------------------------------------------------
PHASE_COL = "Phase_study"
SID_COL = "sid"

# ------------------------------------------------------------
# Your event dict (import if you prefer)
# ------------------------------------------------------------
CVD_EVENT_COLS: Dict[str, str] = {
    "CoronaryArtery": "cat",
    "coronaryartery_slv": "cat",
    "CABG": "cat",
    "cabg_slv": "cat",
    "Angioplasty": "cat",
    "angioplasty_slv": "cat",
    "PeriphVascular": "cat",
    "periphvascular_slv": "cat",
    "HeartAttack": "cat",
    "heartattack_slv": "cat",
    "Stroke": "cat",
    "stroke_slv": "cat",
    "TIA": "cat",
    "tia_slv": "cat",
    "CongestHeartFail": "cat",
    "congestheartfail_slv": "cat",
}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def load_feather(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_feather(path)


def pick_col(df: pd.DataFrame, col: str) -> Tuple[Optional[str], str]:
    """
    Pick the best available column name for a given variable.
    Priority:
      1) exact name: col
      2) clinical-suffixed: col + "_clin"
    Returns:
      (resolved_col_name or None, reason)
    """
    if col in df.columns:
        return col, "as_is"
    alt = f"{col}_clin"
    if alt in df.columns:
        return alt, "used__clin_suffix"
    return None, "missing"


def to_binary01(series: pd.Series) -> pd.Series:
    """
    Coerce a series to {0,1,NA}.
    Many COPDGene event fields are 0/1 or sometimes 1/"Checked" etc.
    We coerce numeric where possible; otherwise map common truthy strings.
    """
    if series is None:
        return pd.Series(pd.NA)

    s = series.copy()

    # numeric fast path
    sn = pd.to_numeric(s, errors="coerce")
    if sn.notna().any():
        # treat nonzero as 1? For these event fields, should be 0/1.
        # We'll be strict: keep only 0/1, other values -> NA (flagged).
        out = pd.Series(pd.NA, index=s.index, dtype="Int64")
        out.loc[sn == 0] = 0
        out.loc[sn == 1] = 1
        # other numeric values remain NA
        return out

    # string path
    ss = s.astype("string").str.strip().str.lower()
    out = pd.Series(pd.NA, index=s.index, dtype="Int64")
    out.loc[ss.isin(["0", "no", "n", "false", "f"])] = 0
    out.loc[ss.isin(["1", "yes", "y", "true", "t", "checked", "check"])] = 1
    return out


def rate_and_na(x01: pd.Series) -> Dict[str, Any]:
    x = x01
    n = int(len(x))
    n_na = int(x.isna().sum())
    n_non = n - n_na
    n1 = int((x == 1).sum())
    n0 = int((x == 0).sum())
    rate1 = float(n1 / n_non) if n_non > 0 else np.nan
    return {
        "n": n,
        "n_nonmissing": n_non,
        "n_na": n_na,
        "n_1": n1,
        "n_0": n0,
        "rate_1_among_nonmissing": rate1,
    }


def crosstab_stats(x: pd.Series, y: pd.Series) -> Dict[str, Any]:
    """
    x,y are Int64 {0,1,NA}.
    Return normalized crosstab plus key conditional rates.
    """
    m = x.notna() & y.notna()
    if int(m.sum()) == 0:
        return {"n_compared": 0}

    xx = x[m].astype(int)
    yy = y[m].astype(int)

    tab = pd.crosstab(xx, yy)  # counts
    # safe extraction
    def get(a, b):
        try:
            return int(tab.loc[a, b])
        except Exception:
            return 0

    n00 = get(0, 0)
    n01 = get(0, 1)
    n10 = get(1, 0)
    n11 = get(1, 1)
    n = n00 + n01 + n10 + n11

    # conditional: P(y=1 | x=0) and P(y=1 | x=1)
    p_y1_x0 = float(n01 / (n00 + n01)) if (n00 + n01) > 0 else np.nan
    p_y1_x1 = float(n11 / (n10 + n11)) if (n10 + n11) > 0 else np.nan

    # "consistency" if y is slv (incident): when y=1, does x=1?
    # i.e., P(x=1 | y=1)
    p_x1_y1 = float(n11 / (n01 + n11)) if (n01 + n11) > 0 else np.nan

    return {
        "n_compared": int(n),
        "counts": {"x0_y0": n00, "x0_y1": n01, "x1_y0": n10, "x1_y1": n11},
        "P(y=1|x=0)": p_y1_x0,
        "P(y=1|x=1)": p_y1_x1,
        "P(x=1|y=1)": p_x1_y1,
    }


def find_pairs(event_cols: Dict[str, str]) -> List[Tuple[str, str]]:
    """
    Return paired variables (base, slv) from an event_cols dict, robust to
    case inconsistencies such as "Stroke" vs "stroke_slv".

    Pairing rule (case-insensitive)
    -------------------------------
    For any key K that ends with "_slv" (case-insensitive),
    let base_lower = K_lower without the suffix "_slv".
    If there exists a non-slv key whose lower() == base_lower, we treat them as a pair.

    If multiple candidates exist for the same base_lower (rare), we select one by:
      1) prefer keys that do NOT end with "_slv" (should be base)
      2) then prefer the shortest key (often the canonical one)
      3) then lexical order (stable tie-break)

    Returns
    -------
    List of (base_key, slv_key) using the original key strings in event_cols.
    Sorted by base_key (case-insensitive).
    """
    keys = list(event_cols.keys())

    # lower -> list of original keys (handle rare collisions)
    lower_to_keys: Dict[str, List[str]] = {}
    for k in keys:
        lower_to_keys.setdefault(k.lower(), []).append(k)

    def _choose_base(candidates: List[str]) -> str:
        # candidates are original keys whose lower() matches base_lower
        # prefer non-slv keys
        non_slv = [c for c in candidates if not c.lower().endswith("_slv")]
        pool = non_slv if non_slv else candidates
        # stable tie-breakers
        pool = sorted(pool, key=lambda x: (len(x), x.lower()))
        return pool[0]

    pairs: List[Tuple[str, str]] = []
    seen = set()

    for k in keys:
        kl = k.lower()
        if not kl.endswith("_slv"):
            continue

        base_lower = kl[:-4]  # remove "_slv"
        candidates = lower_to_keys.get(base_lower, [])
        if not candidates:
            continue

        base_key = _choose_base(candidates)
        pair = (base_key, k)

        # de-dup: avoid duplicates if there are multiple representations
        # (e.g., multiple slv keys mapping to same base)
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)

    pairs.sort(key=lambda t: t[0].lower())
    return pairs


# ------------------------------------------------------------
# Main sanity check
# ------------------------------------------------------------
def sanity_check(df: pd.DataFrame) -> Dict[str, Any]:
    report: Dict[str, Any] = {}

    # (1) check existence of all event cols
    existence = {}
    missing = []
    used_clin_suffix = []
    for col in CVD_EVENT_COLS.keys():
        resolved, reason = pick_col(df, col)
        existence[col] = {"resolved": resolved, "reason": reason}
        if resolved is None:
            missing.append(col)
        elif reason == "used__clin_suffix":
            used_clin_suffix.append(col)

    report["event_cols_existence"] = existence
    report["missing_event_cols"] = missing
    report["n_missing_event_cols"] = int(len(missing))
    report["event_cols_using__clin_suffix"] = used_clin_suffix
    report["n_event_cols_using__clin_suffix"] = int(len(used_clin_suffix))

    # quick phase distribution
    if PHASE_COL in df.columns:
        ph = pd.to_numeric(df[PHASE_COL], errors="coerce")
        report["phase_counts"] = ph.value_counts(dropna=False).to_dict()
    else:
        report["phase_counts"] = "Phase_study column missing!"

    # (2) pairwise x vs x_slv stats
    pairs = find_pairs(CVD_EVENT_COLS)
    pair_reports = {}

    for base, slv in pairs:
        base_col, base_reason = pick_col(df, base)
        slv_col, slv_reason = pick_col(df, slv)

        pr: Dict[str, Any] = {
            "base": {"name": base, "resolved": base_col, "reason": base_reason},
            "slv": {"name": slv, "resolved": slv_col, "reason": slv_reason},
        }

        if base_col is None or slv_col is None:
            pr["status"] = "SKIP_missing_columns"
            pair_reports[f"{base}__{slv}"] = pr
            continue

        # coerce to 0/1/NA
        x_all = to_binary01(df[base_col])
        y_all = to_binary01(df[slv_col])

        # overall stats
        pr["overall_base"] = rate_and_na(x_all)
        pr["overall_slv"] = rate_and_na(y_all)
        pr["overall_crosstab"] = crosstab_stats(x_all, y_all)

        # per-phase stats
        per_phase = {}
        if PHASE_COL in df.columns:
            ph = pd.to_numeric(df[PHASE_COL], errors="coerce")
            for p in [1, 2, 3]:
                idx = (ph == p)
                if int(idx.sum()) == 0:
                    continue
                x = x_all[idx]
                y = y_all[idx]
                per_phase[f"P{p}"] = {
                    "base": rate_and_na(x),
                    "slv": rate_and_na(y),
                    "crosstab": crosstab_stats(x, y),
                }

                # additional specific checks:
                # - Does slv exist / is it mostly NA at P1?
                if p == 1:
                    per_phase[f"P{p}"]["slv_should_be_missing_or_zero_check"] = {
                        "slv_na_rate": float(y.isna().mean()),
                        "slv_rate1_among_nonmissing": per_phase[f"P{p}"]["slv"]["rate_1_among_nonmissing"],
                    }

        pr["per_phase"] = per_phase
        pair_reports[f"{base}__{slv}"] = pr

    report["paired_event_stats"] = pair_reports
    report["paired_event_pairs"] = [f"{a}__{b}" for a, b in pairs]

    return report


def main():
    DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "ct_clin_integration" / "COPD_ct_clinical_merged.feather"      # CT table (COPD optimal CT series only)

    df = load_feather(DATA_PATH)

    # Basic column sanity
    if SID_COL not in df.columns or PHASE_COL not in df.columns:
        raise ValueError(f"Expected columns '{SID_COL}' and '{PHASE_COL}' in merged table.")

    rep = sanity_check(df)

    # Print a concise summary
    print("=" * 80)
    print("[SANITY] CVD_EVENT_COLS existence")
    print("=" * 80)
    print(f"Missing event cols: {rep['n_missing_event_cols']}")
    if rep["missing_event_cols"]:
        print("  ->", rep["missing_event_cols"])
    print(f"Using _clin suffix: {rep['n_event_cols_using__clin_suffix']}")
    if rep["event_cols_using__clin_suffix"]:
        print("  ->", rep["event_cols_using__clin_suffix"])

    print("\n" + "=" * 80)
    print("[SANITY] Phase counts")
    print("=" * 80)
    print(rep["phase_counts"])

    # Print per-pair highlights
    print("\n" + "=" * 80)
    print("[SANITY] Paired (x vs x_slv) highlights (overall)")
    print("=" * 80)
    for k, pr in rep["paired_event_stats"].items():
        if pr.get("status", "").startswith("SKIP"):
            print(f"- {k}: {pr['status']}")
            continue
        base_r1 = pr["overall_base"]["rate_1_among_nonmissing"]
        slv_r1 = pr["overall_slv"]["rate_1_among_nonmissing"]
        p_x1_y1 = pr["overall_crosstab"].get("P(x=1|y=1)", np.nan)
        print(f"- {k}: base_rate1={base_r1:.4f}  slv_rate1={slv_r1:.4f}  P(base=1|slv=1)={p_x1_y1:.4f}")

    # Save full JSON report for detailed inspection
    out_json = DATA_PATH.with_suffix(".cvd_event_sanity.json")
    with open(out_json, "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(f"\n[WRITE] Full report saved to: {out_json}")


if __name__ == "__main__":
    main()
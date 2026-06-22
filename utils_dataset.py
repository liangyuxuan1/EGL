from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import utils_cache

# ============================================================
# 1) Specs: dataset / task / preprocess
# ============================================================

@dataclass(frozen=True)
class DatasetSpec:
    """How to load and canonicalize a dataset."""
    name: str                          # e.g., "COPDGene", "NLST"
    feather_path: Path                 # canonical or source feather
    id_col: str                        # e.g., "sid" or "nlst_id"
    visit_col: Optional[str] = None    # e.g., "Phase_study" or "visit"
    dataset_col: Optional[str] = None  # if a mixed table contains both COPD/NLST
    dataset_value: Optional[str] = None  # e.g., "COPDGene" to filter mixed table

    # Optional: column mapping to canonical names (source_col -> canonical_col)
    colmap: Optional[Dict[str, str]] = None

    # Optional: if you want to enforce (id, visit) uniqueness at the very start
    require_unique_id_visit: bool = True


@dataclass(frozen=True)
class TaskSpec:
    """What label to build and what cohort selection/mode to apply."""
    name: str                      # e.g., "CVD_STATUS", "CVD_PRED_5Y"
    mode: str                      # "Status" or "Prediction" (or more explicit)
    event_cols: Union[List[str], Dict[str, str]]      # columns used to define the disease label/events, list or dict
    label_builder: str             # key into LABEL_BUILDERS registry

    # Optional knobs passed into label builder
    label_kwargs: Dict[str, Any] = None

    def __post_init__(self):
        object.__setattr__(self, "label_kwargs", self.label_kwargs or {})


@dataclass(frozen=True)
class PreprocessSpec:
    meta_csv: Path
    allow_maybe: bool = True
    outlier_p_lo: float = 0.01
    outlier_p_hi: float = 0.99
    missing_rate_threshold: float = 0.7  # candidate pool filtering (you already do)
    # You can add: imputation policy, scaling policy, etc.


@dataclass(frozen=True)
class RunSpec:
    llm_model: Optional[str]
    global_seed: int
    cv_model: str
    warmup_seed_offset: int
    warmup_n_min: int
    warmup_cap_global: int
    warmup_mad_k: float
    warmup_floor_q: float
    warmup_k_grid: Tuple[int, ...]
    warmup_sets_per_k: int
    warmup_anchor_frac: float
    warmup_missing_rate_max: float
    warmup_prefer_non_diag: bool
    run_num: int
    iters: int
    k_min: int
    k_max: int
    p_node: int
    p_edge: int
    edge_from: str
    use_llm_probes: bool
    use_llm_gibbs: bool
    out_root: Path


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    dataset: DatasetSpec
    task: TaskSpec
    preprocess: PreprocessSpec
    run: RunSpec


# ============================================================
# 2) Registry: plug-in style label builders
# ============================================================

LabelBuilderFn = Callable[..., Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]]

LABEL_BUILDERS: Dict[str, LabelBuilderFn] = {}

def register_label_builder(name: str):
    def _wrap(fn: LabelBuilderFn):
        LABEL_BUILDERS[name] = fn
        return fn
    return _wrap


# ============================================================
# 3) Existing helpers you already have (load_data, outlier, pool, var map)
#    (Reuse your existing implementations)
# ============================================================

def load_data(feather_path: Path) -> pd.DataFrame:
    if not feather_path.exists():
        raise FileNotFoundError(f"Data file not found: {feather_path}")
    return pd.read_feather(feather_path)


def _check_unique_id_visit(df: pd.DataFrame, id_col: str, visit_col: Optional[str]) -> None:
    """Fail fast if duplicates exist for the same (id, visit)."""
    if visit_col is None:
        return
    v = pd.to_numeric(df[visit_col], errors="coerce")
    dup_mask = (
        pd.DataFrame({id_col: df[id_col].values, "_visit": v.values})
        .dropna(subset=["_visit"])
        .duplicated(subset=[id_col, "_visit"], keep=False)
    )
    if bool(dup_mask.any()):
        dup_pairs = (
            pd.DataFrame({id_col: df.loc[dup_mask, id_col].values, "_visit": v.loc[dup_mask].values})
            .value_counts()
            .head(10)
            .to_dict()
        )
        raise ValueError(
            f"Found multiple rows for the same ({id_col}, {visit_col}). "
            f"Examples (top 10 with counts): {dup_pairs}"
        )


def build_var_info_map(meta_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Build a lookup map from META_CSV (COPD_ct_variable_summary.csv).

    Why:
    - LLM cannot infer the medical meaning from custom var_name strings (e.g., diag_*).
    - We provide a compact "glossary" to ground the controller decisions.

    Expected columns (from your CSV):
      - var_name, clinical_domain, meaning, missing_rate, notes
    """
    m = {}
    if meta_df is None or meta_df.empty:
        return m
    if "var_name" not in meta_df.columns:
        return m
    for _, r in meta_df.iterrows():
        v = r.get("var_name", None)
        if not isinstance(v, str) or not v:
            continue
        domain = r.get("clinical_domain", "NA")
        meaning = r.get("meaning", "")
        miss = r.get("missing_rate", np.nan)
        notes = r.get("notes", "")
        try:
            miss_f = float(miss)
        except Exception:
            miss_f = np.nan
        m[str(v)] = {
            "domain": str(domain) if isinstance(domain, str) else "NA",
            "meaning": str(meaning) if isinstance(meaning, str) else "",
            "missing_rate": miss_f if np.isfinite(miss_f) else None,
            "notes": str(notes) if isinstance(notes, str) else "",
        }
    return m

# ================================================================
# Candidate pool loader: missing_rate过高的feature过滤掉
# 缺失率 > 0.9：通常信息极少，分析意义很小，建议直接剔除
# 0.7–0.9：高风险区间，只有在该变量临床价值很强、且有可靠的缺失机制/填补方案时才保留
# 0.3–0.7：可保留，但需要明确缺失处理策略（如缺失指示变量 + 合理填补）
# < 0.3：一般可放心使用
# 这里阈值取 0.7
# ================================================================

def load_candidate_pool(df_meta: pd.DataFrame, df: pd.DataFrame, allow_maybe: bool = False,
                        missing_rate_threshold: float = 0.7) -> Tuple[Dict[str, str], pd.DataFrame]:
    for col in ["var_name", "vtype", "used_for_prediction"]:
        if col not in df_meta.columns:
            raise ValueError(f"meta csv missing required column: {col}")

    keep = {"yes"} | ({"maybe"} if allow_maybe else set())
    df_meta = df_meta.copy()
    df_meta["used_for_prediction"] = df_meta["used_for_prediction"].astype(str).str.lower()
    df_meta = df_meta[df_meta["used_for_prediction"].isin(keep)].copy()

    meta = df_meta[df_meta["var_name"].isin(df.columns)].copy()

    if "missing_rate" in meta.columns:
        meta = meta[meta["missing_rate"] <= missing_rate_threshold].copy()

    feature_schema = {r["var_name"]: r["vtype"] for _, r in meta.iterrows()}
    return feature_schema, meta

# ---------------------------------------------------------
# Outlier handling based on metadata
# ---------------------------------------------------------
def _to_bool(x) -> bool:
    """Robust bool casting for meta flags."""
    if x is None:
        return False
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return bool(int(x))
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"1", "true", "t", "yes", "y"}:
            return True
        if s in {"0", "false", "f", "no", "n", ""}:
            return False
    try:
        return bool(x)
    except Exception:
        return False


def filter_outliers_using_meta(
    df: pd.DataFrame,
    meta_csv: Path,
    low_percentile: float = 0.01,
    high_percentile: float = 0.99,
    add_row_flag: bool = True,          # 是否保留 is_outlier_any
    add_per_var_flags: bool = True,     # 是否写回 is_outlier__<var>
    outlier_prefix: str = "is_outlier__",  # per-var flag 前缀
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    - For each eligible numeric float feature, compute per-sample outlier mask:
        mask[col] = (x < low_q) | (x > high_q)   on NON-missing entries only
    - Clip values to [low_q, high_q]
    - Write variable-level outlier summary into meta (returned)
    - Optionally write per-sample flags back to df_out as:
        df_out[f"{outlier_prefix}{col}"] in {0,1}
      For paired raw/indexed:
        raw shares the SAME mask as indexed (mask_source = from:<indexed>)
    - Optionally write coarse row flag: is_outlier_any (any variable outlier)
    """
    meta_path = Path(meta_csv)
    if not meta_path.exists():
        raise FileNotFoundError(f"meta_csv not found: {meta_path}")

    meta = pd.read_csv(meta_path)

    required_cols = {"var_name", "vtype", "dtype", "need_filtering"}
    missing_cols = required_cols - set(meta.columns)
    if missing_cols:
        raise ValueError(f"meta_csv missing required columns: {sorted(missing_cols)}")

    df_out = df.copy()

    # -----------------------------
    # Pair mapping: indexed -> raw
    # -----------------------------
    paired_index_to_raw = {
        "la_volume_indexed": "la_volume",
        "lv_volume_indexed": "lv_volume",
        "ra_volume_indexed": "ra_volume",
        "rv_volume_indexed": "rv_volume",
        "ascending_aorta_index": "ascending_aorta_diameter",
        "pa_index": "pa_diameter",
    }
    raw_partners = set(paired_index_to_raw.values())

    # -----------------------------
    # Prepare meta columns to update
    # -----------------------------
    str_cols = ["outlier_mask_source", "outlier_handling"]
    num_cols = ["outlier_low_q", "outlier_high_q", "n_nonmissing", "n_outliers", "outlier_rate"]

    for col in num_cols:
        if col not in meta.columns:
            meta[col] = np.nan

    for col in str_cols:
        if col not in meta.columns:
            meta[col] = pd.Series(pd.NA, index=meta.index, dtype="string")
        else:
            meta[col] = meta[col].astype("string")

    # -----------------------------
    # Compute quantiles for eligible columns
    #   - include raw partners too (for raw clipping)
    # -----------------------------
    quantiles: Dict[str, Tuple[float, float]] = {}

    def _eligible(col_name: str, row_meta: pd.Series) -> bool:
        vtype = str(row_meta.get("vtype", "")).strip().lower()
        dtype_meta = str(row_meta.get("dtype", "")).strip().lower()
        need_filter = _to_bool(row_meta.get("need_filtering", False))
        return need_filter and (vtype == "num") and (dtype_meta == "float64") and (col_name in df_out.columns)

    # quantiles for need_filtering variables in meta
    for i, row in meta.iterrows():
        col = row["var_name"]
        if not _eligible(col, row):
            continue
        s = pd.to_numeric(df_out[col], errors="coerce").dropna()
        if s.empty:
            continue
        quantiles[col] = (float(s.quantile(low_percentile)), float(s.quantile(high_percentile)))

    # quantiles for raw partners (even if they won't define masks)
    for raw_col in raw_partners:
        if raw_col not in df_out.columns:
            continue
        if raw_col in quantiles:
            continue
        s = pd.to_numeric(df_out[raw_col], errors="coerce").dropna()
        if s.empty:
            continue
        quantiles[raw_col] = (float(s.quantile(low_percentile)), float(s.quantile(high_percentile)))

    # -----------------------------
    # Optional: initialize row flag + per-var flags
    # -----------------------------
    is_outlier_any = pd.Series(False, index=df_out.index)

    # helper to write per-var flags to df_out
    def _write_flag(var: str, mask_bool: pd.Series) -> None:
        if not add_per_var_flags:
            return
        col_flag = f"{outlier_prefix}{var}"
        # int64 0/1, align index
        df_out.loc[:, col_flag] = mask_bool.reindex(df_out.index).fillna(False).astype("int64")

    processed = 0

    # -----------------------------
    # Main pass
    # -----------------------------
    for i, row in meta.iterrows():
        col = row["var_name"]

        if col not in df_out.columns:
            meta.loc[i, "outlier_handling"] = "missing_in_df"
            continue

        vtype = str(row.get("vtype", "")).strip().lower()
        dtype_meta = str(row.get("dtype", "")).strip().lower()
        need_filter = _to_bool(row.get("need_filtering", False))

        if (not need_filter) or (vtype != "num") or (dtype_meta != "float64"):
            meta.loc[i, "outlier_handling"] = "skipped_by_meta"
            continue

        # raw partners never define mask directly; handled when indexed is processed
        if col in raw_partners:
            meta.loc[i, "outlier_handling"] = "raw_partner_skip_mask"
            continue

        if col not in quantiles:
            meta.loc[i, "outlier_handling"] = "skipped_no_quantiles"
            continue

        low_q, high_q = quantiles[col]
        s = pd.to_numeric(df_out[col], errors="coerce")

        non_missing = s.notna()
        mask = pd.Series(False, index=df_out.index)
        mask.loc[non_missing] = (s.loc[non_missing] < low_q) | (s.loc[non_missing] > high_q)

        # clip
        df_out.loc[:, col] = s.clip(lower=low_q, upper=high_q)

        # write per-sample flag for this variable
        _write_flag(col, mask)

        # update coarse row flag
        if add_row_flag:
            is_outlier_any |= mask

        # meta stats
        n_non = int(non_missing.sum())
        n_out = int(mask.sum())
        out_rate = float(n_out / n_non) if n_non > 0 else np.nan

        meta.loc[i, "outlier_low_q"] = float(low_q)
        meta.loc[i, "outlier_high_q"] = float(high_q)
        meta.loc[i, "n_nonmissing"] = n_non
        meta.loc[i, "n_outliers"] = n_out
        meta.loc[i, "outlier_rate"] = out_rate
        meta.loc[i, "outlier_mask_source"] = "self"
        meta.loc[i, "outlier_handling"] = "clipped"

        processed += 1

        # -----------------------------
        # Propagate to paired raw (if any)
        # -----------------------------
        if col in paired_index_to_raw:
            raw_col = paired_index_to_raw[col]
            if raw_col in df_out.columns:
                raw_low, raw_high = quantiles.get(raw_col, (low_q, high_q))
                raw_s = pd.to_numeric(df_out[raw_col], errors="coerce")
                df_out.loc[:, raw_col] = raw_s.clip(lower=raw_low, upper=raw_high)

                # raw shares SAME mask -> write per-sample flag
                _write_flag(raw_col, mask)
                if add_row_flag:
                    is_outlier_any |= mask

                # update meta row for raw if present
                raw_meta_idx = meta.index[meta["var_name"] == raw_col].tolist()
                if raw_meta_idx:
                    j = raw_meta_idx[0]
                    raw_non = raw_s.notna()
                    n_non_r = int(raw_non.sum())
                    n_out_r = int((mask & raw_non).sum())
                    out_rate_r = float(n_out_r / n_non_r) if n_non_r > 0 else np.nan

                    meta.loc[j, "outlier_low_q"] = float(raw_low)
                    meta.loc[j, "outlier_high_q"] = float(raw_high)
                    meta.loc[j, "n_nonmissing"] = n_non_r
                    meta.loc[j, "n_outliers"] = n_out_r
                    meta.loc[j, "outlier_rate"] = out_rate_r
                    meta.loc[j, "outlier_mask_source"] = f"from:{col}"
                    meta.loc[j, "outlier_handling"] = "clipped_sharedmask"

    if processed == 0:
        print("[OUTLIER] No numeric float columns processed (check meta flags / dtype).")

    if add_row_flag:
        df_out["is_outlier_any"] = is_outlier_any.astype("int64")

    return df_out, meta


def class_counts(y: pd.Series) -> Tuple[int, int]:
    pos = int(y.sum())
    neg = int(len(y) - pos)
    return pos, neg


@dataclass(frozen=True)
class DatasetCtx:
    """
    Immutable dataset context.

    Everything inside this ctx is intended to be:
      - computed once (after loading and preprocessing)
      - stable / read-only afterward
      - safely shared across functions without passing many args

    Fields:
    - mode: "Status" or "Prediction"
    - dataset: dataset name (e.g., "COPDGene", "NLST")
    - task: task name (e.g., "CVD_STATUS", "CVD_PRED_5Y")

    - df_all: full dataframe after outlier filtering
    - y_all: label vector aligned with df_all rows
    - prevalence: class prevalence (pos / total)

    - meta_updated: meta table after outlier filtering update
    - var_info_map: variable meaning map for prompting / logging

    - pool_schema: feature schema used by evaluator
    - pool_meta: candidate feature meta table
    - all_features: stable ordered list of candidate features (universe)

    - feature_sig: SHA256 signature of `all_features` order (critical for caching)
    """
    mode: str
    dataset: str
    task: str

    df_all: pd.DataFrame
    y_all: np.ndarray
    prevalence: float

    meta_updated: pd.DataFrame
    var_info_map: Dict[str, Any]

    pool_schema: Dict[str, Any]
    pool_meta: pd.DataFrame
    all_features: List[str]

    feature_sig: str
    stats: Dict[str, Any]

    def n_samples(self) -> int:
        return int(len(self.df_all))

    def n_features(self) -> int:
        return int(len(self.all_features))


# ============================================================
# 4) Canonicalization hook: keep it simple at first
# ============================================================

def canonicalize_df(df: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    """
    Minimal canonicalization:
      - optional dataset filtering if table is mixed
      - optional column renaming via colmap
    """
    out = df.copy()

    if spec.dataset_col and spec.dataset_value is not None:
        if spec.dataset_col not in out.columns:
            raise ValueError(f"dataset_col '{spec.dataset_col}' not found in df for {spec.name}")
        out = out[out[spec.dataset_col].astype(str) == str(spec.dataset_value)].copy()

    if spec.colmap:
        # Only rename keys that exist
        rename = {k: v for k, v in spec.colmap.items() if k in out.columns}
        out = out.rename(columns=rename)

    return out


# ============================================================
# 5) Build dataset ctx: stable pipeline, no dataset/task branching
# ============================================================
def build_dataset_ctx(
    *,
    dataset_spec: DatasetSpec,
    task_spec: TaskSpec,
    preprocess_spec: PreprocessSpec,
) -> DatasetCtx:
    # ---------------------------
    # Load + canonicalize
    # ---------------------------
    df_all = load_data(dataset_spec.feather_path)
    df_all = canonicalize_df(df_all, dataset_spec)

    # Optional uniqueness check
    if dataset_spec.require_unique_id_visit and dataset_spec.visit_col:
        _check_unique_id_visit(df_all, dataset_spec.id_col, dataset_spec.visit_col)

    # ---------------------------
    # Label + cohort selection
    # ---------------------------
    if task_spec.label_builder not in LABEL_BUILDERS:
        raise KeyError(f"Unknown label_builder '{task_spec.label_builder}'. "
                       f"Available: {sorted(LABEL_BUILDERS.keys())}")

    builder = LABEL_BUILDERS[task_spec.label_builder]

    # Standardize argument names for all label builders:
    # they should accept df, event_cols, sid_col, phase_col (if applicable), plus kwargs
    df_all, y_all, stats = builder(
        df = df_all,
        event_cols = task_spec.event_cols,  # Dict[str,str]
        **task_spec.label_kwargs,
    )

    pos, neg = class_counts(y_all)
    prevalence = float(pos) / float(max(1, pos + neg))
    print(f"[INFO] {dataset_spec.name}/{task_spec.name}: {len(df_all)} samples, "
          f"{pos} pos, {neg} neg, prevalence={prevalence:.4f}")

    # ---------------------------
    # Outlier filtering
    # ---------------------------
    df_all, meta_updated = filter_outliers_using_meta(
        df_all,
        preprocess_spec.meta_csv,
        preprocess_spec.outlier_p_lo,
        preprocess_spec.outlier_p_hi,
    )

    # ---------------------------
    # Var map + candidate pool
    # ---------------------------
    var_info_map = build_var_info_map(meta_updated)

    pool_schema, pool_meta = load_candidate_pool(
        meta_updated,
        df=df_all,
        allow_maybe=preprocess_spec.allow_maybe,
        missing_rate_threshold=preprocess_spec.missing_rate_threshold,
    )

    # ---------------------------
    # Stable feature universe ordering (your current policy)
    # ---------------------------
    feats = []
    for x in pool_meta["var_name"].tolist():
        if isinstance(x, str):
            x = x.strip()
            if x:
                feats.append(x)

    seen = set()
    all_features = []
    for f in feats:
        if f not in seen:
            all_features.append(f)
            seen.add(f)

    feature_sig = utils_cache.feature_universe_signature(all_features)
    print("[FEATURE_UNIVERSE_SHA256]", feature_sig)

    return DatasetCtx(
        mode=task_spec.mode,
        dataset=dataset_spec.name,
        task=task_spec.name,
        df_all=df_all,
        y_all=np.asarray(y_all, dtype=int),
        prevalence=prevalence,
        meta_updated=meta_updated,
        var_info_map=var_info_map,
        pool_schema=pool_schema,
        pool_meta=pool_meta,
        all_features=all_features,
        feature_sig=feature_sig,
        stats=stats,
    )


# ============================================================
# 6) Register your existing label builders
# ============================================================

def build_cvd_label_status(
    df: pd.DataFrame,
    event_cols: Dict[str, str],
    *,
    sid_col: str = "sid",
    phase_col: str = "Phase_study",
    col_all: str = "event_all",
    col_5y: str = "event_5y",
    col_10y: str = "event_10y",
    col_start_phase: str = "start_phase",
    # ---- New knobs for "clean status cohort" ----
    clean_status_cohort: bool = True,
    require_phase1_observed: bool = True,
    drop_decrease_patterns: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Plan A (status/outcome modeling): build a ONE-row-per-sid cohort for CVD status.

    Why this Plan A function exists
    -------------------------------
    In status/outcome modeling, we want a cross-sectional cohort where each subject
    contributes exactly one sample. This removes within-subject dependence and allows
    standard row-level CV (no group split needed), which is convenient for updating
    your paper's Plan A results.

    Label definition (Plan A)
    -------------------------
    label(sid) = 1 if the subject has event==1 in ANY observed phase (P1/P2/P3),
                 else 0.

    Row selection rule (A1, conservative)
    -------------------------------------
    For each sid:
      - If sid ever has event==1: select the EARLIEST phase where event==1.
      - Else (never-event): select the EARLIEST observed phase (typically Phase 1).

    "Clean status cohort" option (recommended for main experiments)
    ---------------------------------------------------------------
    Real-world longitudinal EHR-style event flags sometimes show non-monotonic patterns
    such as 1->0 across phases, which is usually noise/inconsistency for chronic outcomes.
    Also, subjects missing Phase 1 can introduce stronger time/visit confounding because
    their single selected row may come from Phase 2 or Phase 3.

    If clean_status_cohort=True, we optionally enforce:
      1) require_phase1_observed:
           keep only sids with Phase 1 observed (P1 not NA).
      2) drop_decrease_patterns:
           drop sids showing event decreases (P1=1,P2=0) or (P2=1,P3=0).

    These filters make the status cohort more consistent and less noisy, and are
    typically safer for a "main" result table. You can run a sensitivity analysis
    with clean_status_cohort=False (or toggling the two flags individually).

    Input assumptions & early validation
    ------------------------------------
    - The input df is expected to have at most one row per (sid, phase).
      We still check this and fail fast if duplicates exist (common merging bug).

    Returns
    -------
    (df_out, label, stats)
      - df_out: one row per sid, copied from df, with added columns:
          * col_all, col_5y, col_10y, col_start_phase
      - label: Plan A status label (1=ever-event, 0=never-event) aligned to df_out rows
      - stats: cohort counts + pattern diagnostics computed before/after cleaning
    """
    # ------------------------------------------------------------
    # 0) Input validation + per-(sid, phase) uniqueness check
    # ------------------------------------------------------------
    missing_cols = [c for c in event_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing CVD event columns: {missing_cols}")
    if sid_col not in df.columns:
        raise ValueError(f"Missing sid column: {sid_col}")
    if phase_col not in df.columns:
        raise ValueError(f"Missing phase column: {phase_col}")

    phase_series_check = pd.to_numeric(df[phase_col], errors="coerce")
    dup_mask = (
        pd.DataFrame({sid_col: df[sid_col].values, "_phase": phase_series_check.values})
        .dropna(subset=["_phase"])
        .duplicated(subset=[sid_col, "_phase"], keep=False)
    )
    if bool(dup_mask.any()):
        dup_pairs = (
            pd.DataFrame(
                {
                    sid_col: df.loc[dup_mask, sid_col].values,
                    "_phase": phase_series_check.loc[dup_mask].values,
                }
            )
            .value_counts()
            .head(10)
            .to_dict()
        )
        raise ValueError(
            "Found multiple rows for the same (sid, phase). "
            "This usually means duplicates were introduced during merging. "
            f"Examples (top 10 (sid, phase) with counts): {dup_pairs}"
        )

    df_out = df.copy()

    # ------------------------------------------------------------
    # 1) Row-level event flag: event_any
    #    IMPORTANT: do NOT treat NA as negative.
    #    Drop rows with no observed event information (all-NA across event keys).
    # ------------------------------------------------------------
    event_keys = list(event_cols.keys())

    # Coerce event columns to numeric {0,1,NA}. Keep ONLY 0/1; others -> NA.
    events_raw = df_out[event_keys].apply(pd.to_numeric, errors="coerce")
    events01 = events_raw.where(events_raw.isin([0, 1]), other=pd.NA)

    # Drop rows where ALL event columns are missing (no label information).
    has_event_info = events01.notna().any(axis=1)
    n_drop_no_event_info = int((~has_event_info).sum())
    if n_drop_no_event_info > 0:
        df_out = df_out.loc[has_event_info].copy()
        events01 = events01.loc[has_event_info].copy()

    # Compute event_any without converting NA->0.
    event_any = (events01 == 1).any(axis=1).astype(int)
    df_out[col_all] = event_any

    # ---------------------------------------------------------
    # 2) Per-subject/per-phase event presence: P1, P2, P3
    # ---------------------------------------------------------
    phase_series_raw = pd.to_numeric(df_out[phase_col], errors="coerce")
    phase_event = (
        pd.DataFrame(
            {
                sid_col: df_out[sid_col].values,
                "_phase": phase_series_raw.values,
                "_event": df_out[col_all].values,  # use col_all directly
            }
        )
        .dropna(subset=["_phase"])
        .groupby([sid_col, "_phase"], dropna=False)["_event"]
        .max()
        .unstack("_phase")
    )
    phase_event = phase_event.reindex(columns=[1, 2, 3])
    p1 = phase_event.get(1)
    p2 = phase_event.get(2)
    p3 = phase_event.get(3)

    # ---------------------------------------------------------
    # 3) Horizon flags (diagnostics only; NOT used as Plan A label)
    # ---------------------------------------------------------
    event_5y_p1 = ((p1 == 0) & (p2 == 1)).astype(int)  # P1->P2 conversion
    event_5y_p2 = ((p2 == 0) & (p3 == 1)).astype(int)  # P2->P3 conversion
    event_10y = ((p1 == 0) & (p2 == 0) & (p3 == 1)).astype(int)

    sid_to_5y_p1 = df_out[sid_col].map(event_5y_p1)
    sid_to_5y_p2 = df_out[sid_col].map(event_5y_p2)
    sid_to_10y = df_out[sid_col].map(event_10y)

    event_5y_row = np.where(
        phase_series_raw == 1,
        sid_to_5y_p1,
        np.where(phase_series_raw == 2, sid_to_5y_p2, 0),
    )
    event_10y_row = np.where(phase_series_raw == 1, sid_to_10y, 0)

    df_out[col_5y] = pd.Series(event_5y_row, index=df_out.index).fillna(0).astype(int)
    df_out[col_10y] = pd.Series(event_10y_row, index=df_out.index).fillna(0).astype(int)

    # ---------------------------------------------------------
    # 4) Pattern diagnostics BEFORE cleaning (sid-level)
    # ---------------------------------------------------------
    pattern_df_all = phase_event.rename(columns={1: "P1", 2: "P2", 3: "P3"}).copy()
    pattern_df_all = pattern_df_all.where(~pattern_df_all.isna(), other="NA")
    pattern_str_all = (
        "P1=" + pattern_df_all["P1"].astype(str)
        + ",P2=" + pattern_df_all["P2"].astype(str)
        + ",P3=" + pattern_df_all["P3"].astype(str)
    )
    pattern_counts_before = pattern_str_all.value_counts(dropna=False).to_dict()

    decrease_1_to_2_before = int(((p1 == 1) & (p2 == 0)).sum())
    decrease_2_to_3_before = int(((p2 == 1) & (p3 == 0)).sum())
    decrease_1_to_3_before = int(((p1 == 1) & (p3 == 0)).sum())

    # ---------------------------------------------------------
    # 5) Optional "clean status cohort" filtering at SID level
    #     IMPORTANT: do NOT recompute event_any with fillna(0).
    #     We keep df_out[col_all] as the source of truth.
    # ---------------------------------------------------------
    sid_keep = pd.Series(True, index=phase_event.index)

    if clean_status_cohort:
        if require_phase1_observed:
            sid_keep &= p1.notna()

        if drop_decrease_patterns:
            sid_keep &= ~((p1 == 1) & (p2 == 0))
            sid_keep &= ~((p2 == 1) & (p3 == 0))

    kept_sids = sid_keep[sid_keep].index
    df_out = df_out[df_out[sid_col].isin(kept_sids)].copy()

    # Recompute phase_event for the kept cohort using the existing col_all
    phase_series_raw = pd.to_numeric(df_out[phase_col], errors="coerce")
    phase_event_kept = (
        pd.DataFrame(
            {
                sid_col: df_out[sid_col].values,
                "_phase": phase_series_raw.values,
                "_event": df_out[col_all].values,  # reuse col_all; no fillna(0)
            }
        )
        .dropna(subset=["_phase"])
        .groupby([sid_col, "_phase"], dropna=False)["_event"]
        .max()
        .unstack("_phase")
    )
    phase_event_kept = phase_event_kept.reindex(columns=[1, 2, 3])
    p1k = phase_event_kept.get(1)
    p2k = phase_event_kept.get(2)
    p3k = phase_event_kept.get(3)

    # ---------------------------------------------------------
    # 6) Plan A cohort selection: EXACTLY ONE row per sid
    # ---------------------------------------------------------
    sid_has_event = (p1k == 1) | (p2k == 1) | (p3k == 1)

    chosen_phase_by_sid = pd.Series(index=phase_event_kept.index, dtype="float")

    # earliest positive phase among {1,2,3}
    pos_phase_min = []
    for ph, vec in [(1, p1k), (2, p2k), (3, p3k)]:
        pos_phase_min.append(pd.Series(np.where(vec == 1, ph, np.nan), index=phase_event_kept.index))
    pos_phase_min = pd.concat(pos_phase_min, axis=1).min(axis=1)

    # earliest observed phase among {1,2,3}
    obs_phase_min = []
    for ph, vec in [(1, p1k), (2, p2k), (3, p3k)]:
        obs_phase_min.append(pd.Series(np.where(~vec.isna(), ph, np.nan), index=phase_event_kept.index))
    obs_phase_min = pd.concat(obs_phase_min, axis=1).min(axis=1)

    sid_has_event_f = sid_has_event.fillna(False)
    chosen_phase_by_sid.loc[sid_has_event_f] = pos_phase_min.loc[sid_has_event_f]
    chosen_phase_by_sid.loc[~sid_has_event_f] = obs_phase_min.loc[~sid_has_event_f]

    sid_series = df_out[sid_col]
    phase_series = pd.to_numeric(df_out[phase_col], errors="coerce")
    chosen_phase_row = sid_series.map(chosen_phase_by_sid)

    keep_mask = chosen_phase_row.notna() & (phase_series == chosen_phase_row)
    df_out = df_out.loc[keep_mask].copy()

    vc = df_out[sid_col].value_counts()
    if (vc > 1).any():
        bad = vc[vc > 1].head(10).to_dict()
        raise RuntimeError(
            f"Internal error: df_out has multiple rows per sid after selection. Examples: {bad}"
        )

    df_out[col_start_phase] = pd.to_numeric(df_out[phase_col], errors="coerce")

    label = df_out[sid_col].map(sid_has_event).fillna(False).astype(int)
    label = pd.Series(label.values, index=df_out.index, dtype=int)
    label.name = "label"
    df_out["label"] = label

    # ---------------------------------------------------------
    # 7) Pattern diagnostics AFTER cleaning (sid-level)
    # ---------------------------------------------------------
    pattern_df_after = phase_event_kept.rename(columns={1: "P1", 2: "P2", 3: "P3"}).copy()
    pattern_df_after = pattern_df_after.where(~pattern_df_after.isna(), other="NA")
    pattern_str_after = (
        "P1=" + pattern_df_after["P1"].astype(str)
        + ",P2=" + pattern_df_after["P2"].astype(str)
        + ",P3=" + pattern_df_after["P3"].astype(str)
    )
    pattern_counts_after = pattern_str_after.value_counts(dropna=False).to_dict()

    decrease_1_to_2_after = int(((p1k == 1) & (p2k == 0)).sum())
    decrease_2_to_3_after = int(((p2k == 1) & (p3k == 0)).sum())
    decrease_1_to_3_after = int(((p1k == 1) & (p3k == 0)).sum())

    # ---------------------------------------------------------
    # 8) Summary stats (after selection: one row per sid)
    # ---------------------------------------------------------
    sid_total = int(df_out.shape[0])
    sid_pos = int(label.sum())
    sid_neg = int(sid_total - sid_pos)
    sid_prev = float(sid_pos / sid_total) if sid_total > 0 else 0.0

    stats: Dict[str, Any] = {
        # New: how many rows dropped due to missing event information
        "dropped_rows_no_event_info": int(n_drop_no_event_info),

        # BEFORE vs AFTER cohort cleaning (sid-level patterns)
        "pattern_counts_before": pattern_counts_before,
        "pattern_counts_after": pattern_counts_after,
        "decrease_1_to_2_before": decrease_1_to_2_before,
        "decrease_2_to_3_before": decrease_2_to_3_before,
        "decrease_1_to_3_before": decrease_1_to_3_before,
        "decrease_1_to_2_after": decrease_1_to_2_after,
        "decrease_2_to_3_after": decrease_2_to_3_after,
        "decrease_1_to_3_after": decrease_1_to_3_after,

        # Final cohort size/prevalence (one row per sid)
        "sid_total": sid_total,
        "sid_pos": sid_pos,
        "sid_neg": sid_neg,
        "sid_prevalence": sid_prev,

        # sample-level == sid-level because output is one row per sid
        "sample_total": sid_total,
        "sample_pos": sid_pos,
        "sample_neg": sid_neg,
        "sample_prevalence": sid_prev,

        # How many selected rows come from each phase (helps diagnose time confounding)
        "chosen_phase_counts": df_out[col_start_phase].value_counts(dropna=False).to_dict(),

        # Record which cleaning knobs were applied
        "clean_status_cohort": bool(clean_status_cohort),
        "require_phase1_observed": bool(require_phase1_observed) if clean_status_cohort else False,
        "drop_decrease_patterns": bool(drop_decrease_patterns) if clean_status_cohort else False,
    }

    return df_out, label, stats


def build_cvd_label_prediction(
    df: pd.DataFrame,
    event_cols: Dict[str, str],
    *,
    sid_col: str = "sid",
    phase_col: str = "Phase_study",
    col_all: str = "event_all",
    col_5y: str = "event_5y",
    col_10y: str = "event_10y",
    col_start_phase: str = "start_phase",
    clean_prediction_cohort: bool = True,
    drop_decrease_patterns: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Plan B (rolling 5-year prediction) for an ASCVD endpoint.

    Endpoint definition (ASCVD)
    ---------------------------
    In the current project setup, we operationalize ASCVD using visit-level diagnosis
    indicators from the integrated CT table (e.g., `diag_cvd`), which is a composite
    OR of:
      - coronary artery disease / angina
      - myocardial infarction / heart attack
      - stroke / TIA
      - peripheral vascular disease

    IMPORTANT: this is a VISIT-LEVEL STATUS endpoint (ever/diagnosed by the visit),
    not a perfectly adjudicated time-to-event record. Therefore, the rolling label
    below should be interpreted as:
        "ASCVD status transition from 0 to 1 within the inter-visit window"
    rather than a precise incident-time label.

    What this function returns
    --------------------------
    This function returns a *row-level* binary label for a 5-year rolling prediction task:
      - label = 1: the subject is a 0->1 converter within a 5-year window, and we keep
                   the START-OF-WINDOW row as the positive sample.
      - label = 0: the subject is confirmed event-free over the full window, and we keep
                   exactly ONE eligible start-of-window row as a negative sample.

    Mapping to COPDGene phases
    --------------------------
    COPDGene provides up to three visits (phases) roughly spaced by ~5 years.
    We treat:
      - Phase 1 -> Phase 2 as one 5-year prediction window
      - Phase 2 -> Phase 3 as another 5-year prediction window

    Label construction (5-year rolling)
    -----------------------------------
    1) Build a per-row event indicator `event_any` from `event_cols`:
         event_any(row) = OR_{c in event_cols} [c == 1]
       NA handling is crucial:
         - We DO NOT treat NA as 0 (negative).
         - Rows where ALL event columns are NA are dropped (no label information).

    2) Aggregate `event_any` to per-subject/per-phase presence:
         Pk(sid) in {0,1,NA} for k in {1,2,3}
       where Pk=1 means ASCVD-positive at that phase, Pk=0 means explicitly negative,
       and NA means that phase is not observed for that subject.

    3) Define *conversion* patterns (subject-level):
       - 5y conversion at start Phase 1:
           converter_p1:  P1 == 0 AND P2 == 1
       - 5y conversion at start Phase 2:
           converter_p2:  P2 == 0 AND P3 == 1
       - If both are true (rare), we prefer the earlier one (start at Phase 1).

       These conversion definitions implement a "baseline exclusion" requirement:
       the start phase must be event-free (0) so we do NOT predict ASCVD using an
       already-positive visit.

    4) Define NEGATIVE eligibility with censoring control:
       A negative start-of-window sample is only valid if we can *observe the window endpoint*
       and confirm the subject remained event-free through that window:
         - Start at Phase 1 as negative requires: P2 observed AND P2 == 0
         - Start at Phase 2 as negative requires: P3 observed AND P3 == 0

       Additionally, we restrict negatives to "never-event" subjects:
         never_event(sid): (P1,P2,P3) contains no 1s in any observed phase.
       This is conservative, but it prevents label contamination due to ambiguous status.

    5) Output cohort is one-row-per-sid:
       - For positives: keep ONLY the start-phase row (Phase 1 or Phase 2) where conversion begins.
       - For negatives: keep the earliest eligible negative start-phase row per subject
                        (Phase 1 if eligible, otherwise Phase 2).

    Columns added to df_out
    -----------------------
    - col_all: per-row event_any (ASCVD status at that visit)
    - col_5y / col_10y: diagnostic horizon flags mapped back to rows (NOT the training label)
    - col_start_phase: the chosen start phase for the kept training row (1 or 2)
    - label: the final 5-year rolling prediction label aligned to df_out rows

    Returns
    -------
    (df_filtered, label, stats)
      - df_filtered: filtered dataframe containing exactly one row per sid
      - label: row-level 5-year rolling prediction label (1=converter start, 0=confirmed negative)
      - stats: diagnostics (pattern counts, prevalence, how many rows were dropped due to missing labels, etc.)
    """
    # ------------------------------------------------------------
    # 0) Basic input validation + per-(sid, phase) uniqueness check
    # ------------------------------------------------------------
    missing_cols = [c for c in event_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing CVD event columns: {missing_cols}")
    if sid_col not in df.columns:
        raise ValueError(f"Missing sid column: {sid_col}")
    if phase_col not in df.columns:
        raise ValueError(f"Missing phase column: {phase_col}")

    phase_series_check = pd.to_numeric(df[phase_col], errors="coerce")
    dup_mask = (
        pd.DataFrame({sid_col: df[sid_col].values, "_phase": phase_series_check.values})
        .dropna(subset=["_phase"])
        .duplicated(subset=[sid_col, "_phase"], keep=False)
    )
    if bool(dup_mask.any()):
        dup_pairs = (
            pd.DataFrame(
                {
                    sid_col: df.loc[dup_mask, sid_col].values,
                    "_phase": phase_series_check.loc[dup_mask].values,
                }
            )
            .value_counts()
            .head(10)
            .to_dict()
        )
        raise ValueError(
            "Found multiple rows for the same (sid, phase). "
            "This usually means duplicates were introduced during merging. "
            f"Examples (top 10 (sid, phase) with counts): {dup_pairs}"
        )

    df_out = df.copy()

    # ------------------------------------------------------------
    # 1) Row-level event flag: event_any (no NA->0)
    #    Drop rows with no observed event information.
    # ------------------------------------------------------------
    event_keys = list(event_cols.keys())

    # Coerce to numeric {0,1,NA}. Keep only 0/1; others -> NA.
    events_raw = df_out[event_keys].apply(pd.to_numeric, errors="coerce")
    events01 = events_raw.where(events_raw.isin([0, 1]), other=pd.NA)

    # Drop rows where ALL event columns are missing (no label information).
    has_event_info = events01.notna().any(axis=1)
    n_drop_no_event_info = int((~has_event_info).sum())
    if n_drop_no_event_info > 0:
        df_out = df_out.loc[has_event_info].copy()
        events01 = events01.loc[has_event_info].copy()

    # event_any: OR across explicit 1s; NA contributes nothing.
    event_any = (events01 == 1).any(axis=1).astype(int)
    df_out[col_all] = event_any

    # ---------------------------------------------------------
    # 2) Per-subject/per-phase event presence: P1, P2, P3
    # ---------------------------------------------------------
    phase_series_raw = pd.to_numeric(df_out[phase_col], errors="coerce")
    phase_event = (
        pd.DataFrame(
            {
                sid_col: df_out[sid_col].values,
                "_phase": phase_series_raw.values,
                "_event": df_out[col_all].values,  # reuse col_all
            }
        )
        .dropna(subset=["_phase"])
        .groupby([sid_col, "_phase"], dropna=False)["_event"]
        .max()
        .unstack("_phase")
    )
    phase_event = phase_event.reindex(columns=[1, 2, 3])
    p1 = phase_event.get(1)
    p2 = phase_event.get(2)
    p3 = phase_event.get(3)

    decrease_pattern = ((p1 == 1) & (p2 == 0)) | ((p2 == 1) & (p3 == 0))
    decrease_pattern = decrease_pattern.fillna(False)
    n_drop_decrease_sids = (
        int(decrease_pattern.sum())
        if clean_prediction_cohort and drop_decrease_patterns
        else 0
    )
    if clean_prediction_cohort and drop_decrease_patterns:
        kept_sids = decrease_pattern[~decrease_pattern].index
        df_out = df_out[df_out[sid_col].isin(kept_sids)].copy()
        phase_series_raw = pd.to_numeric(df_out[phase_col], errors="coerce")
        phase_event = phase_event.loc[kept_sids].copy()
        p1 = phase_event.get(1)
        p2 = phase_event.get(2)
        p3 = phase_event.get(3)

    # ---------------------------------------------------------
    # 3) Horizon flags (diagnostics only)
    # ---------------------------------------------------------
    event_5y_p1 = ((p1 == 0) & (p2 == 1)).astype(int)   # P1->P2 conversion
    event_5y_p2 = ((p2 == 0) & (p3 == 1)).astype(int)   # P2->P3 conversion
    event_10y = ((p1 == 0) & (p2 == 0) & (p3 == 1)).astype(int)

    sid_to_5y_p1 = df_out[sid_col].map(event_5y_p1)
    sid_to_5y_p2 = df_out[sid_col].map(event_5y_p2)
    sid_to_10y = df_out[sid_col].map(event_10y)

    event_5y_row = np.where(
        phase_series_raw == 1,
        sid_to_5y_p1,
        np.where(phase_series_raw == 2, sid_to_5y_p2, 0),
    )
    event_10y_row = np.where(phase_series_raw == 1, sid_to_10y, 0)

    df_out[col_5y] = pd.Series(event_5y_row, index=df_out.index).fillna(0).astype(int)
    df_out[col_10y] = pd.Series(event_10y_row, index=df_out.index).fillna(0).astype(int)

    # ---------------------------------------------------------
    # 4) Rolling 5-year prediction cohort construction
    # ---------------------------------------------------------
    # Positive subjects:
    # - P1=0,P2=1 -> start_phase=1
    # - P2=0,P3=1 -> start_phase=2 (only if not already pos_p1)
    pos_p1 = (p1 == 0) & (p2 == 1)
    pos_p2 = (p2 == 0) & (p3 == 1)

    # "Never-event" subjects: no observed event in any phase.
    sid_has_event = (p1 == 1) | (p2 == 1) | (p3 == 1)
    never_event_sid = ~sid_has_event

    # Map sid -> positive start phase (1 or 2); NaN otherwise
    sid_to_start_phase = pd.Series(index=phase_event.index, dtype="float")
    sid_to_start_phase.loc[pos_p1] = 1
    sid_to_start_phase.loc[pos_p2 & ~pos_p1] = 2  # prefer earlier conversion

    phase_series = pd.to_numeric(df_out[phase_col], errors="coerce")
    sid_series = df_out[sid_col]
    start_phase_row = sid_series.map(sid_to_start_phase)

    # Positive rows: keep ONLY the start phase row
    pos_rows = (start_phase_row.notna()) & (phase_series == start_phase_row)

    # ---------------------------------------------------------
    # Negative rows (censoring-aware)
    # ---------------------------------------------------------
    # To label a start-phase row as negative, we require:
    # - subject is never-event (no phase has event==1)
    # - the 5y window endpoint is observed AND event-free:
    #     * start at P1 -> require P2 observed and P2==0
    #     * start at P2 -> require P3 observed and P3==0
    sid_to_p2 = pd.Series(p2, index=phase_event.index)
    sid_to_p3 = pd.Series(p3, index=phase_event.index)

    neg_ok_start1_by_sid = never_event_sid & (sid_to_p2 == 0)
    neg_ok_start2_by_sid = never_event_sid & (sid_to_p3 == 0)

    neg_eligible_rows = (
        ((phase_series == 1) & sid_series.map(neg_ok_start1_by_sid).fillna(False)) |
        ((phase_series == 2) & sid_series.map(neg_ok_start2_by_sid).fillna(False))
    )

    # Pick exactly one eligible negative row per sid using a deterministic
    # earliest-window policy: Phase 1 if eligible, otherwise Phase 2.
    eligible_df = df_out.loc[neg_eligible_rows, [sid_col, phase_col]].copy()
    neg_idx = (
        eligible_df.sort_values([sid_col, phase_col])
        .drop_duplicates(subset=[sid_col], keep="first")
        .index
    )

    neg_rows = pd.Series(False, index=df_out.index)
    neg_rows.loc[neg_idx] = True

    # ---------------------------------------------------------
    # 5) Filter to final cohort (one row per sid)
    # ---------------------------------------------------------
    keep_mask = pos_rows | neg_rows
    df_out = df_out.loc[keep_mask].copy()

    # start_phase for kept rows
    df_out[col_start_phase] = np.where(
        pos_rows.loc[df_out.index],
        start_phase_row.loc[df_out.index],
        pd.to_numeric(df_out[phase_col], errors="coerce"),
    )

    # Final rolling label: 1 only for positive start-phase rows
    label = pd.Series(0, index=df_out.index, dtype=int)
    label.loc[pos_rows.loc[df_out.index]] = 1
    label.name = "label"
    df_out["label"] = label

    # ---------------------------------------------------------
    # 6) Stats (computed on df_out only; reuse col_all)
    # ---------------------------------------------------------
    phase_series_tmp = pd.to_numeric(df_out[phase_col], errors="coerce")
    phase_event_tmp = (
        pd.DataFrame(
            {
                sid_col: df_out[sid_col].values,
                "_phase": phase_series_tmp.values,
                "_event": df_out[col_all].values,  # reuse col_all; no fillna(0)
            }
        )
        .dropna(subset=["_phase"])
        .groupby([sid_col, "_phase"], dropna=False)["_event"]
        .max()
        .unstack("_phase")
    )
    phase_event_tmp = phase_event_tmp.reindex(columns=[1, 2, 3])
    p1_tmp = phase_event_tmp.get(1)
    p2_tmp = phase_event_tmp.get(2)
    p3_tmp = phase_event_tmp.get(3)

    pattern_df = phase_event_tmp.rename(columns={1: "P1", 2: "P2", 3: "P3"})
    pattern_df = pattern_df.where(~pattern_df.isna(), other="NA")
    pattern_str = (
        "P1=" + pattern_df["P1"].astype(str)
        + ",P2=" + pattern_df["P2"].astype(str)
        + ",P3=" + pattern_df["P3"].astype(str)
    )

    decrease_1_to_2 = int(((p1_tmp == 1) & (p2_tmp == 0)).sum())
    decrease_2_to_3 = int(((p2_tmp == 1) & (p3_tmp == 0)).sum())
    decrease_1_to_3 = int(((p1_tmp == 1) & (p3_tmp == 0)).sum())

    sid_label = label.groupby(df_out[sid_col]).max()
    sid_total = int(sid_label.shape[0])
    sid_pos = int(sid_label.sum())
    sid_neg = int(sid_total - sid_pos)
    sid_prev = float(sid_pos / sid_total) if sid_total > 0 else 0.0

    sample_total = int(label.shape[0])
    sample_pos = int(label.sum())
    sample_neg = int(sample_total - sample_pos)
    sample_prev = float(sample_pos / sample_total) if sample_total > 0 else 0.0

    pos_sid_by_start = (
        df_out.loc[label == 1, [sid_col, col_start_phase]]
        .dropna(subset=[col_start_phase])
        .drop_duplicates(subset=[sid_col])
    )
    sid_pos_p1 = int((pos_sid_by_start[col_start_phase] == 1).sum())
    sid_pos_p2 = int((pos_sid_by_start[col_start_phase] == 2).sum())

    stats: Dict[str, Any] = {
        # NA/drop diagnostics
        "dropped_rows_no_event_info": int(n_drop_no_event_info),

        # pattern diagnostics on the final cohort
        "pattern_counts": pattern_str.value_counts(dropna=False).to_dict(),
        "decrease_1_to_2": decrease_1_to_2,
        "decrease_2_to_3": decrease_2_to_3,
        "decrease_1_to_3": decrease_1_to_3,

        # cohort sizes
        "sid_total": sid_total,
        "sid_pos": sid_pos,
        "sid_neg": sid_neg,
        "sid_prevalence": sid_prev,
        "sid_pos_p1": sid_pos_p1,
        "sid_pos_p2": sid_pos_p2,

        "sample_total": sample_total,
        "sample_pos": sample_pos,
        "sample_neg": sample_neg,
        "sample_prevalence": sample_prev,

        # negative eligibility counts (sid-level)
        "neg_ok_start1_sid_count": int(neg_ok_start1_by_sid.sum()),
        "neg_ok_start2_sid_count": int(neg_ok_start2_by_sid.sum()),

        # clean prediction cohort diagnostics
        "negative_sample_policy": "earliest_eligible_window",
        "clean_prediction_cohort": bool(clean_prediction_cohort),
        "drop_decrease_patterns": bool(drop_decrease_patterns) if clean_prediction_cohort else False,
        "dropped_sids_decrease_pattern": n_drop_decrease_sids,
    }

    return df_out, label, stats


@register_label_builder("CVD_STATUS")
def _lb_cvd_status(df: pd.DataFrame, event_cols: Dict[str, str], *,
                   sid_col: str = "sid", phase_col: str = "Phase_study", **kwargs):
    # call your existing build_cvd_label_status
    return build_cvd_label_status(df, event_cols, sid_col=sid_col, phase_col=phase_col, **kwargs)


@register_label_builder("CVD_PRED_5Y")
def _lb_cvd_pred(df: pd.DataFrame, event_cols: Dict[str, str], *,
                 sid_col: str = "sid", phase_col: str = "Phase_study", **kwargs):
    # call your existing build_cvd_label_prediction
    return build_cvd_label_prediction(df, event_cols, sid_col=sid_col, phase_col=phase_col, **kwargs)

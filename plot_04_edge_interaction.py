#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Edge interaction effect analysis (S04/S05)
==========================================

Goal
----
Given:
  - stable node/edge tables from S04S05_plot_stable_kg.py (process_one)
  - prototype feature distribution from S04S05_plot_path_agenda.py (compute_basin_prototype_and_hitting_time)
We analyze how TOP positive/negative interaction edges affect CV metrics (FP/FN/etc.)
under a clearly defined background feature set.

Key idea (2x2 cell sets per edge u--v)
-------------------------------------
Let B be a chosen "background" feature set (proto or stable nodes, etc.).
For an edge (u,v), define four feature sets:

  S11 = B                   (both u and v present)
  S10 = B - {v}             (only u present)
  S01 = B - {u}             (only v present)
  S00 = B - {u,v}           (neither present)

Evaluate each set with your CV pipeline and obtain a metrics dict, e.g.:
  {"auc":..., "cindex":..., "fp":..., "fn":..., "spec":..., "sens":..., ...}

Then compute:
  - main effects:
      Δ_u = metric(S11) - metric(S01)     (adding u when v is present)
      Δ_v = metric(S11) - metric(S10)     (adding v when u is present)
  - interaction (synergy / redundancy) on that metric:
      Int = metric(S11) - metric(S10) - metric(S01) + metric(S00)

Interpretation (for "higher is better" metrics like AUC/C-index):
  Int < 0 : diminishing returns / redundancy (u and v overlap in what they explain)
  Int > 0 : synergy (u and v complement each other)

For "lower is better" metrics like FP/FN:
  Int < 0 means the joint presence reduces FP/FN more than expected (good synergy for errors),
  so ALWAYS interpret sign together with metric direction; script will store both raw deltas.

What you need to provide/plug in
--------------------------------
1) How to evaluate a feature set and return (metrics_dict, scalar_score).
   - In your codebase you already have eval_set_to_score(...) or something similar.
   - Here we accept an evaluate_set_fn callback. You can:
       a) import your eval function directly
       b) or read from eval_cache SQLite if your cache stores per-set metrics

2) Paths (out_root, method names) consistent with your existing scripts.

This script does NOT change your existing pipeline. It is a pure "analysis" utility.
"""

import os
import json
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple, Optional, Callable

import numpy as np
import pandas as pd
from pathlib import Path
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from tqdm import tqdm

import plot_03_path_agenda as path_agenda
import plot_02_stable_kg as stable_kg
import utils_models
import utils_dataset
from config_cvd import CVD_EVENT_COLS
DATA_FEATHER = Path("data/COPD_ct_clinical_merged.feather")
META_CSV     = Path("data/COPD_ct_variable_summary.csv")
CV_MODEL = "Logistic"

from utils_plot import set_paper_style
set_paper_style()


def _load_saved_dataset_ctx(out_root: str, method: str):
    """
    Restore the minimal dataset context needed for post-hoc edge interaction plots.

    The main EGL runs already save the filtered dataframe, labels, and candidate
    pool under each output directory. Loading those artifacts keeps this plotting
    script aligned with the exact task/run being analyzed and avoids relying on
    the evolving build_dataset_ctx API.
    """
    run_dir = Path(out_root) / method
    df_path = run_dir / "df_all.feather"
    pool_meta_path = run_dir / "pool_meta.csv"
    stats_path = run_dir / "ds_stats.json"

    if not df_path.exists():
        raise FileNotFoundError(f"Missing saved dataframe: {df_path}")
    if not pool_meta_path.exists():
        raise FileNotFoundError(f"Missing saved pool metadata: {pool_meta_path}")

    df_all = pd.read_feather(df_path)
    if "label" not in df_all.columns:
        raise KeyError(f"Saved dataframe does not contain required 'label' column: {df_path}")

    pool_meta = pd.read_csv(pool_meta_path)
    required = {"var_name", "vtype"}
    missing = required.difference(pool_meta.columns)
    if missing:
        raise KeyError(f"Saved pool metadata is missing columns {sorted(missing)}: {pool_meta_path}")

    pool_schema = {
        str(row.var_name): str(row.vtype)
        for row in pool_meta.itertuples(index=False)
        if isinstance(row.var_name, str) and row.var_name in df_all.columns
    }

    stats = {}
    if stats_path.exists():
        with open(stats_path, "r") as f:
            stats = json.load(f)

    y_all = df_all["label"].astype(int).to_numpy()
    pos = int(np.sum(y_all == 1))
    neg = int(np.sum(y_all == 0))
    print(
        f"[INFO] Loaded saved dataset context from {run_dir}: "
        f"{len(df_all)} samples, {pos} pos, {neg} neg, {len(pool_schema)} candidate features"
    )

    return SimpleNamespace(
        df_all=df_all,
        y_all=y_all,
        pool_schema=pool_schema,
        pool_meta=pool_meta,
        stats=stats,
    )

# ------------------------------------------------------------
# 2) Select edges to analyze (top positive / negative)
# ------------------------------------------------------------
def select_top_edges(
    edge_df: pd.DataFrame,
    *,
    u_col: str,
    v_col: str,
    top_k: int,
) -> pd.DataFrame:
    """
    Pick edges by interaction strength (positive and negative), optionally filtering by stability.

    Returns a dataframe with standardized columns: u, v, delta_mean, delta_conf
    """
    df = edge_df.sort_values("delta_conf", ascending=False).copy()
    df = df.head(int(top_k)).copy()

    d = df[[u_col, v_col, "delta_mean", "delta_conf"]].dropna().copy()

    d["delta_mean"] = d["delta_mean"].astype(float)
    d["delta_conf"] = d["delta_conf"].astype(float)

    out = d.drop_duplicates(["u", "v"]).reset_index(drop=True)
    out["sign_group"] = np.where(out["delta_mean"] >= 0, "positive", "negative")
    return out


# ------------------------------------------------------------
# 3) Build the 2x2 sets for one edge
# ------------------------------------------------------------
def make_2x2_sets(background: List[str], u: str, v: str) -> Dict[str, List[str]]:
    """
    Return dict of feature lists for S11/S10/S01/S00.
    Lists are sorted for stable hashing/cache keys.
    """
    B = list(dict.fromkeys(background))
    Bset = set(B)
    if u not in Bset:
        B.append(u)
        Bset.add(u)
    if v not in Bset:
        B.append(v)
        Bset.add(v)

    def _sorted(x: List[str]) -> List[str]:
        # stable deterministic order for caching
        return sorted(list(dict.fromkeys(x)))

    S11 = _sorted(B)
    S10 = _sorted([x for x in B if x != v])
    S01 = _sorted([x for x in B if x != u])
    S00 = _sorted([x for x in B if (x != u and x != v)])

    return {"11": S11, "10": S10, "01": S01, "00": S00}


# ------------------------------------------------------------
# 4) Evaluate 2x2 sets and compute metric deltas
# ------------------------------------------------------------
def compute_edge_metric_deltas(
    metrics_2x2: Dict[str, Dict[str, float]],
    *,
    metric_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Given metrics for 11/10/01/00, compute:
      - main effects: d_u, d_v
      - interaction: int
      - also store raw cell values
    Return:
      out[metric] = {
          "m11":..., "m10":..., "m01":..., "m00":...,
          "d_u":..., "d_v":..., "int":...
      }
    """
    out: Dict[str, Dict[str, float]] = {}

    for m in metric_names:
        m11 = float(metrics_2x2.get("11", {}).get(m, np.nan))
        m10 = float(metrics_2x2.get("10", {}).get(m, np.nan))
        m01 = float(metrics_2x2.get("01", {}).get(m, np.nan))
        m00 = float(metrics_2x2.get("00", {}).get(m, np.nan))

        # d_u: add u when v present => 11 - 01
        # d_v: add v when u present => 11 - 10
        d_u = m11 - m01
        d_v = m11 - m10

        # interaction term on this metric
        inter = m11 - m10 - m01 + m00

        out[m] = {
            "m11": m11, "m10": m10, "m01": m01, "m00": m00,
            "d_u": d_u, "d_v": d_v, "interaction": inter,
        }
    return out


# ------------------------------------------------------------
# 5) Main analysis driver
# ------------------------------------------------------------
def collect_backgrounds_from_iterlog(
    df_iter: pd.DataFrame,
    *,
    tmin: int,
    tmax: int,
    # Optional: if you want to carry over the already-evaluated metrics on the background (s00)
    # from iteration_log into the sweep table (avoids recomputing s00 for those samples).
    metric_names: Optional[List[str]] = None,
    # Sampling / size control (useful when you expand the window or have many traj)
    max_backgrounds: Optional[int] = None,
    dedup: bool = True,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Collect a *multi-background* set of base feature sets B from iteration_log for downstream edge sweeps.

    Why "multi-background"?
    - Any edge (u,v) effect is conditional on a background set B.
    - Using only one fixed background is brittle: strong "always-on" risk factors (e.g., agatston_score)
      can dominate the metrics, making other effects appear negligible.
    - Therefore we treat B as a random variable induced by the discovery process itself and
      estimate edge effects by averaging (or otherwise summarizing) across many B samples.

    What is a "background" here?
    - We reuse the sampled feature set at each (iteration, traj_id):
        B := current_set(iteration=t, run=r)
      This B reflects the current state of the learned sampling distribution (and evidence-gated trust),
      hence is a natural "in-distribution" background for post-hoc edge effect analysis.

    Output structure (DataFrame):
    - Each row is one background sample with:
        bg_id        : unique integer id for the background sample
        iteration    : generation index (same semantics as in iteration_log)
        traj_id      : run id
        bg_set       : Python `set[str]` background features (for fast set ops)
        bg_key       : canonical hashable key (tuple of sorted features) for deduping / grouping
        k_bg         : |B|
      Optional:
        <metric>_00  : if metric columns exist in df_iter, carry them as background metrics (s00)

    Notes:
    - This function does NOT assume that edge endpoints are absent/present in B.
      The downstream sweep function can decide whether to:
        * use union-based cells: m00=score(B), m10=score(B∪{u}), ...
        * or subtract endpoints from B first (removal-based).
      Here we only provide B samples.

    Parameters:
      tmin, tmax:
        Window of iterations to collect backgrounds from (inclusive).
      metric_names:
        If provided, and if df_iter already contains those metrics for the background sets,
        they will be copied into output columns named f"{metric}_00".
        If the metric columns are not present, they are silently skipped.
      max_backgrounds:
        If not None, randomly subsample backgrounds to this count (after filtering and optional dedup).
      dedup:
        If True, identical background sets across (t,r) will be collapsed to one row.
        This is useful when the sampler has partially converged and repeats the same set many times.
      seed:
        Random seed used only for optional subsampling.

    Returns:
      bg_df: pd.DataFrame as described above.
    """
    if df_iter is None or df_iter.empty:
        return pd.DataFrame(columns=["bg_id", "iteration", "traj_id", "bg_set", "bg_key", "k_bg"])

    if "iteration" not in df_iter.columns or "traj_id" not in df_iter.columns:
        raise KeyError("df_iter must contain columns: 'iteration' and 'traj_id'")
    if "current_set" not in df_iter.columns:
        raise KeyError("df_iter must contain parsed 'current_set' (set[str]); call normalize_iter_df first.")

    # -----------------------------
    # 1) Filter the iteration window
    # -----------------------------
    d = df_iter.copy()
    d = d[(d["iteration"] >= int(tmin)) & (d["iteration"] <= int(tmax))].reset_index(drop=True)

    if d.empty:
        return pd.DataFrame(columns=["bg_id", "iteration", "traj_id", "bg_set", "bg_key", "k_bg"])

    # -----------------------------
    # 2) Build canonical background representations
    # -----------------------------
    rows = []
    for rr in d.itertuples(index=False):
        B = rr.current_set
        if not isinstance(B, set):
            # be robust: allow list-like
            try:
                B = set(B)
            except Exception:
                B = set()

        # Canonical hashable key for dedup / grouping (stable order)
        bg_key = tuple(sorted([x for x in B if isinstance(x, str) and x.strip()]))

        row = {
            "iteration": int(rr.iteration),
            "traj_id": int(rr.traj_id),
            "bg_set": set(bg_key),   # store as set for fast set operations downstream
            "bg_key": bg_key,        # store as tuple for dedup/grouping
            "k_bg": int(len(bg_key)),
        }

        # Optionally carry over background metrics if they are already in iteration_log.
        # This allows using m00 from logs (s00) without re-running CV for those backgrounds.
        if metric_names:
            for m in metric_names:
                col = f"{m}_mean"
                if hasattr(rr, col):
                    v = getattr(rr, col)
                    # Save as "<metric>_00" to align with later 2x2 cell naming.
                    row[f"{m}_00"] = float(v) if (v is not None and np.isfinite(v)) else np.nan

        rows.append(row)

    bg_df = pd.DataFrame(rows)

    # -----------------------------
    # 3) Optional deduplication by bg_key
    # -----------------------------
    if dedup and not bg_df.empty:
        # Keep the earliest occurrence (small bias toward earlier in the window, which is fine).
        bg_df = bg_df.sort_values(["iteration", "traj_id"]).drop_duplicates(subset=["bg_key"]).reset_index(drop=True)

    # -----------------------------
    # 4) Optional subsampling (for large windows / many runs)
    # -----------------------------
    if max_backgrounds is not None and max_backgrounds > 0 and len(bg_df) > int(max_backgrounds):
        rng = np.random.default_rng(int(seed))
        keep_idx = rng.choice(len(bg_df), size=int(max_backgrounds), replace=False)
        bg_df = bg_df.iloc[np.sort(keep_idx)].reset_index(drop=True)

    # Assign bg_id
    bg_df.insert(0, "bg_id", np.arange(len(bg_df), dtype=int))

    return bg_df


# ============================================================
# Helpers for Step (E): union-based 2x2 on multiple backgrounds
# ============================================================

def make_2x2_sets_union(background: set, u: str, v: str) -> Dict[str, List[str]]:
    """
    Union-based 2x2 design (recommended for multi-background sweep):
      m00 = score(B)
      m10 = score(B ∪ {u})
      m01 = score(B ∪ {v})
      m11 = score(B ∪ {u, v})

    Notes:
    - This design is well-defined even if u or v is already in B, but then some cells become identical.
      To keep interpretation clean, Step (E) selects backgrounds that (ideally) exclude u and v.
    """
    B = set(background) if isinstance(background, set) else set(background)
    u = str(u); v = str(v)
    return {
        "00": sorted(B),
        "10": sorted(B | {u}),
        "01": sorted(B | {v}),
        "11": sorted(B | {u, v}),
    }


def select_background_rows_for_edge(
    bg_df: pd.DataFrame,
    *,
    u: str,
    v: str,
    # Minimum backgrounds required to compute a meaningful average effect
    min_bg_per_edge: int = 6,
    # Hard cap to control compute (optional)
    max_bg_per_edge: Optional[int] = None,
) -> pd.DataFrame:
    """
    Choose which background sets B to use for evaluating edge (u,v).

    Strategy:
      1) Prefer "clean" backgrounds where u∉B and v∉B, so that union cells are distinct.
      2) If not enough, relax to allow exactly one of {u,v} already in B.
      3) If still not enough, fall back to using all backgrounds.

    This keeps the interpretation of d_u, d_v, interaction closer to "marginal add effects",
    while still being robust when B frequently contains strong ubiquitous features.

    Returns: a filtered bg_df (same columns), possibly capped by max_bg_per_edge.
    """
    u = str(u); v = str(v)

    def _mask_none_in(B: set) -> bool:
        return (u not in B) and (v not in B)

    def _mask_one_in(B: set) -> bool:
        # exactly one of them is already present
        return ((u in B) ^ (v in B))

    # Ensure bg_set exists
    if "bg_set" not in bg_df.columns:
        raise KeyError("bg_df must have column 'bg_set' (set[str]) from collect_backgrounds_from_iterlog()")

    clean = bg_df[bg_df["bg_set"].apply(_mask_none_in)]
    if len(clean) >= min_bg_per_edge:
        out = clean
    else:
        relaxed = bg_df[bg_df["bg_set"].apply(lambda B: _mask_none_in(B) or _mask_one_in(B))]
        if len(relaxed) >= min_bg_per_edge:
            out = relaxed
        else:
            out = bg_df

    # Optional cap (for compute budget)
    if max_bg_per_edge is not None and len(out) > int(max_bg_per_edge):
        out = out.sort_values(["iteration", "traj_id"]).head(int(max_bg_per_edge))

    return out.reset_index(drop=True)


def _eval_feats_metrics(
    feats: List[str],
    *,
    ds,
    metric_names: List[str],
    CV_MODEL: str,
    eval_cache: Dict[Tuple[str, ...], Dict[str, float]],
) -> Dict[str, float]:
    """
    Evaluate CV metrics for a feature set and cache by a canonical tuple key.
    Caching is important because across edges/backgrounds, many sets repeat.

    Returns a dict {metric_name: float}.
    """
    key = tuple(sorted([f for f in feats if f in ds.pool_schema]))
    if key in eval_cache:
        return eval_cache[key]

    selected_schema = {f: ds.pool_schema[f] for f in key}
    metrics_summary, *_ = utils_models.evaluate_auc_and_more_cv(
        df=ds.df_all,
        label=ds.y_all,
        feature_schema=selected_schema,
        model_name=CV_MODEL,
    )
    out = {k: float(metrics_summary.get(f"{k}_mean", np.nan)) for k in metric_names}
    eval_cache[key] = out
    return out


def _get_bg_m00_from_bg_df_row(
    bg_row,
    *,
    metric_names: List[str],
) -> Optional[Dict[str, float]]:
    """
    If bg_df carries background metrics in columns like f"{metric}_00",
    we can reuse them as m00 and skip CV for cell '00'.
    Returns None if not available / missing.
    """
    m00 = {}
    ok = False
    for m in metric_names:
        col = f"{m}_00"
        if hasattr(bg_row, col):
            v = getattr(bg_row, col)
        else:
            v = None
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            m00[m] = np.nan
        else:
            m00[m] = float(v)
            ok = True
    return m00 if ok else None



def run_edge_interaction_effect_analysis(
    *,
    out_root: str,
    method: str,
    top_k_edges: int,
    metric_names: List[str],
    save_dir: str,
):
    # ------------------------------------------------------------
    # (A) Build bg_df
    # ------------------------------------------------------------
    df_iter_raw = path_agenda._load_iter_log(out_root, method)
    df_iter = path_agenda.normalize_iter_df(df_iter_raw)

    # The function signature may differ in your file; adjust here if needed.
    # We only need proto_df; if the function returns extra outputs, we ignore them.
    win_len = 50
    tmax = int(df_iter["iteration"].max())
    tmin = tmax - win_len + 1
    bg_df = collect_backgrounds_from_iterlog(
        df_iter=df_iter,
        tmin=tmin,
        tmax=tmax,
        metric_names=metric_names,
        max_backgrounds=None,
        dedup=True,
        seed=0,
    )

    # ------------------------------------------------------------
    # (B) Load node_df / edge_df by reusing plot_stable_kg.py
    # ------------------------------------------------------------
    kg_path = Path(os.path.join(out_root, method, "kg_snapshot.json"))
    kg = stable_kg.load_kg(kg_path)
    edge_df = stable_kg.parse_edge_df(kg)

    # ------------------------------------------------------------
    # (C) Select edges to analyze
    # ------------------------------------------------------------
    sel_edges = select_top_edges(
        edge_df,
        u_col="u",
        v_col="v",
        top_k=top_k_edges,
    )
    sel_edges.to_csv(os.path.join(save_dir, "selected_edges.csv"), index=False)

    # ------------------------------------------------------------
    # (D) Load dataset
    # ------------------------------------------------------------
    ds = _load_saved_dataset_ctx(out_root, method)

    # ============================================================
    # (E) Evaluate each edge’s 2x2 sets over multiple backgrounds
    # ============================================================
    rows = []

    # Cache CV evaluations across (edge, background) to reduce compute
    eval_cache_sets: Dict[Tuple[str, ...], Dict[str, float]] = {}

    # You can tune these two knobs:
    min_bg_per_edge = 20       # if too low -> noisy; if too high -> may drop too many edges
    max_bg_per_edge = 80       # set e.g. 80 if compute becomes heavy

    for rr in tqdm(sel_edges.itertuples(index=False), total=len(sel_edges), desc="Evaluating edge effects (multi-bg union)"):
        u, v = str(rr.u), str(rr.v)

        # 1) Select which background sets to use for this edge
        bg_sub = select_background_rows_for_edge(
            bg_df,
            u=u, v=v,
            min_bg_per_edge=min_bg_per_edge,
            max_bg_per_edge=max_bg_per_edge,
        )

        # 2) For each background, compute 2x2 metrics, then deltas; collect per-bg deltas for averaging
        per_bg_deltas = []  # list of dict(metric -> dict cells/d_u/d_v/int), same format as compute_edge_metric_deltas output
        per_bg_sizes = []
        per_bg_ids = []

        for bg_row in bg_sub.itertuples(index=False):
            B = bg_row.bg_set
            per_bg_sizes.append(int(len(B)))
            per_bg_ids.append(int(bg_row.bg_id) if hasattr(bg_row, "bg_id") else -1)

            sets_2x2 = make_2x2_sets_union(B, u, v)

            # Build metrics_2x2[cell][metric]
            metrics_2x2: Dict[str, Dict[str, float]] = {}

            # --- cell 00: try reuse from bg_df (if available), else CV ---
            m00_reuse = _get_bg_m00_from_bg_df_row(bg_row, metric_names=metric_names)
            if m00_reuse is not None:
                metrics_2x2["00"] = m00_reuse
            else:
                metrics_2x2["00"] = _eval_feats_metrics(
                    sets_2x2["00"],
                    ds=ds,
                    metric_names=metric_names,
                    CV_MODEL=CV_MODEL,
                    eval_cache=eval_cache_sets,
                )

            # --- cells 10/01/11: must evaluate (union changes the set) ---
            for cell in ["10", "01", "11"]:
                metrics_2x2[cell] = _eval_feats_metrics(
                    sets_2x2[cell],
                    ds=ds,
                    metric_names=metric_names,
                    CV_MODEL=CV_MODEL,
                    eval_cache=eval_cache_sets,
                )

            # Compute deltas for this background
            deltas_bg = compute_edge_metric_deltas(metrics_2x2, metric_names=metric_names)
            per_bg_deltas.append(deltas_bg)

        # 3) Aggregate across backgrounds: mean of each quantity (m00/m10/m01/m11/d_u/d_v/interaction)
        #    You can extend to std/percentiles if you want later.
        base = {
            "u": u, "v": v,
            "delta_mean": float(rr.delta_mean),
            "delta_conf": float(rr.delta_conf),
            "sign_group": str(rr.sign_group),
            "top_k_edges": int(top_k_edges),
            "n_bg_used": int(len(per_bg_deltas)),
            "bg_size_mean": float(np.mean(per_bg_sizes)) if per_bg_sizes else np.nan,
            "bg_size_std": float(np.std(per_bg_sizes)) if per_bg_sizes else np.nan,
            "tmin": int(tmin),
            "tmax": int(tmax),
        }

        # Aggregate per metric
        for m in metric_names:
            # collect per-bg values
            m11 = [d[m]["m11"] for d in per_bg_deltas]
            m10 = [d[m]["m10"] for d in per_bg_deltas]
            m01 = [d[m]["m01"] for d in per_bg_deltas]
            m00 = [d[m]["m00"] for d in per_bg_deltas]
            du  = [d[m]["d_u"] for d in per_bg_deltas]
            dv  = [d[m]["d_v"] for d in per_bg_deltas]
            itv = [d[m]["interaction"] for d in per_bg_deltas]

            # mean aggregation (paper-friendly “expected effect under discovered backgrounds”)
            base[f"{m}_11"] = float(np.nanmean(m11))
            base[f"{m}_10"] = float(np.nanmean(m10))
            base[f"{m}_01"] = float(np.nanmean(m01))
            base[f"{m}_00"] = float(np.nanmean(m00))
            base[f"{m}_d_u"] = float(np.nanmean(du))
            base[f"{m}_d_v"] = float(np.nanmean(dv))
            base[f"{m}_int"] = float(np.nanmean(itv))

            # Optional: add dispersion for uncertainty visualization later
            base[f"{m}_int_std"] = float(np.nanstd(itv))
            base[f"{m}_int_p25"] = float(np.nanpercentile(itv, 25))
            base[f"{m}_int_p50"] = float(np.nanpercentile(itv, 50))
            base[f"{m}_int_p75"] = float(np.nanpercentile(itv, 75))

        rows.append(base)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(save_dir, f"edge_effect_table_{method}.csv"), index=False)
    print("Saved:", os.path.join(save_dir, f"edge_effect_table_{method}.csv"))


def _discover_metrics_from_edge_effect_table(df: pd.DataFrame) -> list[str]:
    """
    Infer base metric names from columns like:
      auc_mean_11, auc_mean_10, ..., auc_mean_int
    Returns sorted unique metric base names, e.g. ["acc_mean", "auc_mean", ...]
    """
    pat = re.compile(r"^(.*)_(11|10|01|00|d_u|d_v|int)$")
    metrics = set()
    for c in df.columns:
        m = pat.match(str(c))
        if m:
            metrics.add(m.group(1))
    return sorted(metrics)


def plot_edge_effect_table(
    edge_table_csv: str,
    *,
    save_dir: str,
    metrics: list[str] | None = None,
    # 用哪个 “int” 作为主展示（MICCAI通常放 AUC/ACC/敏感性即可）
    primary_metric: str = "auc",
    # 若你想看 delta_mean “metric_int” 的关系
    scatter_x: str = "delta_mean",
):
    """
    Post-hoc analysis for edge interaction effects using edge_effect_table.csv.

    The input table is assumed to contain columns:
      u, v, delta_mean, delta_conf, sign_group, bg_size, bg_strategy, ...
    and for each metric M:
      M_11, M_10, M_01, M_00, M_d_u, M_d_v, M_int

    What we compute (per metric):
      - Distribution of M_int stratified by sign_group (positive/negative)
      - Distribution of main effects M_d_u, M_d_v
      - Interaction share: |M_int| / (|M_d_u| + |M_d_v| + eps)

    What we plot (compact, paper-friendly):
      1) For each metric: boxplot of M_int (positive vs negative)
      2) Interaction-share boxplot (positive vs negative)
      3) Scatter: x=scatter_x (delta_mean), y=primary_metric_int
         color by sign_group
    """
    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(edge_table_csv)
    if df.empty:
        raise ValueError(f"Empty table: {edge_table_csv}")

    # Infer metrics
    if metrics is None:
        metrics = _discover_metrics_from_edge_effect_table(df)

    # Sanity check: primary metric exists
    if primary_metric not in metrics:
        raise KeyError(
            f"primary_metric='{primary_metric}' not found. "
            f"Available metrics={metrics}"
        )

    # -----------------------
    # 1) Build a long table
    # -----------------------
    long_rows = []
    eps = 1e-12

    for _, r in df.iterrows():
        sign = r.get("sign_group", "unknown")
        u = r.get("u", "")
        v = r.get("v", "")
        for m in metrics:
            row = {
                "u": u,
                "v": v,
                "edge": f"{u}--{v}",
                "sign_group": sign,
                "delta_mean": float(r.get("delta_mean", np.nan)),
                "delta_conf": float(r.get("delta_conf", np.nan)),
                "metric": m,
                "m11": float(r.get(f"{m}_11", np.nan)),
                "m10": float(r.get(f"{m}_10", np.nan)),
                "m01": float(r.get(f"{m}_01", np.nan)),
                "m00": float(r.get(f"{m}_00", np.nan)),
                "d_u": float(r.get(f"{m}_d_u", np.nan)),
                "d_v": float(r.get(f"{m}_d_v", np.nan)),
                "interaction": float(r.get(f"{m}_int", np.nan)),
            }
            # “交互占比”：用来判断是不是“主要靠交互”在起作用
            denom = abs(row["d_u"]) + abs(row["d_v"]) + eps
            row["interaction_share"] = abs(row["interaction"]) / denom
            long_rows.append(row)

    dfl = pd.DataFrame(long_rows)

    # -----------------------
    # 2) Summary tables
    # -----------------------
    # (a) per metric & sign: mean/median of interaction and share
    summary = (
        dfl.groupby(["metric", "sign_group"], as_index=False)
           .agg(
               n_edges=("edge", "nunique"),
               int_mean=("interaction", "mean"),
               int_median=("interaction", "median"),
               int_p10=("interaction", lambda x: float(np.nanpercentile(x, 10))),
               int_p90=("interaction", lambda x: float(np.nanpercentile(x, 90))),
               share_mean=("interaction_share", "mean"),
               share_median=("interaction_share", "median"),
           )
    )
    summary_path = os.path.join(save_dir, "edge_effect_summary_by_metric_sign.csv")
    summary.to_csv(summary_path, index=False)

    # (b) per edge: primary metric interpretation friendly view
    primary_view = (
        dfl[dfl["metric"] == primary_metric]
          .copy()
          .sort_values("interaction", ascending=True)
    )
    primary_path = os.path.join(save_dir, f"edge_effect_primary_{primary_metric}.csv")
    primary_view.to_csv(primary_path, index=False)

    # -----------------------
    # 3) Plots (compact) MICCAI 论文使用
    # -----------------------
    # 3.1 Boxplot of interaction by metric (positive vs negative)
    # Make grid: one row per metric, 2 boxes (pos/neg).
    metrics_plot = metrics
    n = len(metrics_plot)
    fig_h = 0.3 * n
    fig_w = 2.3

    fig, ax = plt.subplots(1, 1, figsize=(2.2, 1.5))
    # Prepare data: for each metric, collect pos/neg arrays
    positions = []
    data = []
    labels = []
    y = 0
    offset = 0.2
    for m in metrics_plot:
        for sg, dy in [("negative", -offset), ("positive", offset)]:
            vals = dfl[(dfl["metric"] == m) & (dfl["sign_group"] == sg)]["interaction"].dropna().values
            data.append(vals)
            positions.append(y + dy)
            labels.append(f"{m}")
        y += 1

    color_map = {"positive": "tab:orange", "negative": "tab:blue"}
    box_colors = []
    for _m in metrics_plot:
        for _sg in ["negative", "positive"]:
            box_colors.append(color_map[_sg])

    bp = ax.boxplot(
        data,
        vert=False,
        positions=positions,
        widths=0.3,
        showfliers=False,
        patch_artist=True,
    )
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor("none")
        patch.set_alpha(1.0)
        patch.set_edgecolor(c)
        patch.set_linewidth(0.5)
    for median, c in zip(bp.get("medians", []), box_colors):
        median.set_color(c)
        median.set_linewidth(0.5)
    for i, (whisk, cap) in enumerate(zip(bp.get("whiskers", []), bp.get("caps", []))):
        c = box_colors[i // 2]
        whisk.set_color(c)
        whisk.set_linewidth(0.5)
        cap.set_color(c)
        cap.set_linewidth(0.5)
    # legend_handles = [
    #     Patch(facecolor=color_map["negative"], edgecolor="none", label="negative", alpha=1.0),
    #     Patch(facecolor=color_map["positive"], edgecolor="none", label="positive", alpha=1.0),
    # ]
    legend_handles = [
        Line2D([0], [0], color=color_map["negative"], lw=1, label="negative"),
        Line2D([0], [0], color=color_map["positive"], lw=1, label="positive"),
    ]
    leg = ax.legend(handles=legend_handles, loc="best", fontsize = 5, frameon=True, edgecolor="lightgray", framealpha=0.4)
    leg.get_frame().set_linewidth(0.5)
    ax.set_yticks(range(len(metrics_plot)))
    ax.set_yticklabels(metrics_plot)
    ax.set_xlabel("EGL-LLM metric interaction")
    # ax.set_title("Edge interaction effect on CV metrics")
    ax.grid(True, axis="x", alpha=0.2)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "box_metric_int_by_sign.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 3.2 Interaction-share boxplot (per metric, pos/neg)
    fig, ax = plt.subplots(1, 1, figsize=(2.2, 1.5))
    positions, data, labels = [], [], []
    y = 0
    offset = 0.2
    for m in metrics_plot:
        for sg, dy in [("negative", -offset), ("positive", offset)]:
            vals = dfl[(dfl["metric"] == m) & (dfl["sign_group"] == sg)]["interaction_share"].dropna().values
            data.append(vals)
            positions.append(y + dy)
            labels.append(f"{m}")
        y += 1

    bp = ax.boxplot(
        data,
        vert=False,
        positions=positions,
        widths=0.3,
        showfliers=False,
        patch_artist=True,
    )
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor("none")
        patch.set_alpha(1.0)
        patch.set_edgecolor(c)
        patch.set_linewidth(0.5)
    for median, c in zip(bp.get("medians", []), box_colors):
        median.set_color(c)
        median.set_linewidth(0.5)
    for i, (whisk, cap) in enumerate(zip(bp.get("whiskers", []), bp.get("caps", []))):
        c = box_colors[i // 2]
        whisk.set_color(c)
        whisk.set_linewidth(0.5)
        cap.set_color(c)
        cap.set_linewidth(0.5)
    leg = ax.legend(handles=legend_handles, loc="best", fontsize = 5, frameon=True, edgecolor="lightgray", framealpha=0.4)
    leg.get_frame().set_linewidth(0.5)
    ax.set_yticks(range(len(metrics_plot)))
    ax.set_yticklabels(metrics_plot)
    # ax.set_xlabel("abs(metric_interaction) / (abs(d_u)+abs(d_v))")
    ax.set_xlabel("Interaction share")
    # ax.set_title("How much of the gain is truly 'interaction' vs main effects (share)")
    ax.grid(True, axis="x", alpha=0.2)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "box_interaction_share_by_sign.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 3.3 Scatter: x=scatter_x, y=primary_metric_int
    fig, ax = plt.subplots(1, 1, figsize=(2.3, 2.3))
    dfp = primary_view.copy()
    # Separate by sign_group to keep legend clean
    for sg in ["negative", "positive"]:
        sub = dfp[dfp["sign_group"] == sg]
        if sub.empty:
            continue
        ax.scatter(sub[scatter_x], sub["interaction"], s=3, label=sg, alpha=1.0)

    ax.set_xlabel(scatter_x)
    ax.set_ylabel(f"{primary_metric}_int")
    ax.set_title(f"Edge-level interaction: {primary_metric} interaction vs {scatter_x}")
    ax.grid(True, alpha=0.2)
    leg = ax.legend(loc="lower right", fontsize = 5, frameon=True, edgecolor="lightgray", framealpha=1.0)
    leg.get_frame().set_linewidth(0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"scatter_{primary_metric}_int_vs_{scatter_x}.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 3.4 Bar: top1 negative vs top1 positive across all metrics (metric_int)
    df_neg = df[df.get("sign_group", "") == "negative"]
    df_pos = df[df.get("sign_group", "") == "positive"]
    if (not df_neg.empty) or (not df_pos.empty):
        top_neg = df_neg.sort_values("delta_mean", ascending=True).iloc[0] if not df_neg.empty else None
        top_pos = df_pos.sort_values("delta_mean", ascending=False).iloc[0] if not df_pos.empty else None

        metrics_vals_neg = [float(top_neg.get(f"{m}_int", np.nan)) for m in metrics_plot] if top_neg is not None else [np.nan] * len(metrics_plot)
        metrics_vals_pos = [float(top_pos.get(f"{m}_int", np.nan)) for m in metrics_plot] if top_pos is not None else [np.nan] * len(metrics_plot)

        # Sort metrics by negative values (ascending) if available; otherwise by positive.
        if top_neg is not None:
            order = sorted(range(len(metrics_plot)), key=lambda i: metrics_vals_neg[i])
        else:
            order = sorted(range(len(metrics_plot)), key=lambda i: metrics_vals_pos[i])
        metrics_plot = [metrics_plot[i] for i in order]
        metrics_vals_neg = [metrics_vals_neg[i] for i in order]
        metrics_vals_pos = [metrics_vals_pos[i] for i in order]

        y = np.arange(len(metrics_plot))
        h = 0.28

        fig, ax = plt.subplots(1, 1, figsize=(2.2, 1.5))
        if top_neg is not None:
            u_lab = re.sub(r"^diag_", "", str(top_neg.get("u")))
            v_lab = re.sub(r"^diag_", "", str(top_neg.get("v")))
            bars_neg = ax.barh(y - h/2, metrics_vals_neg, height=h, label=f"{u_lab}—{v_lab}")
            # bars_neg = ax.barh(y - h/2, metrics_vals_neg, height=h, color="#068863", label="Top 1 negative")

        if top_pos is not None:
            u_lab = re.sub(r"^diag_", "", str(top_pos.get("u")))
            v_lab = re.sub(r"^diag_", "", str(top_pos.get("v")))
            bars_pos = ax.barh(y + h/2, metrics_vals_pos, height=h, label=f"{u_lab}—{v_lab}")
            # bars_pos = ax.barh(y + h/2, metrics_vals_pos, height=h, color="#EA4518", label="Top 1 positive")

        ax.set_yticks(y)
        ax.set_yticklabels(metrics_plot, rotation=0)
        ax.set_xlabel("Metric interaction")
        # ax.set_title("Top1 negative vs positive edges across metrics")
        ax.grid(True, axis="x", linewidth=0.5, alpha=0.5)
        # Stable legend placement: use figure coordinates + fixed bottom margin.
        leg = ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 0.02),
            bbox_transform=fig.transFigure,
            ncol=2,
            fontsize=5,
            frameon=True,
            edgecolor="lightgray",
            framealpha=0.8,
        )
        leg.get_frame().set_linewidth(0.5)

        # Reduce x-axis ticks to avoid crowding and keep 2-decimal labels.
        # x_min_tmp, x_max_tmp = ax.get_xlim()
        # ax.set_xticks(np.linspace(x_min_tmp, x_max_tmp, 3))
        # ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

        # Debug-only: one-sample t-test vs 0 using summary stats.
        def _ttest_1samp_from_stats(mean: float, std: float, n: float) -> Tuple[float, float, float]:
            if mean is None or std is None or n is None:
                return (np.nan, np.nan, np.nan)
            if np.isnan(mean) or np.isnan(std) or np.isnan(n) or n <= 1 or std == 0:
                return (np.nan, np.nan, np.nan)
            t = mean / (std / math.sqrt(n))
            df = n - 1
            try:
                from scipy import stats  # optional
                p = 2 * stats.t.sf(abs(t), df)
            except Exception:
                # Normal approximation as fallback if scipy is unavailable.
                z = abs(t)
                cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
                p = 2 * (1.0 - cdf)
            return (t, p, df)

        n_neg = float(top_neg.get("n_bg_used", np.nan)) if top_neg is not None else np.nan
        n_pos = float(top_pos.get("n_bg_used", np.nan)) if top_pos is not None else np.nan
        ttest_neg = {}
        ttest_pos = {}
        for m in metrics_plot:
            mean_neg = float(top_neg.get(f"{m}_int", np.nan)) if top_neg is not None else np.nan
            std_neg = float(top_neg.get(f"{m}_int_std", np.nan)) if top_neg is not None else np.nan
            ttest_neg[m] = _ttest_1samp_from_stats(mean_neg, std_neg, n_neg)
            mean_pos = float(top_pos.get(f"{m}_int", np.nan)) if top_pos is not None else np.nan
            std_pos = float(top_pos.get(f"{m}_int_std", np.nan)) if top_pos is not None else np.nan
            ttest_pos[m] = _ttest_1samp_from_stats(mean_pos, std_pos, n_pos)

        def _fdr_bh(pvals: List[float]) -> List[float]:
            p = np.asarray(pvals, dtype=float)
            n = p.size
            adj = np.full(n, np.nan)
            valid = np.isfinite(p)
            if not np.any(valid):
                return adj.tolist()
            p_valid = p[valid]
            order = np.argsort(p_valid)
            ranked = p_valid[order]
            m = ranked.size
            q = ranked * m / (np.arange(1, m + 1))
            # enforce monotonicity
            q = np.minimum.accumulate(q[::-1])[::-1]
            q = np.clip(q, 0.0, 1.0)
            adj_valid = np.empty_like(q)
            adj_valid[order] = q
            adj[valid] = adj_valid
            return adj.tolist()

        pvals_neg = [ttest_neg[m][1] for m in metrics_plot]
        pvals_pos = [ttest_pos[m][1] for m in metrics_plot]
        pvals_neg_fdr = _fdr_bh(pvals_neg)
        pvals_pos_fdr = _fdr_bh(pvals_pos)
        ttest_neg_fdr = {
            m: (ttest_neg[m][0], ttest_neg[m][1], ttest_neg[m][2], pvals_neg_fdr[i])
            for i, m in enumerate(metrics_plot)
        }
        ttest_pos_fdr = {
            m: (ttest_pos[m][0], ttest_pos[m][1], ttest_pos[m][2], pvals_pos_fdr[i])
            for i, m in enumerate(metrics_plot)
        }

        def _p_to_stars(p: float) -> str:
            if p is None or np.isnan(p):
                return ""
            if p < 0.001:
                return "***"
            if p < 0.01:
                return "**"
            if p < 0.05:
                return "*"
            return ""

        # Add significance stars near bar ends (left for negative, right for positive).
        all_vals = [v for v in (metrics_vals_neg + metrics_vals_pos) if np.isfinite(v)]
        span = max(all_vals) - min(all_vals) if all_vals else 0.0
        x_pad = 0.01 * span if span > 0 else 0.01

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = ax.transData.inverted()

        def _text_width_data(s: str, fontsize: float) -> float:
            t = ax.text(0, 0, s, fontsize=fontsize, alpha=0.0)
            bb = t.get_window_extent(renderer=renderer)
            t.remove()
            x0 = inv.transform((0, 0))[0]
            x1 = inv.transform((bb.width, 0))[0]
            return x1 - x0

        x_min, x_max = ax.get_xlim()
        for i, m in enumerate(metrics_plot):
            p_fdr_neg = ttest_neg_fdr[m][3]
            stars_neg = _p_to_stars(p_fdr_neg)
            if stars_neg:
                bar = bars_neg[i]
                x = metrics_vals_neg[i]
                y_center = bar.get_y() + bar.get_height() / 2 - 0.15 # 调整星号的位置
                text_w = _text_width_data(stars_neg, fontsize=5)
                x_star = x - x_pad
                ha = "right"
                if x_star - text_w < x_min+0.1:
                    x_star = x + x_pad
                    ha = "left"
                ax.text(
                    x_star,
                    y_center,
                    stars_neg,
                    va="center",
                    ha=ha,
                    fontsize=5,
                )

            p_fdr_pos = ttest_pos_fdr[m][3]
            stars_pos = _p_to_stars(p_fdr_pos)
            if stars_pos:
                bar = bars_pos[i]
                x = metrics_vals_pos[i]
                y_center = bar.get_y() + bar.get_height() / 2 - 0.15 # 调整星号的位置
                text_w = _text_width_data(stars_pos, fontsize=5)
                x_star = x + x_pad
                ha = "left"
                if x_star + text_w > x_max-0.1:
                    x_star = x - x_pad
                    ha = "right"
                ax.text(
                    x_star,
                    y_center,
                    stars_pos,
                    va="center",
                    ha=ha,
                    fontsize=5,
                )

        fig.subplots_adjust(bottom=0.28)
        fig.savefig(os.path.join(save_dir, "bar_top1_pos_neg_metric_int.png"),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

    print("Saved:")
    print(" -", summary_path)
    print(" -", primary_path)
    print(" - plots in", save_dir)

    return summary, primary_view, dfl


if __name__ == "__main__":
    from plot_all_from_config import main
    main(default_sections=["edge_interaction"])

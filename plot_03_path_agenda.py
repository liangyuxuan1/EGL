#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Path-difference + Agenda-execution analysis for S04/S05 runs
===========================================================

This script compares:
  (A) Baseline vs LLM exploration *paths* (what feature-sets and probes were visited over time)
  (B) For LLM only: agenda execution rate (did must/prefer/exclude/frontier actually take effect?)

Inputs:
  out_root/
    baseline_iters200/iteration_log.csv
    Qwen_iters200/iteration_log.csv
    Qwen_iters200/control_log.csv

Key columns in iteration_log.csv:
  - iteration (int)
  - traj_id (run id)
  - current_set_json (json string of list[str])
  - node_probes (json string of list[dict], each has {"var": <feature>, ...})
  - edge_probes (json string of list[dict], each has {"u": <feat>, "v": <feat>, ...})

Key columns in control_log.csv (LLM only):
  - generation (== iteration)
  - must_include (json string of list[str])
  - prefer_include (json string of list[str])
  - prefer_exclude (json string of list[str])
  - frontier_edges (json string of list[[u,v], ...])

--------------------------------------------------------------------
What is a "path" here?
--------------------------------------------------------------------
We treat each trajectory (traj_id) as a sequence of visited sets:
  S_{t,r} = current_set(t, run=r)

Then "path difference" is quantified by:
  - within-method coherence: how consistent runs are at the same iteration (mean pairwise Jaccard)
  - cross-method divergence: how different baseline vs LLM are at the same iteration (mean Jaccard across methods)
  - dynamical change: how much the set changes over time within each run (flip-like measures)

Additionally, we analyze "evidence sampling path" via probes:
  - Node probes: P_node_{t,r} = set of vars probed at iteration t
  - Edge probes: P_edge_{t,r} = set of undirected edges probed at iteration t

--------------------------------------------------------------------
Agenda execution (LLM only)
--------------------------------------------------------------------
Agenda is a controller output. Execution is checked at three layers:

  (1) Base-set compliance: agenda items are present in current_set S_{t,r}
      - hit_must_cur, hit_prefer_cur, viol_excl_cur, hit_frontier_end_cur

  (2) Probe compliance: agenda items are actually probed (in node_probes / edge_probes)
      - hit_must_nodeprobe, hit_prefer_nodeprobe, viol_excl_nodeprobe
      - frontier feasibility under strict reuse: endpoints must be in node_probes
      - frontier edge execution: frontier_edges ∩ edge_probes

  (3) Edge budget health: does edge_probes count reach P_EDGE?
      - edge_probe_count_shortfall = P_EDGE - |edge_probes| (averaged across runs)

These metrics explain *why* agenda seems ineffective:
  - If must/prefer rarely appear in current_set, the controller cannot be realized without soft shaping.
  - If frontier endpoints are not in node_probes, frontier edges are infeasible under strict reuse.
"""

import os
import math
import json
import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils_plot import set_paper_style
set_paper_style()


# ---------------------------
# Helpers: IO + JSON parsing
# ---------------------------
def _load_iter_log(out_root: str, method: str) -> pd.DataFrame:
    path = os.path.join(out_root, method, "iteration_log.csv")
    df = pd.read_csv(path)

    if "iteration" not in df.columns:
        raise KeyError(f"'iteration' column not found in {path}. Columns={list(df.columns)[:30]}")

    if "traj_id" not in df.columns:
        raise KeyError(f"'traj_id' column not found in {path}. Columns={list(df.columns)[:30]}")

    df["_source_path"] = path
    return df


def _load_control_log(out_root: str, llm_method: str) -> pd.DataFrame:
    path = os.path.join(out_root, llm_method, "control_log.csv")
    df = pd.read_csv(path)
    # generation == iteration semantics
    if "generation" not in df.columns:
        raise KeyError(f"'generation' column not found in {path}. Columns={list(df.columns)[:30]}")
    df["_source_path"] = path
    return df


def _safe_json_loads(x: Any, default: Any):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return default
    if isinstance(x, (list, dict)):
        return x
    if not isinstance(x, str):
        return default
    s = x.strip()
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default


def _as_set_list(x: Any) -> List[str]:
    """Parse json list[str] and normalize to list[str]."""
    arr = _safe_json_loads(x, default=[])
    if not isinstance(arr, list):
        return []
    out = []
    for v in arr:
        if isinstance(v, str) and v.strip():
            out.append(v.strip())
    return out


def _parse_node_probes(x: Any) -> List[str]:
    """Parse node_probes json: list of dicts; extract 'var'."""
    arr = _safe_json_loads(x, default=[])
    if not isinstance(arr, list):
        return []
    out = []
    for it in arr:
        if isinstance(it, dict):
            v = it.get("var", None)
            if isinstance(v, str) and v.strip():
                out.append(v.strip())
    return out


def _edge_key(u: str, v: str) -> Tuple[str, str]:
    if u <= v:
        return (u, v)
    return (v, u)


def _parse_edge_probes(x: Any) -> List[Tuple[str, str]]:
    """Parse edge_probes json: list of dicts; extract (u,v) as undirected key."""
    arr = _safe_json_loads(x, default=[])
    if not isinstance(arr, list):
        return []
    out = []
    for it in arr:
        if isinstance(it, dict):
            u = it.get("u", None)
            v = it.get("v", None)
            if isinstance(u, str) and isinstance(v, str) and u.strip() and v.strip() and u != v:
                out.append(_edge_key(u.strip(), v.strip()))
    return out


def _parse_frontier_edges(x: Any) -> List[Tuple[str, str]]:
    """Parse frontier_edges json: list[[u,v], ...] and normalize to undirected keys."""
    arr = _safe_json_loads(x, default=[])
    if not isinstance(arr, list):
        return []
    out = []
    for e in arr:
        if isinstance(e, (list, tuple)) and len(e) == 2:
            u, v = e[0], e[1]
            if isinstance(u, str) and isinstance(v, str) and u.strip() and v.strip() and u != v:
                out.append(_edge_key(u.strip(), v.strip()))
    return out


def _frontier_endpoints(frontier_edges: List[Tuple[str, str]]) -> List[str]:
    end = []
    for (u, v) in frontier_edges:
        end.append(u)
        end.append(v)
    # unique keep order
    seen = set()
    out = []
    for x in end:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    return float(inter / max(1, uni))


def mean_pairwise_jaccard(sets: List[set]) -> float:
    """Mean pairwise Jaccard across a list of sets. If <2 sets, return 1.0."""
    if len(sets) <= 1:
        return 1.0
    vals = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            vals.append(jaccard(sets[i], sets[j]))
    return float(np.mean(vals)) if vals else 1.0


# ---------------------------
# Build normalized tables
# ---------------------------
def normalize_iter_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Expand json-string columns into parsed list/set-friendly columns.
    Returns a row-per-(iteration,traj_id) dataframe.
    """
    df = df_raw.copy()

    if "current_set_json" not in df.columns:
        raise KeyError("iteration_log missing 'current_set_json' column")
    if "node_probes" not in df.columns:
        raise KeyError("iteration_log missing 'node_probes' column")
    if "edge_probes" not in df.columns:
        raise KeyError("iteration_log missing 'edge_probes' column")

    df["current_set_list"] = df["current_set_json"].apply(_as_set_list)
    df["node_probe_list"] = df["node_probes"].apply(_parse_node_probes)
    df["edge_probe_list"] = df["edge_probes"].apply(_parse_edge_probes)

    # Convert to python sets (fast for intersections)
    df["current_set"] = df["current_set_list"].apply(lambda x: set(x))
    df["node_probe_set"] = df["node_probe_list"].apply(lambda x: set(x))
    df["edge_probe_set"] = df["edge_probe_list"].apply(lambda x: set(x))

    # Basic sizes (useful for sanity + plotting)
    df["k_current"] = df["current_set"].apply(len)
    df["k_nodeprobe"] = df["node_probe_set"].apply(len)
    df["k_edgeprobe"] = df["edge_probe_set"].apply(len)

    return df


def normalize_control_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Parse agenda fields into list/set columns.
    Row-per-generation (==iteration).
    """
    df = df_raw.copy()

    for c in ["must_include", "prefer_include", "prefer_exclude", "frontier_edges"]:
        if c not in df.columns:
            # allow missing; fill empty
            df[c] = "[]"

    df["must_list"] = df["must_include"].apply(_as_set_list)
    df["prefer_list"] = df["prefer_include"].apply(_as_set_list)
    df["excl_list"] = df["prefer_exclude"].apply(_as_set_list)
    df["frontier_list"] = df["frontier_edges"].apply(_parse_frontier_edges)

    df["must_set"] = df["must_list"].apply(set)
    df["prefer_set"] = df["prefer_list"].apply(set)
    df["excl_set"] = df["excl_list"].apply(set)
    df["frontier_set"] = df["frontier_list"].apply(set)

    df["frontier_end_list"] = df["frontier_list"].apply(_frontier_endpoints)
    df["frontier_end_set"] = df["frontier_end_list"].apply(set)

    # sizes
    df["k_must"] = df["must_set"].apply(len)
    df["k_prefer"] = df["prefer_set"].apply(len)
    df["k_excl"] = df["excl_set"].apply(len)
    df["k_frontier"] = df["frontier_set"].apply(len)
    df["k_fend"] = df["frontier_end_set"].apply(len)

    return df.sort_values("generation").reset_index(drop=True)


# ---------------------------
# A) Path difference metrics
# ---------------------------
def compute_path_metrics(dfA: pd.DataFrame, dfB: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-iteration path difference metrics.

    Metrics:
      - within_A_jaccard_cur: mean pairwise Jaccard among runs' current_set at same iteration
      - within_B_jaccard_cur: same for method B
      - cross_AB_jaccard_cur: mean Jaccard between method A and B at same iteration (paired by traj_id if possible)
      - within_A_jaccard_nodeprobe / edgeprobe: same for probes
      - within_B_jaccard_nodeprobe / edgeprobe
      - cross_AB_jaccard_nodeprobe / edgeprobe
      - mean_k_current_A/B, mean_k_nodeprobe_A/B, mean_k_edgeprobe_A/B
      - flip-like dynamics within each method:
          mean_delta_add_A/B, mean_delta_remove_A/B, mean_delta_sym_A/B
        where delta is computed per run between iteration t-1 -> t, then averaged at t.
    """
    iters = sorted(set(dfA["iteration"].unique()) | set(dfB["iteration"].unique()))
    trajA = sorted(dfA["traj_id"].unique())
    trajB = sorted(dfB["traj_id"].unique())
    traj_inter = sorted(set(trajA) & set(trajB))

    rows = []

    # Pre-index for quick lookup: (iter,traj)->row
    keyA = {(int(r.iteration), int(r.traj_id)): r for r in dfA.itertuples(index=False)}
    keyB = {(int(r.iteration), int(r.traj_id)): r for r in dfB.itertuples(index=False)}

    for t in iters:
        # gather sets across runs
        SA = []
        PA_n = []
        PA_e = []
        for r in trajA:
            rr = keyA.get((int(t), int(r)), None)
            if rr is None:
                continue
            SA.append(rr.current_set)
            PA_n.append(rr.node_probe_set)
            PA_e.append(rr.edge_probe_set)

        SB = []
        PB_n = []
        PB_e = []
        for r in trajB:
            rr = keyB.get((int(t), int(r)), None)
            if rr is None:
                continue
            SB.append(rr.current_set)
            PB_n.append(rr.node_probe_set)
            PB_e.append(rr.edge_probe_set)

        within_A_cur = mean_pairwise_jaccard(SA)
        within_B_cur = mean_pairwise_jaccard(SB)
        within_A_n = mean_pairwise_jaccard(PA_n)
        within_B_n = mean_pairwise_jaccard(PB_n)
        within_A_e = mean_pairwise_jaccard(PA_e)
        within_B_e = mean_pairwise_jaccard(PB_e)

        # cross-method jaccard: if shared traj_ids exist, compare same traj_id; else full cross average
        cross_cur_vals = []
        cross_n_vals = []
        cross_e_vals = []

        if len(traj_inter) > 0:
            for r in traj_inter:
                ra = keyA.get((int(t), int(r)), None)
                rb = keyB.get((int(t), int(r)), None)
                if ra is None or rb is None:
                    continue
                cross_cur_vals.append(jaccard(ra.current_set, rb.current_set))
                cross_n_vals.append(jaccard(ra.node_probe_set, rb.node_probe_set))
                cross_e_vals.append(jaccard(ra.edge_probe_set, rb.edge_probe_set))
        else:
            # full cross (can be expensive if many runs; ok for small run count)
            for a in SA:
                for b in SB:
                    cross_cur_vals.append(jaccard(a, b))
            for a in PA_n:
                for b in PB_n:
                    cross_n_vals.append(jaccard(a, b))
            for a in PA_e:
                for b in PB_e:
                    cross_e_vals.append(jaccard(a, b))

        cross_cur = float(np.mean(cross_cur_vals)) if cross_cur_vals else np.nan
        cross_n = float(np.mean(cross_n_vals)) if cross_n_vals else np.nan
        cross_e = float(np.mean(cross_e_vals)) if cross_e_vals else np.nan

        # mean sizes
        mean_k_cur_A = float(np.mean([len(s) for s in SA])) if SA else np.nan
        mean_k_cur_B = float(np.mean([len(s) for s in SB])) if SB else np.nan
        mean_k_n_A = float(np.mean([len(s) for s in PA_n])) if PA_n else np.nan
        mean_k_n_B = float(np.mean([len(s) for s in PB_n])) if PB_n else np.nan
        mean_k_e_A = float(np.mean([len(s) for s in PA_e])) if PA_e else np.nan
        mean_k_e_B = float(np.mean([len(s) for s in PB_e])) if PB_e else np.nan

        # flip-like metrics within each method (run-wise delta from t-1)
        def _deltas(method_key):
            adds, rems, syms = [], [], []
            if t == iters[0]:
                return np.nan, np.nan, np.nan
            tprev = iters[iters.index(t) - 1]
            if method_key == "A":
                runs = trajA
                kmap = keyA
            else:
                runs = trajB
                kmap = keyB
            for r in runs:
                cur = kmap.get((int(t), int(r)), None)
                prv = kmap.get((int(tprev), int(r)), None)
                if cur is None or prv is None:
                    continue
                s1 = prv.current_set
                s2 = cur.current_set
                adds.append(len(s2 - s1))
                rems.append(len(s1 - s2))
                syms.append(len(s1.symmetric_difference(s2)))
            if not adds:
                return np.nan, np.nan, np.nan
            return float(np.mean(adds)), float(np.mean(rems)), float(np.mean(syms))

        addA, remA, symA = _deltas("A")
        addB, remB, symB = _deltas("B")

        rows.append({
            "iteration": int(t),
            "within_A_jaccard_cur": within_A_cur,
            "within_B_jaccard_cur": within_B_cur,
            "cross_AB_jaccard_cur": cross_cur,
            "within_A_jaccard_nodeprobe": within_A_n,
            "within_B_jaccard_nodeprobe": within_B_n,
            "cross_AB_jaccard_nodeprobe": cross_n,
            "within_A_jaccard_edgeprobe": within_A_e,
            "within_B_jaccard_edgeprobe": within_B_e,
            "cross_AB_jaccard_edgeprobe": cross_e,
            "mean_k_current_A": mean_k_cur_A,
            "mean_k_current_B": mean_k_cur_B,
            "mean_k_nodeprobe_A": mean_k_n_A,
            "mean_k_nodeprobe_B": mean_k_n_B,
            "mean_k_edgeprobe_A": mean_k_e_A,
            "mean_k_edgeprobe_B": mean_k_e_B,
            "mean_add_A": addA,
            "mean_remove_A": remA,
            "mean_symdiff_A": symA,
            "mean_add_B": addB,
            "mean_remove_B": remB,
            "mean_symdiff_B": symB,
        })

    return pd.DataFrame(rows).sort_values("iteration").reset_index(drop=True)


# ---------------------------
# B) Agenda execution metrics
# ---------------------------
def compute_agenda_execution(df_llm_iter: pd.DataFrame, df_control: pd.DataFrame, P_EDGE: Optional[int] = None) -> pd.DataFrame:
    """
    For each iteration t and run r, compute agenda execution metrics, then average across runs.

    Definitions (paper-ready):
      - Base-set compliance: controller intent realized at sampling layer
          hit_must_cur    = |must ∩ current_set| / |must|
          hit_prefer_cur  = |prefer ∩ current_set| / |prefer|
          viol_excl_cur   = |exclude ∩ current_set| / |exclude|
          hit_fend_cur    = |frontier_endpoints ∩ current_set| / |frontier_endpoints|

      - Probe compliance: controller intent realized at evidence acquisition layer
          hit_must_nodeprobe, hit_prefer_nodeprobe, viol_excl_nodeprobe (same form but with node_probe_set)

      - Frontier feasibility/execution under strict reuse:
          feasible_frontier = fraction of frontier edges whose BOTH endpoints are in node_probe_set
          exec_frontier     = fraction of frontier edges that appear in edge_probe_set

      - Edge budget health:
          edge_shortfall = max(0, P_EDGE - |edge_probe_set|) (only if P_EDGE provided)
    """
    # join on generation == iteration
    ctl = df_control.rename(columns={"generation": "iteration"}).copy()
    ctl = ctl[["iteration", "must_set", "prefer_set", "excl_set", "frontier_set", "frontier_end_set",
               "k_must", "k_prefer", "k_excl", "k_frontier", "k_fend"]].copy()

    # Expand per-run by merging control onto each (iteration,traj)
    df = df_llm_iter.merge(ctl, on="iteration", how="left")

    def _rate(inter_n: int, denom: int) -> float:
        return float(inter_n / max(1, denom))

    # Per-row metrics (each is for one run)
    hit_must_cur = []
    hit_prefer_cur = []
    viol_excl_cur = []
    hit_fend_cur = []

    hit_must_np = []
    hit_prefer_np = []
    viol_excl_np = []
    hit_fend_np = []

    feasible_frontier = []
    exec_frontier = []
    edge_shortfall = []

    for rr in df.itertuples(index=False):
        must = rr.must_set if isinstance(rr.must_set, set) else set()
        pref = rr.prefer_set if isinstance(rr.prefer_set, set) else set()
        excl = rr.excl_set if isinstance(rr.excl_set, set) else set()
        fr = rr.frontier_set if isinstance(rr.frontier_set, set) else set()
        fend = rr.frontier_end_set if isinstance(rr.frontier_end_set, set) else set()

        cur = rr.current_set
        np_set = rr.node_probe_set
        ep_set = rr.edge_probe_set

        hit_must_cur.append(_rate(len(must & cur), len(must)))
        hit_prefer_cur.append(_rate(len(pref & cur), len(pref)))
        viol_excl_cur.append(_rate(len(excl & cur), len(excl)))
        hit_fend_cur.append(_rate(len(fend & cur), len(fend)))

        hit_must_np.append(_rate(len(must & np_set), len(must)))
        hit_prefer_np.append(_rate(len(pref & np_set), len(pref)))
        viol_excl_np.append(_rate(len(excl & np_set), len(excl)))
        hit_fend_np.append(_rate(len(fend & np_set), len(fend)))

        # Frontier feasibility: both endpoints must be node-probed (strict reuse proxy)
        if len(fr) == 0:
            feasible_frontier.append(np.nan)
            exec_frontier.append(np.nan)
        else:
            # compute feasibility by endpoints in node probes
            feas = 0
            for (u, v) in fr:
                if (u in np_set) and (v in np_set):
                    feas += 1
            feasible_frontier.append(float(feas / max(1, len(fr))))
            exec_frontier.append(float(len(fr & ep_set) / max(1, len(fr))))

        if P_EDGE is None:
            edge_shortfall.append(np.nan)
        else:
            edge_shortfall.append(float(max(0, int(P_EDGE) - int(len(ep_set)))))

    df["_hit_must_cur"] = hit_must_cur
    df["_hit_prefer_cur"] = hit_prefer_cur
    df["_viol_excl_cur"] = viol_excl_cur
    df["_hit_fend_cur"] = hit_fend_cur

    df["_hit_must_np"] = hit_must_np
    df["_hit_prefer_np"] = hit_prefer_np
    df["_viol_excl_np"] = viol_excl_np
    df["_hit_fend_np"] = hit_fend_np

    df["_feasible_frontier"] = feasible_frontier
    df["_exec_frontier"] = exec_frontier
    df["_edge_shortfall"] = edge_shortfall

    # Aggregate across runs per iteration
    cols = [
        "_hit_must_cur", "_hit_prefer_cur", "_viol_excl_cur", "_hit_fend_cur",
        "_hit_must_np", "_hit_prefer_np", "_viol_excl_np", "_hit_fend_np",
        "_feasible_frontier", "_exec_frontier", "_edge_shortfall",
        "k_must", "k_prefer", "k_excl", "k_frontier", "k_fend",
        "k_current", "k_nodeprobe", "k_edgeprobe",
    ]
    g = df.groupby("iteration", as_index=False)[cols].mean()
    return g.sort_values("iteration").reset_index(drop=True)


# ---------------------------
# Footprint / frequency views
# ---------------------------
def feature_frequency(df_iter: pd.DataFrame, col_set: str, t_min: Optional[int] = None, t_max: Optional[int] = None) -> pd.Series:
    """
    Count how frequently each feature appears in a set-column across a time window.
    col_set must be one of: "current_set", "node_probe_set".
    """
    df = df_iter.copy()
    if t_min is not None:
        df = df[df["iteration"] >= int(t_min)]
    if t_max is not None:
        df = df[df["iteration"] <= int(t_max)]
    counts: Dict[str, int] = {}
    for s in df[col_set].tolist():
        for f in s:
            counts[f] = counts.get(f, 0) + 1
    return pd.Series(counts).sort_values(ascending=False)


def edge_frequency(df_iter: pd.DataFrame, col_set: str = "edge_probe_set", t_min: Optional[int] = None, t_max: Optional[int] = None) -> pd.Series:
    """
    Count frequency of probed edges in a time window.
    """
    df = df_iter.copy()
    if t_min is not None:
        df = df[df["iteration"] >= int(t_min)]
    if t_max is not None:
        df = df[df["iteration"] <= int(t_max)]
    counts: Dict[Tuple[str, str], int] = {}
    for s in df[col_set].tolist():
        for e in s:
            counts[e] = counts.get(e, 0) + 1
    ser = pd.Series({f"{u}--{v}": c for (u, v), c in counts.items()})
    return ser.sort_values(ascending=False)


# ---------------------------
# Plotting (simple, reusable)
# ---------------------------
def _plot_two_methods_series(
    ax,
    df: pd.DataFrame,
    xcol: str,
    yA: str,
    yB: str,
    *,
    labelA: str,
    labelB: str,
    title: str,
    ylabel: str,
):
    """
    Plot two series (Baseline vs LLM) in one axis with legend and y-label.
    This replaces many scattered one-off plot calls and makes panels consistent.
    """
    if yA in df.columns:
        ax.plot(df[xcol], df[yA], marker="o", markersize=3, label=labelA)
    if yB in df.columns:
        ax.plot(df[xcol], df[yB], marker="o", markersize=3, label=labelB)

    ax.set_title(title)
    ax.set_xlabel("Generation (= iteration)")
    ax.set_ylabel(ylabel)
    ax.grid(False)

    # Always show legend if any line exists
    if len(ax.lines) > 0:
        ax.legend(loc="best")


def _plot_multi_lines(
    ax,
    df: pd.DataFrame,
    xcol: str,
    ycols: list[str],
    *,
    labels: list[str] | None,
    title: str,
    ylabel: str,
):
    """
    Plot multiple lines in one axis.
    Important: labels must be provided to avoid missing legends (your request).
    """
    if labels is None:
        labels = ycols
    for c, lab in zip(ycols, labels):
        if c in df.columns:
            ax.plot(df[xcol], df[c], marker="o", markersize=3, label=lab)

    ax.set_title(title)
    ax.set_xlabel("Generation (= iteration)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2)

    if len(ax.lines) > 0:
        ax.legend(loc="best")


def plot_path_panels(df_path: pd.DataFrame, out_path: str, *, labelA="Baseline", labelB="LLM"):
    """
    One figure for PATH metrics.

    Panels included (paper-oriented):
      1) within-method coherence on current_set (mean pairwise Jaccard across runs)
      2) cross-method similarity on current_set (Baseline vs LLM)
      3) within-method coherence on edge_probes
      4) within-run dynamics: mean |symdiff(S_t, S_{t-1})| across runs

    Y-labels are explicit so the panels are interpretable when exported to the paper.
    """
    panels = [
        ("within_A_jaccard_cur", "within_B_jaccard_cur",
         "Within-method coherence (current_set)", "Mean pairwise Jaccard"),
        ("within_A_jaccard_nodeprobe", "within_B_jaccard_nodeprobe",
         "Within-method coherence (node_probes)", "Mean pairwise Jaccard"),
        ("within_A_jaccard_edgeprobe", "within_B_jaccard_edgeprobe",
         "Within-method coherence (edge_probes)", "Mean pairwise Jaccard"),
        ("cross_AB_jaccard_cur", None,
         "Cross-method similarity (current_set)", "Mean Jaccard (A vs B)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.reshape(-1)

    for i, (yA, yB, title, ylabel) in enumerate(panels):
        ax = axes[i]
        if yB is None:
            # single-line case (cross_AB is already a single curve)
            _plot_multi_lines(
                ax, df_path, "iteration",
                [yA],
                labels=["Baseline vs LLM"],
                title=title,
                ylabel=ylabel,
            )
        else:
            _plot_two_methods_series(
                ax, df_path, "iteration",
                yA, yB,
                labelA=labelA, labelB=labelB,
                title=title,
                ylabel=ylabel,
            )

    fig.suptitle("Path difference metrics", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_agenda_panels(df_exec: pd.DataFrame, out_path: str):
    """
    One figure for AGENDA execution/compliance metrics (LLM only).

    Panels:
      1) Base-set compliance vs current_set
      2) Probe compliance vs node_probe_set
      3) Frontier feasibility vs execution (strict reuse proxy)

    Note:
      - We intentionally do NOT plot edge_shortfall (per your request).
      - Each panel has y-label and legend (paper-ready).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharex=True)
    axes = np.array(axes).reshape(-1)

    # (1) current_set compliance
    _plot_multi_lines(
        axes[0],
        df_exec,
        "iteration",
        ["_hit_must_cur", "_hit_prefer_cur", "_hit_fend_cur", "_viol_excl_cur"],
        labels=["hit must (current_set)", "hit prefer (current_set)", "hit frontier-end (current_set)", "violate excl (current_set)"],
        title="Agenda → base-set compliance",
        ylabel="Rate",
    )

    # (2) node_probes compliance
    _plot_multi_lines(
        axes[1],
        df_exec,
        "iteration",
        ["_hit_must_np", "_hit_prefer_np", "_hit_fend_np", "_viol_excl_np"],
        labels=["hit must (node_probes)", "hit prefer (node_probes)", "hit frontier-end (node_probes)", "violate excl (node_probes)"],
        title="Agenda → probe compliance",
        ylabel="Rate",
    )

    # (3) frontier feasibility/execution
    _plot_multi_lines(
        axes[2],
        df_exec,
        "iteration",
        ["_feasible_frontier", "_exec_frontier"],
        labels=["feasible frontier (endpoints in node_probes)", "executed frontier (in edge_probes)"],
        title="Frontier feasibility vs execution",
        ylabel="Rate",
    )

    fig.suptitle("Agenda execution metrics (LLM)", y=1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_footprint_panels(
    freqA: pd.DataFrame,
    freqB: pd.DataFrame,
    npA: pd.DataFrame,
    npB: pd.DataFrame,
    epA: pd.DataFrame,
    epB: pd.DataFrame,
    *,
    topk: int,
    labelA: str,
    labelB: str,
    window_title: str,
    out_path: str,
):
    """
    One figure for FOOTPRINT comparisons in a chosen window.

    Each input freq table is expected to be sorted descending by frequency and contain:
      - column 0: item (feature name or edge key)
      - column 1: count/frequency
    If your tables have explicit column names, adapt the access lines accordingly.
    """

    def _as_series(df: pd.DataFrame, k: int) -> pd.Series:
        # robust: accept Series or DataFrame (["item","count"] or unnamed columns)
        if df is None:
            return pd.Series(dtype=float)
        if isinstance(df, pd.Series):
            return df.head(k)
        if len(df) == 0:
            return pd.Series(dtype=float)
        if "item" in df.columns and "count" in df.columns:
            s = df.set_index("item")["count"]
        else:
            s = df.set_index(df.columns[0])[df.columns[1]]
        return s.head(k)

    def _plot_bar(ax, sA: pd.Series, sB: pd.Series, title: str, ylabel: str):
        # union of top-k items from both for fair side-by-side comparison
        items = list(dict.fromkeys(list(sA.index) + list(sB.index)))[:topk]
        a = np.array([float(sA.get(it, 0.0)) for it in items], dtype=float)
        b = np.array([float(sB.get(it, 0.0)) for it in items], dtype=float)

        y = np.arange(len(items))
        h = 0.45

        ax.barh(y - h/2, a, height=h, label=labelA)
        ax.barh(y + h/2, b, height=h, label=labelB)

        ax.set_title(title)
        ax.set_xlabel(ylabel)
        ax.set_yticks(y)
        ax.set_yticklabels(items, fontsize=8)
        ax.grid(True, axis="x", alpha=0.2)
        ax.legend(loc="best")

    s_cur_A = _as_series(freqA, topk)
    s_cur_B = _as_series(freqB, topk)
    s_np_A  = _as_series(npA, topk)
    s_np_B  = _as_series(npB, topk)
    s_ep_A  = _as_series(epA, topk)
    s_ep_B  = _as_series(epB, topk)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes = np.array(axes).reshape(-1)

    _plot_bar(axes[0], s_cur_A, s_cur_B, f"Footprint: current_set frequency\n{window_title}", "Count")
    _plot_bar(axes[1], s_np_A,  s_np_B,  f"Footprint: node_probes frequency\n{window_title}", "Count")
    _plot_bar(axes[2], s_ep_A,  s_ep_B,  f"Footprint: edge_probes frequency\n{window_title}", "Count")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------
# Main entry v1
# ---------------------------
def run_path_and_agenda_analysis(
    out_root: str,
    method_a: str = "baseline_iters200",
    method_b: str = "Qwen_iters200",
    *,
    save_dir: Optional[str] = None,
    P_EDGE: Optional[int] = 20,
    topk: int = 30,
):
    """
    Generates (paper-ready, compact):
      - path_metrics_by_iter.csv + path_panels.png
      - agenda_exec_by_iter.csv + agenda_panels.png (LLM only)
      - footprint_panels.png + frequency tables (top200 CSVs)

    Design principle:
      - Each category produces ONE figure with multiple subplots to enable horizontal comparison.
      - All subplots include explicit y-axis labels.
      - All agenda plots include legends.
      - Edge shortfall plot is intentionally omitted (not informative in your setting).
    """
    if save_dir is None:
        save_dir = os.path.join(out_root, "kg_path_agenda")
    os.makedirs(save_dir, exist_ok=True)

    # 1) Load and normalize iteration logs
    df_raw_a = _load_iter_log(out_root, method_a)
    df_raw_b = _load_iter_log(out_root, method_b)
    dfA = normalize_iter_df(df_raw_a)
    dfB = normalize_iter_df(df_raw_b)

    print("Loaded iteration logs:")
    print(" - A:", method_a, "from", df_raw_a["_source_path"].iloc[0], "rows=", len(dfA))
    print(" - B:", method_b, "from", df_raw_b["_source_path"].iloc[0], "rows=", len(dfB))

    # 2) Path difference metrics + ONE consolidated plot
    df_path = compute_path_metrics(dfA, dfB)
    path_csv = os.path.join(save_dir, "path_metrics_by_iter.csv")
    df_path.to_csv(path_csv, index=False)
    print("Saved:", path_csv)

    path_fig = os.path.join(save_dir, "path_panels.png")
    plot_path_panels(df_path, path_fig, labelA="Baseline", labelB="LLM")
    print("Saved:", path_fig)

    # 3) Agenda execution (LLM only) + ONE consolidated plot
    df_exec = None
    try:
        df_ctl_raw = _load_control_log(out_root, method_b)
        df_ctl = normalize_control_df(df_ctl_raw)
        print("Loaded control log:", df_ctl_raw["_source_path"].iloc[0], "rows=", len(df_ctl))

        df_exec = compute_agenda_execution(dfB, df_ctl, P_EDGE=P_EDGE)

        exec_csv = os.path.join(save_dir, "agenda_exec_by_iter.csv")
        df_exec.to_csv(exec_csv, index=False)
        print("Saved:", exec_csv)

        agenda_fig = os.path.join(save_dir, "agenda_panels.png")
        plot_agenda_panels(df_exec, agenda_fig)
        print("Saved:", agenda_fig)

    except Exception as e:
        print("[WARN] control_log not analyzed:", repr(e))

    # 4) Footprint / frequency comparisons (last window) + ONE consolidated plot
    t_max = int(max(dfA["iteration"].max(), dfB["iteration"].max()))
    t_min = max(0, t_max - 50 + 1)

    # These functions are assumed already present in your program:
    #   - feature_frequency(df, "current_set", ...)
    #   - feature_frequency(df, "node_probe_set", ...)
    #   - edge_frequency(df, "edge_probe_set", ...)
    freqA = feature_frequency(dfA, "current_set", t_min=t_min, t_max=t_max)
    freqB = feature_frequency(dfB, "current_set", t_min=t_min, t_max=t_max)

    npA = feature_frequency(dfA, "node_probe_set", t_min=t_min, t_max=t_max)
    npB = feature_frequency(dfB, "node_probe_set", t_min=t_min, t_max=t_max)

    epA = edge_frequency(dfA, "edge_probe_set", t_min=t_min, t_max=t_max)
    epB = edge_frequency(dfB, "edge_probe_set", t_min=t_min, t_max=t_max)

    footprint_fig = os.path.join(save_dir, "footprint_panels.png")
    plot_footprint_panels(
        freqA, freqB, npA, npB, epA, epB,
        topk=topk,
        labelA="EGL",
        labelB="EGL-LLM",
        window_title=f"Window [{t_min}, {t_max}]",
        out_path=footprint_fig,
    )
    print("Saved:", footprint_fig)

    # Save frequency tables (useful for paper tables)
    freqA.head(200).to_csv(os.path.join(save_dir, "freq_currentset_baseline_top200.csv"), index=False)
    freqB.head(200).to_csv(os.path.join(save_dir, "freq_currentset_llm_top200.csv"), index=False)
    npA.head(200).to_csv(os.path.join(save_dir, "freq_nodeprobes_baseline_top200.csv"), index=False)
    npB.head(200).to_csv(os.path.join(save_dir, "freq_nodeprobes_llm_top200.csv"), index=False)
    epA.head(200).to_csv(os.path.join(save_dir, "freq_edgeprobes_baseline_top200.csv"), index=False)
    epB.head(200).to_csv(os.path.join(save_dir, "freq_edgeprobes_llm_top200.csv"), index=False)

    print("\nDone. Outputs in:", save_dir)
    print("Key files:")
    print(" -", path_csv)
    if df_exec is not None:
        print(" -", os.path.join(save_dir, "agenda_exec_by_iter.csv"))
    print(" -", footprint_fig)


# ============================
# NEW: iteration-level "state"
# ============================
def build_iteration_level_state(
    df_iter: pd.DataFrame,
    set_col: str = "current_set",
    *,
    agg: str = "union",
) -> pd.DataFrame:
    """
    Build an iteration-level representation of the exploration "state".

    Why this is useful (paper-friendly):
    - Your raw logs are per-(iteration, traj_id). That is too detailed for a compact figure.
    - For a trajectory map, we need one "state" per iteration per method.

    We aggregate across runs at each iteration:
      agg="union" (default): S_t = union_r S_{t,r}
        - Interpretable as "what the method collectively visited at iteration t"
        - Good for visualizing basin overlap between methods.

      (You can later add agg="median_run" etc, but union is robust and cheap.)

    Returns a DataFrame:
      iteration, state_set (python set), k_state (size)
    """
    rows = []
    for t, g in df_iter.groupby("iteration", as_index=False):
        sets = g[set_col].tolist()
        if agg == "union":
            s = set().union(*sets) if sets else set()
        elif agg == "intersection":
            s = set.intersection(*sets) if sets else set()
        else:
            raise ValueError(f"Unknown agg='{agg}'. Use 'union' or 'intersection'.")
        rows.append({"iteration": int(t), "state_set": s, "k_state": len(s)})
    return pd.DataFrame(rows).sort_values("iteration").reset_index(drop=True)


# ==========================================
# NEW: Classical MDS (no sklearn dependency)
# ==========================================
def classical_mds(D: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Classical MDS embedding from a distance matrix (N x N).
    This avoids introducing sklearn dependency and is deterministic.

    Steps:
      1) Convert distances to centered Gram matrix B via double-centering:
         B = -0.5 * J * (D^2) * J,  where J = I - 1/N * 11^T
      2) Eigendecomposition of B
      3) Take top positive eigenvalues/vectors

    Returns:
      X: (N, n_components) embedding.

    Notes:
      - If some eigenvalues are negative (numerical), we clamp at 0.
      - Works well when distances are approximately Euclidean; for Jaccard-distance
        it still gives a useful low-D "map" for visual comparison.
    """
    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square distance matrix.")
    N = D.shape[0]
    if N < 2:
        return np.zeros((N, n_components), dtype=float)

    # Double centering
    J = np.eye(N) - np.ones((N, N)) / N
    D2 = D ** 2
    B = -0.5 * J @ D2 @ J

    # Eigendecomposition (symmetric)
    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1]
    w = w[idx]
    V = V[:, idx]

    w = np.maximum(w, 0.0)
    k = min(n_components, N)
    L = np.diag(np.sqrt(w[:k]))
    X = V[:, :k] @ L
    if X.shape[1] < n_components:
        X = np.pad(X, ((0, 0), (0, n_components - X.shape[1])), mode="constant")
    return X


# ====================================================
# NEW: Path trajectory embedding (paper-ready figure)
# ====================================================
def plot_path_trajectory_embedding(
    dfA: pd.DataFrame,
    dfB: pd.DataFrame,
    *,
    set_col: str = "current_set",
    agg: str = "union",
    labelA: str = "Baseline",
    labelB: str = "LLM",
    out_path: str,
):
    """
    Paper-friendly alternative to many path curves:
    - Convert per-iteration sets into a 2D trajectory map.
    - Show whether Baseline and LLM trajectories overlap (same basin) or diverge.

    Construction:
      1) Build iteration-level states for each method: S^A_t, S^B_t
      2) Create a combined list of states:
          [("A", t, S^A_t), ..., ("B", t, S^B_t)]
      3) Compute distance matrix using Jaccard distance: d=1-Jaccard
      4) Embed with classical MDS to 2D
      5) Plot 2 polylines (ordered by iteration), and mark start/end.

    Interpretation:
      - Strong overlap in 2D = both methods explore similar basins.
      - If LLM reaches a "stable region" earlier, its path may become shorter / more clustered.
    """
    stA = build_iteration_level_state(dfA, set_col=set_col, agg=agg)
    stB = build_iteration_level_state(dfB, set_col=set_col, agg=agg)

    items = []
    for r in stA.itertuples(index=False):
        items.append(("A", int(r.iteration), r.state_set))
    for r in stB.itertuples(index=False):
        items.append(("B", int(r.iteration), r.state_set))

    N = len(items)
    D = np.zeros((N, N), dtype=float)
    for i in range(N):
        si = items[i][2]
        for j in range(i + 1, N):
            sj = items[j][2]
            d = 1.0 - jaccard(si, sj)
            D[i, j] = D[j, i] = d

    X = classical_mds(D, n_components=2)

    # indices for A and B in correct time order
    idxA = [i for i, (m, t, _) in enumerate(items) if m == "A"]
    idxB = [i for i, (m, t, _) in enumerate(items) if m == "B"]
    idxA = sorted(idxA, key=lambda i: items[i][1])
    idxB = sorted(idxB, key=lambda i: items[i][1])

    def _alpha_schedule(n: int, *, tail: int = 50, a_min: float = 0.05, a_max: float = 0.9, gamma: float = 2.2):
        if n <= 0:
            return np.array([], dtype=float)
        t = np.linspace(0.0, 1.0, n)
        a = a_min + (a_max - a_min) * (t ** gamma)
        if n > tail:
            start = max(a[n - tail], 0.6)
            a[n - tail :] = np.linspace(start, a_max, tail)
        return a

    def _plot_traj(ax, X, idx, color, label):
        if not idx:
            return
        from matplotlib.collections import LineCollection
        from matplotlib.lines import Line2D

        pts = X[idx, :]
        n = len(idx)
        alphas = _alpha_schedule(n)

        # line segments with per-segment alpha
        if n >= 2:
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            seg_colors = [(color[0], color[1], color[2], float(a)) for a in alphas[1:]]
            lc = LineCollection(segs, colors=seg_colors, linewidths=1.2)
            ax.add_collection(lc)

        # points with per-point alpha
        pt_colors = [(color[0], color[1], color[2], float(a)) for a in alphas]
        ax.scatter(pts[:, 0], pts[:, 1], s=15, c=pt_colors, edgecolors="face", linewidths=0)

        # legend handle
        ax.add_line(Line2D([0], [0], color=color, lw=2, label=label))

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    _plot_traj(ax, X, idxA, color=(0.121, 0.466, 0.705), label=labelA)  # tab:blue
    _plot_traj(ax, X, idxB, color=(1.0, 0.498, 0.055), label=labelB)    # tab:orange

    # mark starts/ends
    if idxA:
        ax.scatter([X[idxA[0], 0]], [X[idxA[0], 1]], s=80, marker="s", edgecolors="face", linewidths=0)
        ax.scatter([X[idxA[-1], 0]], [X[idxA[-1], 1]], s=80, marker="X", edgecolors="face", linewidths=0)
    if idxB:
        ax.scatter([X[idxB[0], 0]], [X[idxB[0], 1]], s=80, marker="s", edgecolors="face", linewidths=0)
        ax.scatter([X[idxB[-1], 0]], [X[idxB[-1], 1]], s=80, marker="X", edgecolors="face", linewidths=0)

    ax.set_title(f"Trajectory map via MDS (state={agg} over runs, set={set_col})")
    ax.set_xlabel("MDS-1")
    ax.set_ylabel("MDS-2")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_path_trajectory_embedding_3d(
    dfA: pd.DataFrame,
    dfB: pd.DataFrame,
    *,
    set_col: str = "current_set",
    agg: str = "union",
    labelA: str = "Baseline",
    labelB: str = "LLM",
    elev: float = 20.0,
    azim: float = 55.59,
    out_path: str,
):
    """
    3D trajectory map:
      - x/y from MDS embedding
      - z from time order (iteration index)
    """
    stA = build_iteration_level_state(dfA, set_col=set_col, agg=agg)
    stB = build_iteration_level_state(dfB, set_col=set_col, agg=agg)

    items = []
    for r in stA.itertuples(index=False):
        items.append(("A", int(r.iteration), r.state_set))
    for r in stB.itertuples(index=False):
        items.append(("B", int(r.iteration), r.state_set))

    N = len(items)
    D = np.zeros((N, N), dtype=float)
    for i in range(N):
        si = items[i][2]
        for j in range(i + 1, N):
            sj = items[j][2]
            d = 1.0 - jaccard(si, sj)
            D[i, j] = D[j, i] = d

    X = classical_mds(D, n_components=2)

    idxA = [i for i, (m, t, _) in enumerate(items) if m == "A"]
    idxB = [i for i, (m, t, _) in enumerate(items) if m == "B"]
    idxA = sorted(idxA, key=lambda i: items[i][1])
    idxB = sorted(idxB, key=lambda i: items[i][1])

    def _alpha_schedule(n: int, *, tail: int = 50, a_min: float = 0.05, a_max: float = 0.9, gamma: float = 2.2):
        if n <= 0:
            return np.array([], dtype=float)
        t = np.linspace(0.0, 1.0, n)
        a = a_min + (a_max - a_min) * (t ** gamma)
        if n > tail:
            start = max(a[n - tail], 0.6)
            a[n - tail :] = np.linspace(start, a_max, tail)
        return a

    def _plot_traj_3d(ax, X, idx, color, label):
        if not idx:
            return
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        from matplotlib.lines import Line2D

        pts2d = X[idx, :]
        z = np.array([items[i][1] for i in idx], dtype=float)
        pts3d = np.column_stack([pts2d[:, 0], pts2d[:, 1], z])
        n = len(idx)
        alphas = _alpha_schedule(n)

        if n >= 2:
            segs = np.stack([pts3d[:-1], pts3d[1:]], axis=1)
            seg_colors = [(color[0], color[1], color[2], float(a)) for a in alphas[1:]]
            lc = Line3DCollection(segs, colors=seg_colors, linewidths=1.0)
            ax.add_collection(lc)

        pt_colors = [(color[0], color[1], color[2], float(a)) for a in alphas]
        ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2], s=10, c=pt_colors, edgecolors="face", linewidths=0)

        ax.add_line(Line2D([0], [0], color=color, lw=2, label=label))

        # mark start/end
        ax.scatter([pts3d[0, 0]], [pts3d[0, 1]], [pts3d[0, 2]], s=20, marker="s", color=color, alpha=0.9, edgecolors="face", linewidths=0)
        ax.scatter([pts3d[-1, 0]], [pts3d[-1, 1]], [pts3d[-1, 2]], s=20, marker="X", color=color, alpha=0.9, edgecolors="face", linewidths=0)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")

    _plot_traj_3d(ax, X, idxA, color=(0.121, 0.466, 0.705), label=labelA)  # tab:blue
    _plot_traj_3d(ax, X, idxB, color=(1.0, 0.498, 0.055), label=labelB)    # tab:orange

    # auto-tighten x/y limits to reduce empty space
    x_min, x_max = np.min(X[:, 0]), np.max(X[:, 0])
    y_min, y_max = np.min(X[:, 1]), np.max(X[:, 1])
    pad_x = 0.05 * (x_max - x_min + 1e-9)
    pad_y = 0.05 * (y_max - y_min + 1e-9)
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)

    # 手动设置
    # ax.set_xlim(-0.3, 0.3)
    # ax.set_ylim(-0.3, 0.3)

    ax.view_init(elev=elev, azim=azim)

    ax.set_title(f"Trajectory map via MDS (z=time, state={agg}, set={set_col})")
    ax.set_xlabel("MDS-1")
    ax.set_ylabel("MDS-2")
    ax.set_zlabel("Time (iteration order)")
    ax.grid(False)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_path_trajectory_embedding_2d_density(
    dfA: pd.DataFrame,
    dfB: pd.DataFrame,
    *,
    set_col: str = "current_set",
    agg: str = "union",
    labelA: str = "Baseline",
    labelB: str = "LLM",
    tail: int = 500,
    out_path: str,
):
    """
    2D MDS + density contour + late-stage points.
    """
    stA = build_iteration_level_state(dfA, set_col=set_col, agg=agg)
    stB = build_iteration_level_state(dfB, set_col=set_col, agg=agg)

    items = []
    for r in stA.itertuples(index=False):
        items.append(("A", int(r.iteration), r.state_set))
    for r in stB.itertuples(index=False):
        items.append(("B", int(r.iteration), r.state_set))

    N = len(items)
    D = np.zeros((N, N), dtype=float)
    for i in range(N):
        si = items[i][2]
        for j in range(i + 1, N):
            sj = items[j][2]
            d = 1.0 - jaccard(si, sj)
            D[i, j] = D[j, i] = d

    X = classical_mds(D, n_components=2)

    idxA = [i for i, (m, t, _) in enumerate(items) if m == "A"]
    idxB = [i for i, (m, t, _) in enumerate(items) if m == "B"]
    idxA = sorted(idxA, key=lambda i: items[i][1])
    idxB = sorted(idxB, key=lambda i: items[i][1])

    def _gaussian_kernel1d(sigma: float, radius: int) -> np.ndarray:
        x = np.arange(-radius, radius + 1, dtype=float)
        k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
        k /= k.sum()
        return k

    def _smooth2d(z: np.ndarray, sigma: float) -> np.ndarray:
        radius = max(1, int(3 * sigma))
        k = _gaussian_kernel1d(sigma, radius)
        z1 = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, z)
        z2 = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, z1)
        return z2

    def _density_contourf(ax, pts, cmap, hist_range, alpha=0.5):
        if pts.shape[0] < 5:
            return None
        x = pts[:, 0]
        y = pts[:, 1]
        bins = 40
        H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=hist_range)
        H = _smooth2d(H, sigma=1.5)

        if np.all(H <= 0):
            return None
        vals = H[H > 0]
        if vals.size == 0:
            return None
        levels = np.quantile(vals, [0.0, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
        levels = np.unique(levels)
        if levels.size == 0:
            return None

        xcenters = (xedges[:-1] + xedges[1:]) / 2.0
        ycenters = (yedges[:-1] + yedges[1:]) / 2.0
        Xg, Yg = np.meshgrid(xcenters, ycenters, indexing="xy")
        cs = ax.contourf(Xg, Yg, H.T, levels=levels, cmap=cmap, alpha=alpha, extend="max")
        # cs = ax.contour(Xg, Yg, H.T, levels=levels, colors="k", linewidths=0.2, alpha=0.5)
        return cs

    # MICCAI 论文使用
    fig, ax = plt.subplots(1, 1, figsize=(2.1, 2.0))
    colorA = "tab:blue"  # tab:blue
    colorB = "tab:orange"    # tab:orange

    ptsA_tail = X[idxA[-tail:], :] if idxA else None
    colorB = (1.0, 0.498, 0.055)    # tab:orange

    ptsA_tail = X[idxA[-tail:], :] if idxA else None
    ptsB_tail = X[idxB[-tail:], :] if idxB else None
    if (ptsA_tail is not None) and (ptsB_tail is not None):
        pts_all = np.vstack([ptsA_tail, ptsB_tail])
    else:
        pts_all = ptsA_tail if ptsA_tail is not None else ptsB_tail
    if pts_all is not None:
        x_min, x_max = float(np.min(pts_all[:, 0])), float(np.max(pts_all[:, 0]))
        y_min, y_max = float(np.min(pts_all[:, 1])), float(np.max(pts_all[:, 1]))
        pad_x = 0.05 * (x_max - x_min + 1e-9)
        pad_y = 0.05 * (y_max - y_min + 1e-9)
        hist_range = [(x_min - pad_x, x_max + pad_x), (y_min - pad_y, y_max + pad_y)]
    else:
        hist_range = None

    csA = _density_contourf(ax, ptsA_tail, cmap="Blues", hist_range=hist_range, alpha=0.9) if ptsA_tail is not None else None
    csB = _density_contourf(ax, ptsB_tail, cmap="Oranges", hist_range=hist_range, alpha=0.6) if ptsB_tail is not None else None

    # stack two colorbars vertically on the right to save space
    if (csA is not None) or (csB is not None):
        box = ax.get_position()
        y0 = box.y0 + 0.17
        x0 = box.x1 + 0.08
        w = 0.02
        h = 0.25
        gap = 0.1
        if csB is not None:
            cax_b = fig.add_axes([x0, y0, w, h])
            cbar_b = fig.colorbar(csB, cax=cax_b, format="%.1f")
            cbar_b.set_label(f"{labelB}", labelpad=-8, fontsize=5)
            if csB.levels is not None and len(csB.levels) > 0:
                cbar_b.set_ticks([csB.levels[0], csB.levels[-1]])
                cbar_b.ax.tick_params(labelsize=5)
        if csA is not None:
            cax_a = fig.add_axes([x0, y0 + h + gap, w, h])
            cbar_a = fig.colorbar(csA, cax=cax_a, format="%.1f")
            cbar_a.set_label(f"{labelA}", labelpad=-8, fontsize=5)
            if csA.levels is not None and len(csA.levels) > 0:
                cbar_a.set_ticks([csA.levels[0], csA.levels[-1]])
                cbar_a.ax.tick_params(labelsize=5)

    # late-stage points
    if ptsA_tail is not None:
        ax.scatter(ptsA_tail[:, 0], ptsA_tail[:, 1], s=1, color=colorA, alpha=0.9, label=labelA, linewidths=0)
    if ptsB_tail is not None:
        ax.scatter(ptsB_tail[:, 0], ptsB_tail[:, 1], s=1, color=colorB, alpha=0.9, label=labelB, linewidths=0)

    # mark start points
    if idxA:
        ax.scatter([X[idxA[0], 0]], [X[idxA[0], 1]], s=30, marker="*", color="green", alpha=1.0, linewidths=0)
    if idxB:
        ax.scatter([X[idxB[0], 0]], [X[idxB[0], 1]], s=30, marker="*", color="#0eff5e", alpha=1.0, linewidths=0)

    import matplotlib.ticker as mticker

    # ax.set_title(f"Trajectory density via MDS (late {tail} points)")
    ax.set_xlabel("MDS-1")
    ax.set_ylabel("MDS-2")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.grid(True, alpha=0.2)
    leg = ax.legend(loc="best", fontsize = 5, frameon=True, edgecolor="lightgray", framealpha=1.0)
    leg.get_frame().set_linewidth(0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ==========================================================
# NEW: Agenda size distribution (mean / quantiles over time)
# ==========================================================
def compute_agenda_size_distribution(df_ctl: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize agenda sizes across generations.

    Input:
      df_ctl: output of normalize_control_df(control_log.csv), already includes:
        k_must, k_prefer, k_excl, k_frontier, k_fend

    Output:
      A compact table with mean, p10, p50, p90 for each agenda component.

    Why:
      - You said control_log does not store distributions. This provides it.
      - These numbers explain why hit-rates hover around ~0.4 (because agenda occupies <= ~40% budget etc).
    """
    cols = ["k_must", "k_prefer", "k_excl", "k_frontier", "k_fend"]
    rows = []
    for c in cols:
        x = df_ctl[c].dropna().astype(float).values
        if x.size == 0:
            rows.append({"item": c, "mean": np.nan, "p10": np.nan, "p50": np.nan, "p90": np.nan})
            continue
        rows.append({
            "item": c,
            "mean": float(np.mean(x)),
            "p10": float(np.quantile(x, 0.10)),
            "p50": float(np.quantile(x, 0.50)),
            "p90": float(np.quantile(x, 0.90)),
        })
    return pd.DataFrame(rows)


def plot_agenda_size_table(df_size: pd.DataFrame, out_path: str):
    """
    Render agenda size distribution as a single compact table-figure.

    This is MICCAI-friendly:
      - One small table instead of many curves
      - Directly communicates controller "how much it asks for"
    """
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 2.2))
    ax.axis("off")

    df2 = df_size.copy()
    df2["mean"] = df2["mean"].map(lambda v: "-" if pd.isna(v) else f"{v:.2f}")
    df2["p10"] = df2["p10"].map(lambda v: "-" if pd.isna(v) else f"{v:.0f}")
    df2["p50"] = df2["p50"].map(lambda v: "-" if pd.isna(v) else f"{v:.0f}")
    df2["p90"] = df2["p90"].map(lambda v: "-" if pd.isna(v) else f"{v:.0f}")

    cell_text = df2[["item", "mean", "p10", "p50", "p90"]].values.tolist()
    table = ax.table(
        cellText=cell_text,
        colLabels=["agenda item", "mean", "p10", "p50", "p90"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(1, 1.2)

    ax.set_title("Agenda size distribution over generations", pad=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ==========================================================
# NEW: Per-(iter,traj) hit distributions (not only the mean)
# ==========================================================
def compute_agenda_hit_rows(
    df_llm_iter: pd.DataFrame,
    df_ctl: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute per-row (iteration, traj_id) agenda hit fractions & hit counts.
    This is the basis for "long-tail / bimodality" analysis.

    Output columns (each row corresponds to one run at one iteration):
      - hit_must_cur_frac, hit_prefer_cur_frac, viol_excl_cur_frac, hit_fend_cur_frac
      - hit_must_np_frac,  hit_prefer_np_frac,  viol_excl_np_frac,  hit_fend_np_frac
      - hit_must_cur_n, hit_prefer_cur_n, ...
      - k_must, k_prefer, k_excl, k_fend, k_frontier
      - feasible_frontier_frac, executed_frontier_frac

    Note:
      - This is *distributional* (you can plot histograms, see if two modes exist).
      - It complements compute_agenda_execution() which averages over runs.
    """
    ctl = df_ctl.rename(columns={"generation": "iteration"}).copy()
    keep = ["iteration", "must_set", "prefer_set", "excl_set", "frontier_set", "frontier_end_set",
            "k_must", "k_prefer", "k_excl", "k_frontier", "k_fend"]
    ctl = ctl[keep].copy()

    df = df_llm_iter.merge(ctl, on="iteration", how="left")

    def _frac(hit: int, denom: int) -> float:
        return float(hit / max(1, denom))

    out_rows = []
    for rr in df.itertuples(index=False):
        must = rr.must_set if isinstance(rr.must_set, set) else set()
        pref = rr.prefer_set if isinstance(rr.prefer_set, set) else set()
        excl = rr.excl_set if isinstance(rr.excl_set, set) else set()
        fr = rr.frontier_set if isinstance(rr.frontier_set, set) else set()
        fend = rr.frontier_end_set if isinstance(rr.frontier_end_set, set) else set()

        cur = rr.current_set
        np_set = rr.node_probe_set
        ep_set = rr.edge_probe_set

        # current_set hits
        hm_cur = len(must & cur)
        hp_cur = len(pref & cur)
        vx_cur = len(excl & cur)
        hf_cur = len(fend & cur)

        # node_probes hits
        hm_np = len(must & np_set)
        hp_np = len(pref & np_set)
        vx_np = len(excl & np_set)
        hf_np = len(fend & np_set)

        # frontier feasibility/execution
        if len(fr) == 0:
            feas = np.nan
            exe = np.nan
        else:
            feas_n = 0
            for (u, v) in fr:
                if (u in np_set) and (v in np_set):
                    feas_n += 1
            feas = float(feas_n / max(1, len(fr)))
            exe = float(len(fr & ep_set) / max(1, len(fr)))

        out_rows.append({
            "iteration": int(rr.iteration),
            "traj_id": int(rr.traj_id),

            "hit_must_cur_n": hm_cur,
            "hit_prefer_cur_n": hp_cur,
            "viol_excl_cur_n": vx_cur,
            "hit_fend_cur_n": hf_cur,

            "hit_must_np_n": hm_np,
            "hit_prefer_np_n": hp_np,
            "viol_excl_np_n": vx_np,
            "hit_fend_np_n": hf_np,

            "hit_must_cur_frac": _frac(hm_cur, len(must)),
            "hit_prefer_cur_frac": _frac(hp_cur, len(pref)),
            "viol_excl_cur_frac": _frac(vx_cur, len(excl)),
            "hit_fend_cur_frac": _frac(hf_cur, len(fend)),

            "hit_must_np_frac": _frac(hm_np, len(must)),
            "hit_prefer_np_frac": _frac(hp_np, len(pref)),
            "viol_excl_np_frac": _frac(vx_np, len(excl)),
            "hit_fend_np_frac": _frac(hf_np, len(fend)),

            "feasible_frontier_frac": feas,
            "executed_frontier_frac": exe,

            "k_must": int(rr.k_must) if not pd.isna(rr.k_must) else 0,
            "k_prefer": int(rr.k_prefer) if not pd.isna(rr.k_prefer) else 0,
            "k_excl": int(rr.k_excl) if not pd.isna(rr.k_excl) else 0,
            "k_frontier": int(rr.k_frontier) if not pd.isna(rr.k_frontier) else 0,
            "k_fend": int(rr.k_fend) if not pd.isna(rr.k_fend) else 0,

            "k_current": int(rr.k_current),
            "k_nodeprobe": int(rr.k_nodeprobe),
            "k_edgeprobe": int(rr.k_edgeprobe),
        })

    return pd.DataFrame(out_rows).sort_values(["iteration", "traj_id"]).reset_index(drop=True)


def plot_agenda_hit_histograms(
    df_hits: pd.DataFrame,
    out_path: str,
    *,
    window: tuple[int, int] | None = None,
):
    """
    Distributional view for long-tail / bimodality:
    - Histograms of hit fractions, not just the mean curves.

    Recommended for supplement or 1 compact main-fig:
      Panels (2x2):
        (1) hit_must_cur_frac
        (2) hit_prefer_cur_frac
        (3) hit_must_np_frac
        (4) hit_prefer_np_frac

    If window=(tmin,tmax) is provided, only uses those iterations (e.g., last-50).
    """
    df = df_hits.copy()
    if window is not None:
        tmin, tmax = window
        df = df[(df["iteration"] >= int(tmin)) & (df["iteration"] <= int(tmax))].copy()

    cols = [
        ("hit_must_cur_frac", "must → current_set (fraction)"),
        ("hit_prefer_cur_frac", "prefer → current_set (fraction)"),
        ("hit_must_np_frac", "must → node_probes (fraction)"),
        ("hit_prefer_np_frac", "prefer → node_probes (fraction)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(5, 3))
    axes = axes.reshape(-1)

    for ax, (c, title) in zip(axes, cols):
        x = df[c].dropna().astype(float).values
        ax.hist(x, bins=20)
        ax.set_title(title)
        ax.set_xlabel("fraction")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.2)

    if window is None:
        fig.suptitle("Agenda hit distributions (all iterations)", y=1.02)
    else:
        fig.suptitle(f"Agenda hit distributions (window [{window[0]}, {window[1]}])", y=1.02)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ==================================================
# NEW: Agenda funnel (controller→sampler→evidence)
# ==================================================
def plot_agenda_funnel(
    df_hits: pd.DataFrame,
    out_path: str,
    *,
    window: tuple[int, int],
):
    """
    MICCAI-friendly single-figure "execution funnel".

    For a chosen window (e.g., last-50 iterations), we summarize:
      Stage 1: agenda items present in current_set (sampler compliance)
      Stage 2: agenda items present in node_probes (evidence compliance)
      Stage 3: frontier edges executed in edge_probes (frontier-specific execution)

    We show MUST vs PREFER as two groups.

    This is usually much easier to read than many time-series curves.
    """
    tmin, tmax = window
    dfw = df_hits[(df_hits["iteration"] >= int(tmin)) & (df_hits["iteration"] <= int(tmax))].copy()

    # mean fractions in window
    must_cur = float(dfw["hit_must_cur_frac"].mean())
    must_np = float(dfw["hit_must_np_frac"].mean())
    pref_cur = float(dfw["hit_prefer_cur_frac"].mean())
    pref_np = float(dfw["hit_prefer_np_frac"].mean())
    fr_exec = float(dfw["executed_frontier_frac"].mean())

    # For frontier, feasibility is informative too
    fr_feas = float(dfw["feasible_frontier_frac"].mean())

    stages = ["→ current_set", "→ node_probes", "frontier exec"]
    must_vals = [must_cur, must_np, np.nan]  # frontier isn't "must" (agenda item type differs)
    pref_vals = [pref_cur, pref_np, np.nan]

    # plot as grouped bars + frontier bars
    fig, ax = plt.subplots(1, 1, figsize=(4, 2.1))

    x = np.arange(2)  # first two stages for must/prefer
    w = 0.35
    ax.bar(x - w/2, [must_cur, must_np], width=w, label="Must include")
    ax.bar(x + w/2, [pref_cur, pref_np], width=w, label="Prefer include")

    # frontier as separate pair at the end
    x2 = np.array([2.2, 3.2])
    ax.bar(x2[0], fr_feas, width=0.6, label="Frontier feasible")
    ax.bar(x2[1], fr_exec, width=0.6, label="Frontier executed")

    ax.set_xticks([0, 1, 2.2, 3.2])
    ax.set_xticklabels(["Must→cur", "Must→np / Prefer→np", "frontier feasible", "frontier exec"], rotation=0)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("mean fraction")
    ax.set_title(f"Agenda execution funnel (window [{tmin}, {tmax}])")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# NEW: Footprint compact summary (TopK overlap + rank corr ...)
# ============================================================
def _spearman_rank_corr_from_counts(sA: pd.Series, sB: pd.Series) -> float:
    """
    Compute Spearman rank correlation from two count Series (index=item).
    We rank items by count (descending). Items absent get count 0.

    This is paper-friendly:
      - 1 number describing global footprint similarity in the chosen window
      - Complementary to TopK overlap
    """
    if sA is None or sB is None:
        return np.nan
    all_items = list(set(sA.index) | set(sB.index))
    if len(all_items) < 3:
        return np.nan
    a = pd.Series({it: float(sA.get(it, 0.0)) for it in all_items})
    b = pd.Series({it: float(sB.get(it, 0.0)) for it in all_items})
    ra = a.rank(ascending=False, method="average")
    rb = b.rank(ascending=False, method="average")
    return float(ra.corr(rb, method="pearson"))


def compute_footprint_compact_metrics(
    serA: pd.Series,
    serB: pd.Series,
    *,
    topk: int = 30,
    n_show_delta: int = 6,
) -> dict:
    """
    Compute compact footprint comparison metrics for one footprint type:
      - TopK overlap Jaccard
      - Spearman rank correlation over union items
      - Largest absolute count deltas (for interpretability)

    Returns:
      dict with keys:
        topk_jaccard, spearman_rho, delta_items (list of tuples)
    """
    sA = serA.copy() if isinstance(serA, pd.Series) else pd.Series(dtype=float)
    sB = serB.copy() if isinstance(serB, pd.Series) else pd.Series(dtype=float)

    topA = set(sA.head(topk).index.tolist())
    topB = set(sB.head(topk).index.tolist())
    topk_j = jaccard(topA, topB)

    rho = _spearman_rank_corr_from_counts(sA, sB)

    all_items = list(set(sA.index) | set(sB.index))
    deltas = []
    for it in all_items:
        da = float(sA.get(it, 0.0))
        db = float(sB.get(it, 0.0))
        deltas.append((it, db - da, db, da))
    deltas.sort(key=lambda x: abs(x[1]), reverse=True)
    delta_items = deltas[:n_show_delta]

    return {
        "topk_jaccard": float(topk_j),
        "spearman_rho": float(rho) if not pd.isna(rho) else np.nan,
        "delta_items": delta_items,
    }


def plot_footprint_compact_table(
    metrics: dict,
    title: str,
    out_path: str,
):
    """
    Render one compact figure summarizing footprint similarity and key deltas.

    This is designed for MICCAI main text:
      - 1 small table per footprint type (node / edge)
      - Or combine 3 types into a single wider table (see wrapper below)
    """
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 2.6))
    ax.axis("off")

    topk_j = metrics.get("topk_jaccard", np.nan)
    rho = metrics.get("spearman_rho", np.nan)
    deltas = metrics.get("delta_items", [])

    # Build table rows
    rows = [
        ["TopK overlap (Jaccard)", f"{topk_j:.3f}" if not pd.isna(topk_j) else "-"],
        ["Rank corr (Spearman ρ)", f"{rho:.3f}" if not pd.isna(rho) else "-"],
        ["Δcount largest items", ""],
    ]
    for (it, d, b, a) in deltas:
        rows.append([f"  {it}", f"Δ={d:+.0f} (LLM={b:.0f}, Base={a:.0f})"])

    table = ax.table(
        cellText=rows,
        colLabels=["metric", "value"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(1, 1.2)

    ax.set_title(title, pad=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_footprint_compact_triplet(
    curA: pd.Series, curB: pd.Series,
    npA: pd.Series, npB: pd.Series,
    epA: pd.Series, epB: pd.Series,
    *,
    topk: int,
    out_path: str,
    window_title: str,
):
    """
    One figure, three rows: current_set / node_probes / edge_probes metrics.
    Replaces the 3 large bar charts with a single MICCAI-friendly summary table.

    Includes:
      - TopK Jaccard overlap
      - Spearman rho
      - top Δ items (very compact)

    Note:
      - You can put this in main paper; keep the large bar charts for supplementary.
    """
    m_cur = compute_footprint_compact_metrics(curA, curB, topk=topk)
    m_np  = compute_footprint_compact_metrics(npA,  npB,  topk=topk)
    m_ep  = compute_footprint_compact_metrics(epA,  epB,  topk=topk)

    def _fmt(m):
        j = m["topk_jaccard"]
        r = m["spearman_rho"]
        j = "-" if pd.isna(j) else f"{j:.3f}"
        r = "-" if pd.isna(r) else f"{r:.3f}"
        # show only 2 deltas to keep it compact
        ds = m["delta_items"][:2]
        dstr = "; ".join([f"{it}:Δ{d:+.0f}" for it, d, _, _ in ds]) if ds else "-"
        return j, r, dstr

    rows = []
    for name, m in [("current_set", m_cur), ("node_probes", m_np), ("edge_probes", m_ep)]:
        j, r, dstr = _fmt(m)
        rows.append([name, j, r, dstr])

    fig, ax = plt.subplots(1, 1, figsize=(9.0, 2.0))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["footprint", f"Top-{topk} Jaccard", "Spearman ρ", "top Δ items (2)"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(1, 1.25)

    ax.set_title(f"Baseline vs LLM footprint summary — {window_title}", pad=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ==========================================
# NEW MAIN ENTRY (keep old entry untouched)
# ==========================================
def run_paper_compact_views(
    out_root: str,
    method_a: str = "baseline_iters200",
    method_b: str = "Qwen_iters200",
    *,
    save_dir: Optional[str] = None,
    window_last: int = 50,
    topk: int = 15,
):
    """
    New entry point for compact, paper-oriented visualizations.

    It does NOT modify your existing run_path_and_agenda_analysis().
    Instead, it:
      1) loads logs
      2) produces:
         - trajectory embedding (Path)
         - agenda size table + hit-distribution hist + execution funnel (Agenda)
         - footprint compact summary table (Footprint)

    Suggested usage:
      - Keep your current 3 panels for debugging & supplement.
      - Use these compact outputs for MICCAI main text.
    """
    if save_dir is None:
        save_dir = os.path.join(out_root, "kg_path_agenda_paper_views")
    os.makedirs(save_dir, exist_ok=True)

    # ---- load + normalize
    df_raw_a = _load_iter_log(out_root, method_a)
    df_raw_b = _load_iter_log(out_root, method_b)
    dfA = normalize_iter_df(df_raw_a)
    dfB = normalize_iter_df(df_raw_b)

    # ---- window (last N iterations)
    t_max = int(max(dfA["iteration"].max(), dfB["iteration"].max()))
    t_min = max(0, t_max - int(window_last) + 1)
    window = (t_min, t_max)

    # ====================
    # (1) PATH: trajectory map
    # ====================
    traj_fig = os.path.join(save_dir, "path_trajectory_mds.png")
    plot_path_trajectory_embedding(
        dfA, dfB,
        set_col="current_set",
        agg="union",
        labelA="EGL",
        labelB="EGL-LLM",
        out_path=traj_fig,
    )
    print("Saved:", traj_fig)

    traj_fig_3d = os.path.join(save_dir, "path_trajectory_mds_3D.png")
    plot_path_trajectory_embedding_3d(
        dfA, dfB,
        set_col="current_set",
        agg="union",
        labelA="EGL",
        labelB="EGL-LLM",
        out_path=traj_fig_3d,
    )
    print("Saved:", traj_fig_3d)

    traj_fig_2d = os.path.join(save_dir, "path_trajectory_mds_2D_density.png")
    plot_path_trajectory_embedding_2d_density(
        dfA, dfB,
        set_col="current_set",
        agg="union",
        labelA="EGL",
        labelB="EGL-LLM",
        out_path=traj_fig_2d,
    )
    print("Saved:", traj_fig_2d)

    # ====================
    # (2) AGENDA: size distribution + hit distributions + funnel
    # ====================
    df_ctl_raw = _load_control_log(out_root, method_b)
    df_ctl = normalize_control_df(df_ctl_raw)

    # size distribution table
    df_size = compute_agenda_size_distribution(df_ctl)
    df_size.to_csv(os.path.join(save_dir, "agenda_size_distribution.csv"), index=False)
    size_fig = os.path.join(save_dir, "agenda_size_table.png")
    plot_agenda_size_table(df_size, size_fig)
    print("Saved:", size_fig)

    # per-row hit metrics (distributional)
    df_hits = compute_agenda_hit_rows(dfB, df_ctl)
    hits_csv = os.path.join(save_dir, "agenda_hit_rows.csv")
    df_hits.to_csv(hits_csv, index=False)
    print("Saved:", hits_csv)

    # hit distributions (window & full)
    hist_all = os.path.join(save_dir, "agenda_hit_hist_all.png")
    plot_agenda_hit_histograms(df_hits, hist_all, window=None)
    print("Saved:", hist_all)

    hist_win = os.path.join(save_dir, "agenda_hit_hist_window.png")
    plot_agenda_hit_histograms(df_hits, hist_win, window=window)
    print("Saved:", hist_win)

    # funnel (window)
    funnel_fig = os.path.join(save_dir, "agenda_execution_funnel.png")
    plot_agenda_funnel(df_hits, funnel_fig, window=window)
    print("Saved:", funnel_fig)

    # ====================
    # (3) FOOTPRINT: compact summary table (window)
    # ====================
    # Reuse your existing frequency functions
    curA = feature_frequency(dfA, "current_set", t_min=t_min, t_max=t_max)
    curB = feature_frequency(dfB, "current_set", t_min=t_min, t_max=t_max)
    npA = feature_frequency(dfA, "node_probe_set", t_min=t_min, t_max=t_max)
    npB = feature_frequency(dfB, "node_probe_set", t_min=t_min, t_max=t_max)
    epA = edge_frequency(dfA, "edge_probe_set", t_min=t_min, t_max=t_max)
    epB = edge_frequency(dfB, "edge_probe_set", t_min=t_min, t_max=t_max)

    fp_fig = os.path.join(save_dir, "footprint_compact_triplet.png")
    plot_footprint_compact_triplet(
        curA, curB, npA, npB, epA, epB,
        topk=topk,
        out_path=fp_fig,
        window_title=f"Window [{t_min}, {t_max}]",
    )
    print("Saved:", fp_fig)


def find_tstar_absorbing(
    df: pd.DataFrame,
    *,
    col: str,
    tau: float,
    allow_k: int = 0,          # allow dips below tau (but above floor)
    max_drop: float = 0.0,     # hard floor: tau - max_drop
    min_len: int = 30,
    iter_col: str = "iter",
) -> Optional[int]:
    """
    Two-tier absorbing stability point t*:

    - Hard floor: y must NEVER go below (tau - max_drop) after t.
    - Soft dips: allow up to allow_k points with y < tau after t.
    - Require tail length >= min_len.
    """
    if df is None or df.empty:
        return None
    if iter_col not in df.columns or col not in df.columns:
        raise KeyError(f"[t*] missing columns: {iter_col}, {col}")

    d = df.sort_values(iter_col).reset_index(drop=True)
    y = d[col].astype(float).to_numpy()
    n = int(y.size)
    if n < min_len:
        return None

    tau = float(tau)
    floor_thr = tau - float(max_drop)

    finite = np.isfinite(y)
    hard_bad = (~finite) | (y < floor_thr)   # strictly forbidden
    soft_bad = (~finite) | (y < tau)         # counted dips (includes hard_bad, but hard handled separately)

    hard_bad = hard_bad.astype(np.int32)
    soft_bad = soft_bad.astype(np.int32)

    # suffix counts
    suffix_hard = np.zeros(n + 1, dtype=np.int32)
    suffix_soft = np.zeros(n + 1, dtype=np.int32)
    for i in range(n - 1, -1, -1):
        suffix_hard[i] = suffix_hard[i + 1] + hard_bad[i]
        suffix_soft[i] = suffix_soft[i + 1] + soft_bad[i]

    last_t_allowed = n - min_len
    for t in range(0, last_t_allowed + 1):
        if suffix_hard[t] == 0 and suffix_soft[t] <= int(allow_k):
            return int(d.loc[t, iter_col])




def compute_basin_prototype_and_hitting_time(
    df_iter: pd.DataFrame,
    *,
    set_col: str = "current_set",
    # --- prototype construction ---
    t_min: Optional[int] = None,
    t_max: Optional[int] = None,
    proto_keep_q: float = 0.8,
    # --- basin membership / stability ---
    tau: float = 0.98,
    allow_k: int = 0,
    max_drop: float = 0.003,
    min_len: int = 30,
    # --- columns ---
    iter_col: str = "iteration",
    traj_col: str = "traj_id",
) -> dict:
    """
    Build a *basin prototype* set S* from a late window, compute basin-membership curves
    J(S_{t,r}, S*) per (iteration t, run r), summarize distribution (mean/p10/p50/p90),
    and compute an absorbing hitting time t* using your `find_tstar_absorbing()`.

    Why this is useful (paper-ready intuition)
    ------------------------------------------
    Your MDS shows both methods jump within the same "cloud". That can happen even if the
    process has reached an attractor basin: many micro-states (different current_set) can
    share a stable core, and Gibbs sampling mixes among them. To verify "same basin",
    we define a prototype S* and measure membership J(S_{t,r}, S*). If membership becomes
    high and *absorbing-stable* (with allowances), then being in the cloud is consistent
    with "basin mixing", not "failed exploration".

    Prototype definition
    --------------------
    Over a chosen window [t_min, t_max], treat each (t, r) as one sample.
    For every feature f, compute its occurrence rate:
        p(f) = #(samples where f in S_{t,r}) / #(samples)
    Then:
        S* = { f : p(f) >= proto_freq_thr }.

    Outputs
    -------
    Returns a dict with:
      - "prototype_set": set[str]
      - "proto_stats": DataFrame with per-feature frequency in the window
      - "membership_by_iter": DataFrame with columns:
            iter, mean, p10, p50, p90, n_runs
      - "t_star_absorbing": Optional[int]
            index in the sorted iterations list (same convention as find_tstar_absorbing)
      - "t_star_iteration": Optional[int]
            the actual iteration value corresponding to t_star_absorbing
      - "window": (t_min, t_max)
      - "meta": parameters used

    Notes
    -----
    - We compute membership per-run to preserve the distribution (you explicitly want long-tail/bimodality checks).
    - We use p50 (median) as a robust curve for finding t* (absorbing stability). You can change to mean if desired.
    """
    if df_iter is None or df_iter.empty:
        raise ValueError("df_iter is empty")

    if iter_col not in df_iter.columns or traj_col not in df_iter.columns:
        raise KeyError(f"missing columns: {iter_col}, {traj_col}")
    if set_col not in df_iter.columns:
        raise KeyError(f"missing set column: {set_col}")

    d = df_iter.copy()
    d = d.sort_values([iter_col, traj_col]).reset_index(drop=True)

    # choose default window = last 50 iterations (same spirit as your footprint window)
    iters_sorted = sorted(d[iter_col].unique().tolist())
    if len(iters_sorted) == 0:
        raise ValueError("no iterations found")

    if t_max is None:
        t_max = int(iters_sorted[-1])
    if t_min is None:
        t_min = int(max(iters_sorted[0], t_max - 50))

    w = d[(d[iter_col] >= int(t_min)) & (d[iter_col] <= int(t_max))].copy()
    if w.empty:
        raise ValueError(f"prototype window empty: [{t_min}, {t_max}]")

    # --------------------------
    # 1) Build prototype S*
    # --------------------------
    # Each row (t,r) is one sample
    n_samples = int(len(w))
    feat_counts: Dict[str, int] = {}
    for s in w[set_col].tolist():
        if not isinstance(s, set):
            # normalized tables should already have sets; be defensive
            s = set(s) if isinstance(s, (list, tuple)) else set()
        for f in s:
            feat_counts[f] = feat_counts.get(f, 0) + 1

    proto_df = pd.DataFrame({
        "feature": list(feat_counts.keys()),
        "count": list(feat_counts.values()),
    })
    proto_df["freq"] = proto_df["count"] / max(1, n_samples)
    proto_df = proto_df.sort_values("freq", ascending=False).reset_index(drop=True)

    thr = float(np.quantile(proto_df["freq"].astype(float).values, proto_keep_q))
    proto_set = set(proto_df.loc[proto_df["freq"].astype(float) >= thr, "feature"].tolist())

    # --------------------------
    # 2) Membership distribution per iteration
    # --------------------------
    # membership(t,r) = J(S_{t,r}, S*)
    def _j_to_proto(s: Any) -> float:
        if not isinstance(s, set):
            s = set(s) if isinstance(s, (list, tuple)) else set()
        return jaccard(s, proto_set)

    d["_membership"] = d[set_col].apply(_j_to_proto)

    # Per-iteration: keep distribution across runs (do NOT collapse to union over runs)
    rows = []
    for t, g in d.groupby(iter_col):
        vals = g["_membership"].astype(float).to_numpy()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            rows.append({"iter": int(t), "mean": np.nan, "p10": np.nan, "p50": np.nan, "p90": np.nan, "n_runs": 0})
            continue
        rows.append({
            "iter": int(t),
            "mean": float(np.mean(vals)),
            "p10": float(np.quantile(vals, 0.10)),
            "p50": float(np.quantile(vals, 0.50)),
            "p90": float(np.quantile(vals, 0.90)),
            "n_runs": int(vals.size),
        })

    mem = pd.DataFrame(rows).sort_values("iter").reset_index(drop=True)

    # --------------------------
    # 3) Absorbing hitting time t*
    # --------------------------
    # We use p50 (median) for robustness. If you prefer mean, set col="mean".
    t_star_idx = find_tstar_absorbing(
        mem,
        col="p50",
        tau=float(tau),
        allow_k=int(allow_k),
        max_drop=float(max_drop),
        min_len=int(min_len),
        iter_col="iter",
    )

    if t_star_idx is None:
        t_star_iter = None
    else:
        # IMPORTANT: find_tstar_absorbing returns an index in the *sorted samples*,
        # because it scans from t=0..last_t_allowed. Here mem is already sorted by iter.
        # So we map index -> actual iteration value.
        if 0 <= int(t_star_idx) < len(mem):
            t_star_iter = int(mem.loc[int(t_star_idx), "iter"])
        else:
            t_star_iter = None

    return {
        "prototype_set": proto_set,
        "proto_stats": proto_df,
        "membership_by_iter": mem,
        "t_star_absorbing": t_star_idx,
        "t_star_iteration": t_star_iter,
        "window": (int(t_min), int(t_max)),
        "meta": {
            "set_col": set_col,
            "proto_keep_q": float(proto_keep_q),
            "tau": float(tau),
            "allow_k": int(allow_k),
            "max_drop": float(max_drop),
            "min_len": int(min_len),
            "iter_col": iter_col,
            "traj_col": traj_col,
        },
    }


def plot_basin_membership_curves(
    resA: dict,
    resB: dict,
    *,
    labelA: str = "EGL",
    labelB: str = "EGL-LLM",
    out_path: str = "basin_membership_curves.png",
    # cosmetics
    show_band: bool = True,
    band_q: tuple[float, float] = (0.10, 0.90),
    show_proto_overlap: bool = True,
):
    """
    Plot basin-membership curves for two methods and annotate absorbing t*.

    Expected inputs
    ---------------
    resA/resB should be outputs from `compute_basin_prototype_and_hitting_time()`.

    What the plot communicates (paper-ready)
    ----------------------------------------
    - If both methods lie in the same attractor basin, their membership to their own
      prototypes should become high and absorbing-stable (vertical line at t*).
    - If the MDS “cloud overlap” is due to shared basin, we should also see high overlap
      between prototypes S*_A and S*_B (reported in title or legend).

    Plotted lines
    -------------
    For each method:
      - mean membership (solid line)
      - median membership p50 (dashed line)
      - optional band [p10, p90] (shaded) to reveal long-tail / bimodality-like spread

    Annotations
    -----------
      - vertical line at t_star_iteration (if exists) for each method.

    Notes
    -----
    - We keep this as a single, compact figure suitable for MICCAI constraints.
    - We avoid extra panels; distribution info is shown via the quantile band.
    """
    mA = resA["membership_by_iter"].copy()
    mB = resB["membership_by_iter"].copy()

    # sanity
    for need in ["iter", "mean", "p50", "p10", "p90"]:
        if need not in mA.columns or need not in mB.columns:
            raise KeyError(f"membership_by_iter missing columns; need {need}")

    # Prototype overlap (helps interpret “same cloud” as “same basin”)
    protoA = resA.get("prototype_set", set())
    protoB = resB.get("prototype_set", set())
    proto_j = jaccard(set(protoA), set(protoB)) if (protoA or protoB) else np.nan

    # Build figure
    fig, ax = plt.subplots(1, 1, figsize=(7, 3))

    # Helper to draw one method
    def _draw(m: pd.DataFrame, label: str, color_hint: Optional[str] = None):
        x = m["iter"].to_numpy()
        y_mean = m["mean"].to_numpy()
        y_med = m["p50"].to_numpy()

        # Matplotlib default color cycle is fine; we don't hardcode colors here.
        line_med = ax.plot(x, y_med, marker="o", markersize=1.5, label=f"{label}: median")[0]
        # ax.plot(x, y_med, linestyle="--", marker=None, label=f"{label}: median (p50)", color=line_mean.get_color())

        if show_band:
            qlo, qhi = band_q
            # we stored p10/p90; if band_q != (0.10,0.90) you can adapt upstream
            y_lo = m["p10"].to_numpy()
            y_hi = m["p90"].to_numpy()
            ax.fill_between(x, y_lo, y_hi, alpha=0.15, color=line_med.get_color(), label=f"{label}: p10-p90")

        return line_med.get_color()

    colA = _draw(mA, labelA)
    colB = _draw(mB, labelB)

    # Mark absorbing t*
    tA = resA.get("t_star_iteration", None)
    tB = resB.get("t_star_iteration", None)

    if tA is not None:
        ax.axvline(int(tA), linestyle=":", linewidth=1.0, color=colA)
        ax.text(int(tA), 1.02, f"{labelA} t*", color=colA, ha="center", va="bottom")
    if tB is not None:
        ax.axvline(int(tB), linestyle=":", linewidth=1.0, color=colB)
        ax.text(int(tB), 1.02, f"{labelB} t*", color=colB, ha="center", va="bottom")

    # Titles / labels
    title = "Basin membership vs iteration (Jaccard to late-stage prototype)"
    if show_proto_overlap and np.isfinite(proto_j):
        title += f" | prototype overlap J={proto_j:.3f}"
    ax.set_title(title)

    # show prototype window (helps interpret what S* means)
    wA = resA.get("window", None)
    wB = resB.get("window", None)
    if wA is not None and wB is not None:
        ax.set_xlabel(f"Iteration (prototype windows: {labelA}[{wA[0]},{wA[1]}], {labelB}[{wB[0]},{wB[1]}])")
    else:
        ax.set_xlabel("Iteration")

    ax.set_ylabel("Membership: Jaccard(current_set, prototype_set)")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.25)

    # Legend: keep manageable (mean/median + band)
    ax.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main_paper_views(out_root: str = "outputs", method_a: str | None = None, method_b: str | None = None, save_dir: str | None = None, window_last: int = 50, topk: int = 15):
    """
    A separate main so you can keep the original __main__ entry for debugging plots.

    In your script:
      - keep your existing __main__ that calls run_path_and_agenda_analysis()
      - add another __main__ block or a flag to call main_paper_views()
    """
    # method_a = "Baseline_Status_iters500_BaseS_2"
    # method_b = "Qwen_Status_iters500_BaseS_useP_useDB_2"
    if method_a is None:
        method_a = "Baseline_Prediction_iters500_BaseS_1"
    if method_b is None:
        method_b = "Qwen_Prediction_iters500_BaseS_useP_useDB_1"
    if save_dir is None:
        save_dir = os.path.join("outputs", f"Figures_{method_a}_vs_{method_b}", "kg_path_agenda_v2")

    run_paper_compact_views(
        out_root=out_root,
        method_a=method_a,
        method_b=method_b,
        save_dir=save_dir,
        window_last=window_last,
        topk=topk,
    )

    # --- absorbing-stability params ---
    tau_node: float = 0.98
    allow_k_node: int = 0
    max_drop_node: float = 0.003
    min_len: int = 30
    # --- basin prototype params ---
    proto_window_len: int = 50
    proto_keep_q: float = 0.8

    # ------------------------------------------------------------
    # 0) Load + normalize (same as your existing analysis entry)
    # ------------------------------------------------------------
    df_raw_a = _load_iter_log(out_root, method_a)
    df_raw_b = _load_iter_log(out_root, method_b)
    dfA = normalize_iter_df(df_raw_a)
    dfB = normalize_iter_df(df_raw_b)

    print("Loaded iteration logs:")
    print(" - A:", method_a, "from", df_raw_a["_source_path"].iloc[0], "rows=", len(dfA))
    print(" - B:", method_b, "from", df_raw_b["_source_path"].iloc[0], "rows=", len(dfB))

    # ------------------------------------------------------------
    # 1) Basin prototype + absorbing hitting time (NEW)
    # ------------------------------------------------------------
    # Use the same "late window" idea as your footprint: last proto_window_len iterations.
    tmaxA = int(dfA["iteration"].max())
    tmaxB = int(dfB["iteration"].max())
    tmax = int(max(tmaxA, tmaxB))
    tmin = int(max(0, tmax - int(proto_window_len) + 1))

    # Baseline basin
    res_basin_A = compute_basin_prototype_and_hitting_time(
        dfA,
        set_col="current_set",
        t_min=tmin,
        t_max=tmax,
        proto_keep_q=float(proto_keep_q),
        tau=float(tau_node),
        allow_k=int(allow_k_node),
        max_drop=float(max_drop_node),
        min_len=int(min_len),
        iter_col="iteration",
        traj_col="traj_id",
    )

    # LLM basin
    res_basin_B = compute_basin_prototype_and_hitting_time(
        dfB,
        set_col="current_set",
        t_min=tmin,
        t_max=tmax,
        proto_keep_q=float(proto_keep_q),
        tau=float(tau_node),
        allow_k=int(allow_k_node),
        max_drop=float(max_drop_node),
        min_len=int(min_len),
        iter_col="iteration",
        traj_col="traj_id",
    )

    # Save prototype tables (optional but useful for writing/appendix)
    res_basin_A["proto_stats"].to_csv(os.path.join(save_dir, "basin_proto_stats_baseline.csv"), index=False)
    res_basin_B["proto_stats"].to_csv(os.path.join(save_dir, "basin_proto_stats_llm.csv"), index=False)
    res_basin_A["membership_by_iter"].to_csv(os.path.join(save_dir, "basin_membership_by_iter_baseline.csv"), index=False)
    res_basin_B["membership_by_iter"].to_csv(os.path.join(save_dir, "basin_membership_by_iter_llm.csv"), index=False)

    print("Basin t* (Baseline):", res_basin_A["t_star_iteration"], " | window:", res_basin_A["window"])
    print("Basin t* (LLM):     ", res_basin_B["t_star_iteration"], " | window:", res_basin_B["window"])

    # Plot one compact, paper-friendly figure
    basin_fig = os.path.join(save_dir, "basin_membership_curves.png")
    plot_basin_membership_curves(
        res_basin_A,
        res_basin_B,
        labelA="EGL",
        labelB="EGL-LLM",
        out_path=basin_fig,
        show_band=True,          # shows p10–p90 => long-tail / bimodality hints
        band_q=(0.10, 0.90),
        show_proto_overlap=True, # prints prototype overlap J in title
    )
    print("Saved:", basin_fig)

    print("\nDone. Paper-view outputs in:", save_dir)


def main_plot_v1():
    out_root = "outputs"
    # method_a = "Baseline_Status_iters500_BaseS_2"
    # method_b = "Qwen_Status_iters500_BaseS_useP_useDB_2"

    method_a = "Baseline_Prediction_iters500_BaseS_1"
    method_b = "Qwen_Prediction_iters500_BaseS_useP_useDB_1"

    # If you know the intended P_EDGE, set it here. Otherwise set None.
    P_EDGE = 30
    save_dir = os.path.join("outputs", f"Figures_{method_a}_vs_{method_b}", "kg_path_agenda_v1")

    run_path_and_agenda_analysis(
        out_root=out_root,
        method_a=method_a,
        method_b=method_b,
        save_dir=save_dir,
        P_EDGE=P_EDGE,
        topk=15,
    )

if __name__ == "__main__":
    from plot_all_from_config import main
    main(default_sections=["path_agenda"])

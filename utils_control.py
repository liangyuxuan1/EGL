from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set, Deque, TYPE_CHECKING
from collections import deque, Counter
import json
import math
import numpy as np
import random
import pandas as pd

if TYPE_CHECKING:
    from utils_dataset import DatasetCtx
    from utils_cache import EvalCacheSQLite
    from utils_graph import KnowledgeGraph

import utils
import utils_cache

# ================================================================
# Dynamic threshold/floor computation
# ================================================================
@dataclass
class ThresholdQueryResult:
    """
    Generic query result for dynamic thresholds / floors.
    threshold: the numeric threshold/floor value
    source: which bucket was used: obj_phase | phase | global | fixed
    n_used: number of samples used to compute it (for sanity/debug)
    """
    threshold: float
    source: str
    n_used: int


def _robust_thr_mad(xs: List[float], k: float = 2.0) -> float:
    """
    Robust "upper" threshold using MAD:
      thr = median(xs) + k * MAD(xs) * 1.4826
    Used for "unstable" style thresholds (e.g., auc_std).
    """
    arr = np.asarray(xs, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    # 1.4826 makes MAD consistent with std under normality
    thr = med + float(k) * mad * 1.4826
    return float(thr)


def _robust_floor_quantile(xs: List[float], q: float) -> float:
    """
    Robust "lower floor" using quantile:
      floor = quantile(xs, q)
    Used for spec/sens/prec floors as anomaly detectors ("too low" region).
    """
    arr = np.asarray(xs, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    # np.quantile is stable for rolling distributions
    v = float(np.quantile(arr, float(q)))
    return float(v)


@dataclass
class OnlineThresholdState:
    """
    Maintain rolling metric distributions and provide dynamic thresholds/floors.

    Design goals:
    - Avoid hard-coded floors/thresholds that are dataset-dependent.
    - Provide fallback to avoid sparse buckets: (global) -> fixed.
    - Warm-up calibration stage should NOT depend on objective:
        during warm-up, we only update the GLOBAL buckets.

    Supported:
    - AUC std threshold (MAD-based upper threshold) -> for "unstable".
    - Floors for specificity / sensitivity / precision (quantile-based lower floors)
      -> for "too low" anomaly detection.

    Notes:
    - Floors are NOT "targets". They are "abnormally low" cutoffs.
    - Precision is prevalence-sensitive; quantile floors are much safer than c_ppv*pi.
    """

    # minimum samples required to trust a bucket
    n_min: int = 30

    # history caps (rolling windows)
    cap_global: int = 400

    # auc_std robust hyperparam (like "2*std")
    mad_k: float = 2.0

    # quantile used for floors (lower tail)
    floor_q: float = 0.10  # 10% percentile

    # cold-start fallbacks (when all buckets are too small)
    fixed_auc_std: float = 0.0123
    fixed_spec_floor: float = 0.6815
    fixed_sens_floor: float = 0.5867
    fixed_prec_floor: float = 0.3149

    # internal storages: AUC std
    _auc_std_global: Deque[float] = field(default_factory=lambda: deque(maxlen=400))

    # internal storages: specificity floors
    _spec_global: Deque[float] = field(default_factory=lambda: deque(maxlen=400))

    # internal storages: sensitivity floors
    _sens_global: Deque[float] = field(default_factory=lambda: deque(maxlen=400))

    # internal storages: precision floors
    _prec_global: Deque[float] = field(default_factory=lambda: deque(maxlen=400))

    def __post_init__(self) -> None:
        # enforce caps for rolling deques
        self._auc_std_global = deque(self._auc_std_global, maxlen=int(self.cap_global))
        self._spec_global    = deque(self._spec_global,    maxlen=int(self.cap_global))
        self._sens_global    = deque(self._sens_global,    maxlen=int(self.cap_global))
        self._prec_global    = deque(self._prec_global,    maxlen=int(self.cap_global))

        # sanitize quantile
        self.floor_q = float(np.clip(float(self.floor_q), 0.01, 0.49))

    # ---------------------------
    # Update API
    # ---------------------------
    def update_auc_std(self, auc_std: float) -> None:
        """
        Ingest one auc_std sample.

        Always updates:
          - global
        """
        v = utils.safe_float(auc_std)
        if not np.isfinite(v):
            return

        self._auc_std_global.append(v)

    def update_perf_floors(
        self,
        *,
        specificity: Any,
        sensitivity: Any,
        precision: Any,
    ) -> None:
        """
        Ingest one set of (spec, sens, prec) samples.

        Important: these are distributions for "too low" anomaly detection.
        """
        spec = utils.safe_float(specificity)
        sens = utils.safe_float(sensitivity)
        prec = utils.safe_float(precision)

        if np.isfinite(spec):
            self._spec_global.append(float(np.clip(spec, 0.0, 1.0)))
        if np.isfinite(sens):
            self._sens_global.append(float(np.clip(sens, 0.0, 1.0)))
        if np.isfinite(prec):
            self._prec_global.append(float(np.clip(prec, 0.0, 1.0)))

    def update_from_metrics_summary(
        self,
        metrics_summary: Dict[str, Any],
        *,
        keys: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Convenience update from your evaluate_auc_and_more_cv() outputs.

        Default key mapping assumes metrics_summary contains:
          - "auc_std"
          - "specificity_mean"
          - "sensitivity_mean"
          - "precision_mean"

        If your keys differ, pass keys={"auc_std": "...", "spec": "...", "sens": "...", "prec": "..."}.
        """
        keys = keys or {}
        k_auc_std = keys.get("auc_std", "auc_std")
        k_spec    = keys.get("spec", "specificity_mean")
        k_sens    = keys.get("sens", "sensitivity_mean")
        k_prec    = keys.get("prec", "precision_mean")

        self.update_auc_std(metrics_summary.get(k_auc_std, np.nan))
        self.update_perf_floors(
            specificity=metrics_summary.get(k_spec, np.nan),
            sensitivity=metrics_summary.get(k_sens, np.nan),
            precision=metrics_summary.get(k_prec, np.nan),
        )

    # ---------------------------
    # Query API: auc_std threshold
    # ---------------------------
    def get_auc_std_threshold(self) -> ThresholdQueryResult:
        """
        Hierarchical fallback:
          1) global
          2) fixed

        Returns ThresholdQueryResult(threshold, source, n_used).
        """
        # 1) global
        xs = list(self._auc_std_global)
        if len(xs) >= int(self.n_min):
            thr = _robust_thr_mad(xs, k=float(self.mad_k))
            if np.isfinite(thr):
                return ThresholdQueryResult(threshold=float(thr), source="global", n_used=len(xs))

        # 2) fixed
        return ThresholdQueryResult(threshold=float(self.fixed_auc_std), source="fixed", n_used=len(xs))

    # ---------------------------
    # Query API: floors (spec/sens/prec)
    # ---------------------------
    def _get_floor_generic(
        self,
        *,
        global_bucket: Deque[float],
        fixed_floor: float,
    ) -> ThresholdQueryResult:
        """
        Internal shared implementation for spec/sens/prec floors.
        floor = quantile(xs, floor_q), with hierarchical fallback.
        """
        # 1) global
        xs = list(global_bucket)
        if len(xs) >= int(self.n_min):
            fl = _robust_floor_quantile(xs, q=float(self.floor_q))
            if np.isfinite(fl):
                return ThresholdQueryResult(threshold=float(np.clip(fl, 0.0, 1.0)), source="global", n_used=len(xs))

        # 2) fixed
        return ThresholdQueryResult(threshold=float(np.clip(fixed_floor, 0.0, 1.0)), source="fixed", n_used=len(xs))

    def get_spec_floor(self) -> ThresholdQueryResult:
        return self._get_floor_generic(
            global_bucket=self._spec_global,
            fixed_floor=float(self.fixed_spec_floor),
        )

    def get_sens_floor(self) -> ThresholdQueryResult:
        return self._get_floor_generic(
            global_bucket=self._sens_global,
            fixed_floor=float(self.fixed_sens_floor),
        )

    def get_prec_floor(self) -> ThresholdQueryResult:
        return self._get_floor_generic(
            global_bucket=self._prec_global,
            fixed_floor=float(self.fixed_prec_floor),
        )

    # ---------------------------
    # Optional: debug snapshot
    # ---------------------------
    def debug_counts(self) -> Dict[str, int]:
        """
        Quick counts for sanity checking / logging.
        """
        return {
            "auc_std_global": len(self._auc_std_global),
            "spec_global": len(self._spec_global),
            "sens_global": len(self._sens_global),
            "prec_global": len(self._prec_global),
        }

# -----------------------------
# Reward
# -----------------------------
@dataclass
class RewardWeights:
    # 主项：结构确认（直接推动 theta/phi）
    w_confirm_node: float = 1.0
    w_confirm_edge: float = 1.0

    # 次项：补证据（建议 0.05~0.2）
    w_fill_node: float = 0.10
    w_fill_edge: float = 0.10

    # 很弱的性能锚（可设 0；若保留，建议 <=0.1）
    # 20260128: 不要使用， 和confirm重复， 因为confirm就是根据performance计算的
    w_perf_auc: float = 0.0

    # OTS 约束惩罚权重，约束 / 门控（越小越好 -> 通过 penalty 实现）
    w_unstable: float = 3.0
    w_spec_floor: float = 4.0
    w_sens_floor: float = 4.0
    w_prec_floor: float = 4.0

    # node fill gate：|t| 超过该值才更积极补证据
    t0_node_fill: float = 1.0

    # edge fill gate：|t| 超过该值才更积极补证据
    t0_edge_fill: float = 1.5

    # 复杂度弱惩罚（可设0）
    # 需要根据cost_size的定义调整
    # 如果size cost = log1p(10*k/F), w = 0.2~0.4
    # 如果size cost 定义为相对值，w = 5~20

    # 如果 20~30 个 iteration 后 mean k 还在 >30：调到 0.5
    # 如果 k 被压得太狠、AUC 掉很多：降到 0.25
    w_size: float = 0.35 #12.0
    size_alpha: float = 10.0

    # 在不平衡数据上，性能指标采用AUC和AUPRC的加权平均
    # score = perf_alpha * auc_mean + (1-perf_alpha) * auprc_mean
    prevalence: float = 0.18  # 18% 阳性率
    perf_alpha: float = 0.4   # auprc权重更大一些


def compute_performance_score(auc: float, auprc: float, weights: RewardWeights):
    """
    因为auc和auprc的数量级不同, 先归一化, 再加权
    •	AUC 随机基线 ≈ 0.5
	•	AuPRC 随机基线 ≈ prevalence (≈0.18)
    """
    alpha = weights.perf_alpha
    pi = weights.prevalence               # 0.18，可从数据计算或配置给定
    auc_impr = (auc - 0.5) / 0.5          # roughly in [-1, 1]
    pr_impr  = (auprc - pi) / max(1e-6, 1 - pi)  # roughly in [-1, 1]

    perf_score = alpha * auc_impr + (1-alpha) * pr_impr

    return perf_score


def compute_reward(
    *,
    metrics_summary: Dict[str, Any],
    selected_set: List[str],          # base_set
    kg: Any,
    weights: RewardWeights,
    ots: OnlineThresholdState,
    pool_meta: pd.DataFrame,
    node_probes: List[Dict[str, Any]],
    edge_probes: List[Dict[str, Any]],
) -> Tuple[float, Dict[str, Any]]:
    """
    Reward for one iteration under *shared paired probes*.

    Paired-probe design assumptions (NEW, per your latest decision):
    - Node probe is balanced by construction:
        y_with    = score(base_set)                (contains u)
        y_without = score(base_set \\ {u})
        Δ_u = y_with - y_without
    - Edge probe is balanced by construction:
        each edge probe yields a full 2x2 block under same background,
        so cells 00/10/01/11 are always available (NO missing-cell handling).

    Reward design:
    - confirm: exploitation term, per probed object:
        tanh(|t|) * trust_after
    - fill: exploration/evidence-gain term (prevents passive probing):
        trust_gain * (0.3 + 0.7 * gate)
      where:
        trust_gain = max(0, trust_after - trust_before)
        gate = sigmoid(|t| - t0_fill)

    Note:
    - confirm/fill are computed ONLY on objects probed this iteration.
    - perf_term / barrier / size_cost remain global for the evaluated base_set.
    """

    # -------------------------
    # sanitize selected_set (base set)
    # -------------------------
    S = [f for f in selected_set if isinstance(f, str)]
    S = [f.strip() for f in S if f and f.strip()]
    if hasattr(kg, "_all_set"):
        S = [f for f in S if f in kg._all_set]
    Sset = sorted(set(S))
    k = len(Sset)

    # -------------------------
    # 0) domain/type composition from pool_meta (proxy dominance observables)
    # -------------------------
    meta_sub = pool_meta.loc[
        pool_meta["var_name"].isin(Sset),
        ["var_name", "clinical_domain"],
    ].copy()

    n_meta_hit = int(meta_sub["var_name"].nunique())
    n_unknown = int(k - n_meta_hit)

    meta_sub["clinical_domain"] = meta_sub["clinical_domain"].fillna("unknown").astype(str)
    dom_counts = meta_sub["clinical_domain"].value_counts(dropna=False).to_dict()

    if n_unknown > 0:
        dom_counts["unknown_not_in_meta"] = dom_counts.get("unknown_not_in_meta", 0) + n_unknown

    den = float(max(1, k))
    dom_ratios = {d: float(c) / den for d, c in dom_counts.items()}

    if dom_ratios:
        dom_top, dom_top_ratio = max(dom_ratios.items(), key=lambda kv: kv[1])
        ent = 0.0
        for p in dom_ratios.values():
            if p > 1e-12:
                ent += -p * math.log(p)
        ent_norm = float(ent / max(1e-12, math.log(max(2, len(dom_ratios)))))
    else:
        dom_top, dom_top_ratio, ent_norm = "unknown", 0.0, 0.0

    # -------------------------
    # 1) metrics
    # -------------------------
    auc_mean    = utils.as_float(metrics_summary, "auc_mean")
    auc_std     = utils.as_float(metrics_summary, "auc_std")
    auprc_mean  = utils.as_float(metrics_summary, "auprc_mean")
    auprc_std   = utils.as_float(metrics_summary, "auprc_std")
    spec        = utils.as_float(metrics_summary, "specificity_mean")
    spec_std    = utils.as_float(metrics_summary, "specificity_std")
    sens        = utils.as_float(metrics_summary, "sensitivity_mean")
    sens_std    = utils.as_float(metrics_summary, "sensitivity_std")
    prec        = utils.as_float(metrics_summary, "precision_mean")
    prec_std    = utils.as_float(metrics_summary, "precision_std")

    # -------------------------
    # 2) OTS fixed thresholds (NO fallback)
    # -------------------------
    thr_auc_std = float(ots.fixed_auc_std)
    floor_spec  = float(ots.fixed_spec_floor)
    floor_sens  = float(ots.fixed_sens_floor)
    floor_prec  = float(ots.fixed_prec_floor)

    pen_unstable = max(0.0, auc_std - thr_auc_std)
    pen_spec     = max(0.0, floor_spec - spec)
    pen_sens     = max(0.0, floor_sens - sens)
    pen_prec     = max(0.0, floor_prec - prec)

    barrier = (
        weights.w_unstable       * pen_unstable
        + weights.w_spec_floor   * pen_spec
        + weights.w_sens_floor   * pen_sens
        + weights.w_prec_floor   * pen_prec
    )

    # barrier trigger diagnostics (CSV-friendly)
    hit_unstable = 1 if pen_unstable > 0 else 0
    hit_spec     = 1 if pen_spec > 0 else 0
    hit_sens     = 1 if pen_sens > 0 else 0
    hit_prec     = 1 if pen_prec > 0 else 0
    hit_any      = 1 if (hit_unstable or hit_spec or hit_sens or hit_prec) else 0

    contribs = {
        "unstable": weights.w_unstable     * pen_unstable,
        "spec":     weights.w_spec_floor   * pen_spec,
        "sens":     weights.w_sens_floor   * pen_sens,
        "prec":     weights.w_prec_floor   * pen_prec,
    }
    dom_name, dom_val = "none", 0.0
    for name, val in contribs.items():
        if val > dom_val + 1e-12:
            dom_name, dom_val = name, float(val)

    # -------------------------
    # helpers
    # -------------------------
    def _q(xs: List[float], q: float) -> float:
        if not xs:
            return 0.0
        arr = np.asarray(xs, dtype=float)
        return float(np.quantile(arr, q, method="nearest"))

    # -------------------------
    # 3) Parse probes (STRICT to your schemas)
    # -------------------------
    # node_probes: {"var": u, "score_with": base_score, "score_without": sc}
    node_probe_counts = Counter()
    for p in (node_probes or []):
        u = p.get("var", "")
        if isinstance(u, str):
            u = u.strip()
            if u:
                node_probe_counts[u] += 1
    probed_nodes = sorted(node_probe_counts.keys())

    # edge_probes: {"u": u, "v": v, "cells": {"00":..,"10":..,"01":..,"11":..}}
    edge_probe_counts = Counter()
    for p in (edge_probes or []):
        u = p.get("u", "")
        v = p.get("v", "")
        if not (isinstance(u, str) and isinstance(v, str)):
            continue
        u, v = u.strip(), v.strip()
        if not u or not v:
            continue
        if u > v:
            u, v = v, u
        edge_probe_counts[(u, v)] += 1
    probed_edges = sorted(edge_probe_counts.keys())

    # -------------------------
    # 4) CONFIRM + FILL (ONLY probed objects)
    #     confirm = tanh(|t|) * trust_after
    #     fill    = trust_gain * (0.3 + 0.7*gate)
    #       trust_gain = max(0, trust_after - trust_before)
    #
    # trust_before approximation:
    #   - since each probe contributes exactly 1 paired sample to the stat
    #   - n_add_this_iter = #probes for that object in this iteration
    #   - support_before ≈ support_after - n_add_this_iter
    # -------------------------

    # ---- Nodes ----
    node_vals = []
    node_fill_vals = []
    tau_node = float(kg.update_cfg.tau_node)
    min_w_node = float(getattr(kg.update_cfg, "trust_min_w_node", 0.02))

    node_t_list = []
    node_support_list = []
    node_trust_after_list = []
    node_trust_gain_list = []
    node_gate_list = []

    for u in probed_nodes:
        st = kg.node_stat.get(u, None)
        if st is None:
            continue

        support_after = float(st.n_pair())
        trust_after = utils.trust_weight(support_after, tau_node, min_w=min_w_node)

        n_add = float(node_probe_counts.get(u, 0))
        support_before = max(0.0, support_after - n_add)
        trust_before = utils.trust_weight(support_before, tau_node, min_w=min_w_node)

        trust_gain = max(0.0, trust_after - trust_before)

        t = float(st.effect_confidence())  # |t|
        gate = float(utils.sigmoid(t - weights.t0_node_fill))

        node_vals.append(math.tanh(t) * trust_after)
        node_fill_vals.append(trust_gain * (0.3 + 0.7 * gate))

        node_t_list.append(t)
        node_support_list.append(support_after)
        node_trust_after_list.append(trust_after)
        node_trust_gain_list.append(trust_gain)
        node_gate_list.append(gate)

    node_confirm = float(np.mean(node_vals)) if node_vals else 0.0
    node_fill = float(np.mean(node_fill_vals)) if node_fill_vals else 0.0

    # ---- Edges ----
    edge_vals = []
    edge_fill_vals = []
    tau_edge = float(kg.update_cfg.tau_edge)
    min_w_edge = float(getattr(kg.update_cfg, "trust_min_w_edge", 0.02))

    edge_t_list = []
    edge_support_list = []
    edge_trust_after_list = []
    edge_trust_gain_list = []
    edge_gate_list = []

    for (u, v) in probed_edges:
        est = kg.edge_stat.get((u, v), None)
        if est is None:
            continue

        support_after = float(est.n_pair())
        trust_after = utils.trust_weight(support_after, tau_edge, min_w=min_w_edge)

        n_add = float(edge_probe_counts.get((u, v), 0))
        support_before = max(0.0, support_after - n_add)
        trust_before = utils.trust_weight(support_before, tau_edge, min_w=min_w_edge)

        trust_gain = max(0.0, trust_after - trust_before)

        t = float(est.interaction_confidence())  # |t|
        gate = float(utils.sigmoid(t - weights.t0_edge_fill))

        edge_vals.append(math.tanh(t) * trust_after)
        edge_fill_vals.append(trust_gain * (0.3 + 0.7 * gate))

        edge_t_list.append(t)
        edge_support_list.append(support_after)
        edge_trust_after_list.append(trust_after)
        edge_trust_gain_list.append(trust_gain)
        edge_gate_list.append(gate)

    edge_confirm = float(np.mean(edge_vals)) if edge_vals else 0.0
    edge_fill = float(np.mean(edge_fill_vals)) if edge_fill_vals else 0.0

    confirm_term = weights.w_confirm_node * node_confirm + weights.w_confirm_edge * edge_confirm
    fill_term    = weights.w_fill_node * node_fill + weights.w_fill_edge * edge_fill

    # -------------------------
    # 5) Weak performance anchor (bounded, no fallback)
    # -------------------------
    perf_score = compute_performance_score(auc_mean, auprc_mean, weights)
    perf_term = weights.w_perf_auc * float(perf_score)

    # -------------------------
    # 6) Weak complexity cost
    # -------------------------
    F_total = int(len(getattr(kg, "all_features", []) or [])) or max(1, k)
    frac = float(k) / max(1.0, float(F_total))
    size_cost = math.log1p(float(weights.size_alpha) * frac)

    # -------------------------
    # 7) Final reward (raw + squashed)
    # -------------------------
    R_raw = confirm_term + fill_term + perf_term - barrier - weights.w_size * float(size_cost)
    R = math.tanh(R_raw)

    # -------------------------
    # 8) Debug dict (all scalar, CSV-friendly)
    # -------------------------
    dbg: Dict[str, Any] = {
        # reward
        "R": float(R),
        "R_raw": float(R_raw),

        # selection / complexity
        "n_features": int(k),
        "n_features_total": int(F_total),
        "size_cost": float(size_cost),

        # components
        "node_confirm": float(node_confirm),
        "edge_confirm": float(edge_confirm),
        "node_fill": float(node_fill),
        "edge_fill": float(edge_fill),
        "confirm_term": float(confirm_term),
        "fill_term": float(fill_term),
        "perf_score": float(perf_score),
        "perf_term": float(perf_term),

        # metrics
        "auc_mean": float(auc_mean),
        "auc_std": float(auc_std),
        "auprc_mean": float(auprc_mean),
        "auprc_std": float(auprc_std),
        "specificity_mean": float(spec),
        "specificity_std": float(spec_std),
        "sensitivity_mean": float(sens),
        "sensitivity_std": float(sens_std),
        "precision_mean": float(prec),
        "precision_std": float(prec_std),

        # OTS thresholds
        "thr_auc_std": float(thr_auc_std),
        "floor_spec": float(floor_spec),
        "floor_sens": float(floor_sens),
        "floor_prec": float(floor_prec),

        # penalties
        "pen_unstable": float(pen_unstable),
        "pen_spec": float(pen_spec),
        "pen_sens": float(pen_sens),
        "pen_prec": float(pen_prec),

        # barrier triggers
        "barrier": float(barrier),
        "barrier_hit_any": int(hit_any),
        "barrier_hit_unstable": int(hit_unstable),
        "barrier_hit_spec": int(hit_spec),
        "barrier_hit_sens": int(hit_sens),
        "barrier_hit_prec": int(hit_prec),
        "barrier_dom": str(dom_name),
        "barrier_contrib_unstable": float(contribs["unstable"]),
        "barrier_contrib_spec": float(contribs["spec"]),
        "barrier_contrib_sens": float(contribs["sens"]),
        "barrier_contrib_prec": float(contribs["prec"]),

        # probe summary
        "node_probe_n": int(sum(node_probe_counts.values())),
        "node_probe_unique": int(len(probed_nodes)),
        "edge_probe_n": int(sum(edge_probe_counts.values())),
        "edge_probe_unique": int(len(probed_edges)),

        # node stats (probed scoped)
        "node_t_p50": float(_q(node_t_list, 0.50)),
        "node_t_p90": float(_q(node_t_list, 0.90)),
        "node_supp_probe_p10": float(_q(node_support_list, 0.10)),
        "node_supp_probe_p50": float(_q(node_support_list, 0.50)),
        "node_trust_after_mean": float(np.mean(node_trust_after_list)) if node_trust_after_list else 0.0,
        "node_trust_gain_mean": float(np.mean(node_trust_gain_list)) if node_trust_gain_list else 0.0,
        "node_gate_mean": float(np.mean(node_gate_list)) if node_gate_list else 0.0,

        # edge stats (probed scoped)
        "edge_t_p50": float(_q(edge_t_list, 0.50)),
        "edge_t_p90": float(_q(edge_t_list, 0.90)),
        "edge_supp_probe_p10": float(_q(edge_support_list, 0.10)),
        "edge_supp_probe_p50": float(_q(edge_support_list, 0.50)),
        "edge_trust_after_mean": float(np.mean(edge_trust_after_list)) if edge_trust_after_list else 0.0,
        "edge_trust_gain_mean": float(np.mean(edge_trust_gain_list)) if edge_trust_gain_list else 0.0,
        "edge_gate_mean": float(np.mean(edge_gate_list)) if edge_gate_list else 0.0,

        # domain summary
        "dom_top": str(dom_top),
        "dom_top_ratio": float(dom_top_ratio),
        "dom_entropy_norm": float(ent_norm),
        "dom_unknown_not_in_meta_count": int(n_unknown),
        "dom_unknown_not_in_meta_ratio": float(n_unknown / float(max(1, k))),
    }

    # per-domain columns (CSV-friendly)
    for d, c in dom_counts.items():
        key = str(d).strip().lower().replace(" ", "_").replace("/", "_")
        dbg[f"dom_cnt__{key}"] = int(c)
        dbg[f"dom_ratio__{key}"] = float(c) / float(max(1, k))

    return float(R), dbg


# -----------------------------
# Paired probe
# -----------------------------
def eval_set_to_score(
    feat_list: List[str],
    *,
    kg: Any,
    eval_cache: EvalCacheSQLite,
    ds: DatasetCtx,
    model_name: str,
    seed: int,
):
    selected_schema = {f: ds.pool_schema[f] for f in feat_list if f in ds.pool_schema}
    ms, cache_hit, cache_key = utils_cache.cached_evaluate_auc_and_more_cv(
        current_set=feat_list,
        kg=kg,
        cache=eval_cache,          # eval_cache = EvalCacheSQLite("eval_cache.sqlite")
        df=ds.df_all,
        label=ds.y_all,
        selected_schema=selected_schema,
        model_name=model_name,
        seed=int(seed),
    )
    print(f"cache_hit={cache_hit}, cache_key={cache_key}, len_feat_list={len(feat_list)}")

    auprc = utils.as_float(ms, "auprc_mean")
    score = (auprc - ds.prevalence) / max(1e-6, (1 - ds.prevalence))  # keep your current score

    return ms, float(score)


def _marginal_trust_gain(support: float, tau: float, *, min_w: float) -> float:
    """
    Δw = w(s+1)-w(s)
    用于衡量“再 probe 一次”带来的 trust 增益（边际收益）。
    """
    w0 = utils.trust_weight(support, tau, min_w=min_w)
    w1 = utils.trust_weight(support + 1.0, tau, min_w=min_w)
    return float(max(0.0, w1 - w0))


def pick_node_targets_edge_aware(
    *,
    kg: Any,
    base_set: List[str],
    P_NODE: int,
    alpha_node: float = 0.30,   # 越小越偏向 edge（救 edge 建议 0.2~0.4）
    use_topk_edges: int = 8,    # edge 价值用 topK mean，强调最缺的边
    t0_node_fill: float = 2.0,  # 可传 weights.t0_node_fill
    t0_edge_fill: float = 1.0,  # 可传 weights.t0_edge_fill
    temperature: float = 0.0,   # 0=贪心；>0=softmax抽样（增加探索）
) -> List[str]:
    """
    只从 base_set 中选择 node probe 目标（因为你的 node probe 是“删除 u”）。

    关键改动（相比你提供的版本）：
    ------------------------------------------------------------
    你原来是“每个 u 独立打分，然后取 top P_NODE”。
    这会导致：u 的高价值 edge 往往连向某些 v，但 v 不一定也被选进 node_targets，
    从而后续 pick_edge_targets(node_targets) 无法 probe 到这些高价值边 → edge 学得慢。

    本版本改为：用“集合贪心”构建 node_targets：
      - 第一个点：看 u 连向整个 base_set 的 edge 潜力（topK mean）
      - 后续每加入一个点 u：重点看 u 与【已选节点集合】之间的边价值（因为这些边一定会进入 edge_targets，可 reuse）
    这样确保 node_targets 内部诱导子图本身就是“高价值边密集”的，edge 证据增长会快很多。
    ------------------------------------------------------------

    打分公式（保持你的思路不变）：
      node_part(u) = Δw_node(s_u) * gate_node(u)
      edge_part(u,v) = Δw_edge(s_uv) * gate_edge(u,v)

      set-greedy incremental score:
        score_add(u | S) = alpha * node_part(u)
                         + (1-alpha) * topK_mean_{v in S}( edge_part(u,v) )
                         + tiny_tie_breaker

    注意：
    - base_set 会被“确定性清洗”：去空白、去重复、保持首次出现顺序（避免 set 导致不稳定）
    - 如果 temperature>0，可在每一步用 softmax 采样一个节点（无放回），增加探索
    """

    # -------------------------
    # RNG：优先用 kg.rng（若你外部传了全局 rng，建议你把它塞进 kg.rng）
    # -------------------------
    rng = kg.rng if getattr(kg, "rng", None) is not None else random.Random()

    if P_NODE <= 0 or not base_set:
        return []

    # -------------------------
    # 1) base_set 确定性清洗 + 去重（保持顺序）
    #    关键：避免重复 feature 让 gains 被重复统计
    # -------------------------
    clean_nodes = base_set # build_shared_paired_probes已经清洗过，这里直接用

    if len(clean_nodes) == 0:
        return []
    if len(clean_nodes) <= P_NODE:
        return clean_nodes[:]  # 已经不够选了

    # -------------------------
    # 2) 读取信任函数参数
    # -------------------------
    tau_node = float(kg.update_cfg.tau_node)   # 12
    tau_edge = float(kg.update_cfg.tau_edge)   # 8
    min_w_node = float(getattr(kg.update_cfg, "trust_min_w_node", 0.02))
    min_w_edge = float(getattr(kg.update_cfg, "trust_min_w_edge", 0.02))

    # -------------------------
    # 3) 预计算 node_part(u)
    # -------------------------
    node_part: Dict[str, float] = {}
    for u in clean_nodes:
        st = kg.node_stat.get(u, None)
        su = float(st.n_pair()) if st is not None else 0.0
        dwu = _marginal_trust_gain(su, tau_node, min_w=min_w_node)

        tu = float(st.effect_confidence()) if st is not None else 0.0  # |t|
        # gate：仍然保持你原来的 0.3/0.7 结构（解释：见你之前问的那个问题）
        gu = 0.3 + 0.7 * utils.sigmoid(tu - t0_node_fill)

        node_part[u] = float(dwu * gu)

    # -------------------------
    # 4) 预计算 edge_part(u,v)（只在 base_set 内部）
    #    这里用 dict-of-dict 存，避免重复计算
    # -------------------------
    edge_part: Dict[str, Dict[str, float]] = {u: {} for u in clean_nodes}

    for i in range(len(clean_nodes)):
        u = clean_nodes[i]
        for j in range(i + 1, len(clean_nodes)):
            v = clean_nodes[j]
            a, b = (u, v) if u < v else (v, u)

            est = kg.edge_stat.get((a, b), None)
            s_uv = float(est.n_pair()) if est is not None else 0.0
            dwuv = _marginal_trust_gain(s_uv, tau_edge, min_w=min_w_edge)

            t_uv = float(est.interaction_confidence()) if est is not None else 0.0  # |t|
            g_uv = 0.3 + 0.7 * utils.sigmoid(t_uv - t0_edge_fill)

            val = float(dwuv * g_uv)

            edge_part[u][v] = val
            edge_part[v][u] = val

    # -------------------------
    # 5) 一个小工具：计算 “u 与集合 S 之间的 topK mean edge value”
    #    注意：只看 u<->S 的边（这些边后面一定能 probe，因为 S 都会被选入 node_targets）
    # -------------------------
    def topk_mean_to_set(u: str, S: List[str], k_top: int) -> float:
        if not S:
            return 0.0
        vals = []
        mp = edge_part.get(u, {})
        for v in S:
            if v == u:
                continue
            vals.append(float(mp.get(v, 0.0)))
        if not vals:
            return 0.0
        vals.sort(reverse=True)
        top = vals[:max(1, min(k_top, len(vals)))]
        return float(sum(top) / len(top))

    # -------------------------
    # 6) 贪心构建 node_targets
    #    - 第一个点：看其对整个 base_set 的 edge 潜力（否则第一步没 S）
    #    - 后续：看其与已选集合之间的 edge 价值（可 reuse，能直接变成 edge_targets）
    # -------------------------
    selected: List[str] = []
    remaining = clean_nodes[:]

    # ---- step 1: pick first node ----
    first_scores: List[Tuple[str, float]] = []
    for u in remaining:
        # 第一个点的 edge 价值只能看 “连向其它所有候选点”的潜力
        edge_potential = topk_mean_to_set(u, remaining, use_topk_edges)
        score_u = alpha_node * node_part[u] + (1.0 - alpha_node) * edge_potential
        first_scores.append((u, float(score_u)))

    first_scores.sort(key=lambda x: x[1], reverse=True)

    if temperature <= 0.0:
        u0 = first_scores[0][0]
    else:
        # softmax pick one
        pool = first_scores[:max(P_NODE * 6, P_NODE)]
        logits = [s / float(temperature) for _, s in pool]
        m = max(logits)
        ws = [math.exp(x - m) for x in logits]
        Z = sum(ws)
        r = rng.random() * Z
        acc = 0.0
        idx = 0
        for i, w in enumerate(ws):
            acc += w
            if acc >= r:
                idx = i
                break
        u0 = pool[idx][0]

    selected.append(u0)
    remaining = [u for u in remaining if u != u0]

    # ---- steps 2..P_NODE: greedy add nodes ----
    while len(selected) < P_NODE and remaining:
        cand_scores: List[Tuple[str, float]] = []
        for u in remaining:
            # 关键：只看 u 与已选集合之间的边（这些边一定会被 probe）
            edge_value = topk_mean_to_set(u, selected, use_topk_edges)
            score_add = alpha_node * node_part[u] + (1.0 - alpha_node) * edge_value
            cand_scores.append((u, float(score_add)))

        cand_scores.sort(key=lambda x: x[1], reverse=True)

        if temperature <= 0.0:
            u_star = cand_scores[0][0]
        else:
            # softmax w/o replacement
            pool = cand_scores[:max(P_NODE * 6, P_NODE)]
            logits = [s / float(temperature) for _, s in pool]
            m = max(logits)
            ws = [math.exp(x - m) for x in logits]
            Z = sum(ws)
            r = rng.random() * Z
            acc = 0.0
            idx = 0
            for i, w in enumerate(ws):
                acc += w
                if acc >= r:
                    idx = i
                    break
            u_star = pool[idx][0]

        selected.append(u_star)
        remaining = [u for u in remaining if u != u_star]

    return selected


def pick_edge_targets(
    *,
    kg: Any,
    candidate_nodes: List[str],  # 通常传 node_targets
    P_EDGE: int = 1,
    t0_edge_fill: float = 2.0,   # 可传 weights.t0_edge_fill
    temperature: float = 0.0,    # 0=贪心；>0=softmax抽样
) -> List[Tuple[str, str]]:
    """
    选择本轮要做 paired edge probe 的边集合（基于 marginal trust gain）。

    设计点：
    - edge“补 1 次”的价值 = Δw(s)=w(s+1)-w(s)
    - 乘 gate(|t|-t0) 做轻微偏置：更偏向“已经露出信号但证据还少”的边
    - 重要：候选边只在 candidate_nodes 的诱导子图里，这样 edge probe 只需额外算 y00（reuse）

    返回：
      List[(u,v)]，保证 u < v
    """
    rng = kg.rng if getattr(kg, "rng", None) is not None else random.Random()

    if P_EDGE <= 0:
        return []

    # --- 确定性清洗 candidate_nodes ---
    nodes: List[str] = []
    seen = set()
    for x in candidate_nodes:
        if not isinstance(x, str):
            continue
        u = x.strip()
        if not u:
            continue
        if hasattr(kg, "_all_set") and u not in kg._all_set:
            continue
        if u in seen:
            continue
        seen.add(u)
        nodes.append(u)

    if len(nodes) < 2:
        return []

    tau_edge = float(kg.update_cfg.tau_edge)  # 8
    min_w_edge = float(getattr(kg.update_cfg, "trust_min_w_edge", 0.02))

    scored: List[Tuple[Tuple[str, str], float]] = []

    # enumerate all candidate edges among nodes
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a, b = nodes[i], nodes[j]
            u, v = (a, b) if a < b else (b, a)

            est = kg.edge_stat.get((u, v), None)
            s = float(est.n_pair()) if est is not None else 0.0
            dw = _marginal_trust_gain(s, tau_edge, min_w=min_w_edge)

            t = float(est.interaction_confidence()) if est is not None else 0.0  # |t|
            g = 0.3 + 0.7 * utils.sigmoid(t - t0_edge_fill)

            scored.append(((u, v), float(dw * g)))

    if not scored:
        return []

    scored.sort(key=lambda x: x[1], reverse=True)

    if temperature <= 0.0:
        return [e for (e, _) in scored[:min(P_EDGE, len(scored))]]

    # softmax sampling without replacement
    pool = scored[:max(P_EDGE * 8, P_EDGE)]
    chosen: List[Tuple[str, str]] = []
    cand = pool.copy()
    for _ in range(min(P_EDGE, len(cand))):
        logits = [s / float(temperature) for (_, s) in cand]
        m = max(logits)
        ws = [math.exp(x - m) for x in logits]
        Z = sum(ws)
        r = rng.random() * Z
        acc = 0.0
        idx = 0
        for i, w in enumerate(ws):
            acc += w
            if acc >= r:
                idx = i
                break
        chosen.append(cand[idx][0])
        cand.pop(idx)

    return chosen


def _sanitize_set(kg, feats: List[str]) -> List[str]:
    """order-preserving unique + strip + in-universe"""
    out = []
    seen = set()
    for f in feats:
        if not isinstance(f, str):
            continue
        f = f.strip()
        if not f:
            continue
        if hasattr(kg, "_all_set") and f not in kg._all_set:
            continue
        if f in seen:
            continue
        seen.add(f)
        out.append(f)
    return out


def _remove_feats(base: List[str], to_remove: List[str]) -> List[str]:
    """remove a list of features from base (base is assumed unique already)."""
    rm = set([x for x in to_remove if isinstance(x, str)])
    return [f for f in base if f not in rm]


def build_shared_paired_probes(
    *,
    kg: Any,
    eval_cache: EvalCacheSQLite,
    ds: DatasetCtx,
    base_set: List[str],
    base_score: float,
    P_NODE: int,
    P_EDGE: int,
    reward_w: RewardWeights,     # 只用里面的 t0_node_fill / t0_edge_fill（用于挑选时的gate偏置）
    alpha_node: float = 0.30,    # node选择里 node vs edge 的权重（救edge建议0.2~0.4）
    topk_edges: int = 10,        # 选择 node 时，每个 node 最多考虑 topk_edges 个 edge 的潜力
    # edge_from: str = "node_targets",  # "node_targets" or "base_set"
    model_name: str = "Logistic",
    eval_seed: int = 42,
    ablation_cfg: Dict[str, Any] = {},
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Build probes under your shared-probe design:

    Node probe (u in base_set):
      y_with    = score(base_set)          == base_score
      y_without = score(base_set \\ {u})   computed once per selected u

    Edge probe (u,v):
      background = base_set \\ {u,v}
      cells:
        11 = score(base_set)              == base_score
        10 = score(base_set \\ {v})       == node_without[v]   (reuse)
        01 = score(base_set \\ {u})       == node_without[u]   (reuse)
        00 = score(base_set \\ {u,v})     one extra CV per edge

    Returns:
      node_probes, edge_probes, dbg
    """
    edge_from = str(ablation_cfg.get("edge_from", "base_set"))
    use_llm_probes = bool(ablation_cfg.get("use_llm_probes", True))

    # ---- sanitize base_set (unique, stable order) ----
    B = _sanitize_set(kg, base_set)
    if not B:
        return [], [], {"warn": "empty_base_set"}

    # ---- 1) pick node targets (edge-aware) ----
    # baseline params
    agenda = getattr(kg, "control_agenda", None)

    node_targets = pick_node_targets_edge_aware(
        kg=kg,
        base_set=B,
        P_NODE=P_NODE,
        alpha_node=alpha_node,
        use_topk_edges=topk_edges,
        t0_node_fill=float(reward_w.t0_node_fill),
        t0_edge_fill=float(reward_w.t0_edge_fill),
        temperature=0.0,
    )
    node_targets = _sanitize_set(kg, node_targets)

    # [LLM] bias node_targets inside B (do NOT change B here)
    if use_llm_probes and agenda is not None:
        B_set = set(B)

        must = [x for x in getattr(agenda, "must_include", []) if x in B_set]
        pin  = [x for x in getattr(agenda, "prefer_include", []) if x in B_set]
        pex  = set([x for x in getattr(agenda, "prefer_exclude", []) if x in B_set])

        # ---- build frontier endpoints from agenda.frontier_edges ----
        f_end = []
        for (u, v) in getattr(agenda, "frontier_edges", []) or []:
            if u in B_set:
                f_end.append(u)
            if v in B_set:
                f_end.append(v)

        # baseline node_targets already computed by pick_node_targets_edge_aware(...)
        base = list(node_targets)

        # ------------------------------------------------------------------
        # NEW: Hard budgets to prevent "must_include eats all P_NODE"
        # ------------------------------------------------------------------
        P = int(P_NODE)

        # Conservative budgets: keep controller influence, but preserve freedom.
        # Example: P=10 => must<=4, endpoints<=4.
        must_budget = max(1, int(round(agenda.max_include_ratio * P)))
        end_budget  = max(1, int(round(agenda.max_include_ratio * P)))

        # Dedup helpers (order preserving)
        def _uniq(seq):
            out = []
            seen = set()
            for x in seq:
                if x not in seen:
                    out.append(x)
                    seen.add(x)
            return out

        must = _uniq(must)
        f_end = _uniq(f_end)
        pin = _uniq(pin)

        f_end_set = set(f_end)

        # Score: frontier endpoint gets big boost; low missingness preferred.
        # Deterministic and explainable; no history needed.
        def _score(x: str) -> float:
            return (2.0 if x in f_end_set else 0.0) + (1.0 - ds.var_info_map.get(x, {}).get("missing_rate", 0.0))

        # Clip MUST to budget (but favor frontier endpoints inside must)
        if len(must) > must_budget:
            must = sorted(must, key=lambda x: (_score(x), x), reverse=True)[:must_budget]

        # Clip frontier endpoints list too (otherwise endpoints could dominate)
        if len(f_end) > end_budget:
            f_end = sorted(f_end, key=lambda x: (_score(x), x), reverse=True)[:end_budget]

        # Protected: endpoints and must are not delayed by prefer_exclude
        protected = set(must) | set(f_end)

        # ------------------------------------------------------------------
        # Rebuild node_targets with priority order under budgets
        # ------------------------------------------------------------------
        priority = []
        for seq in (must, f_end, pin, base):
            for x in seq:
                if x not in priority:
                    priority.append(x)

        # soft "exclude" = delay to the end unless it's protected
        front = [x for x in priority if (x not in pex) or (x in protected)]
        back  = [x for x in priority if (x in pex) and (x not in protected)]
        priority = front + back

        node_targets = priority[:P]

        # pad if needed (deterministic)
        if len(node_targets) < P:
            for x in B:
                if x not in node_targets:
                    node_targets.append(x)
                    if len(node_targets) >= P:
                        break

    # ---- 2) evaluate node_without for chosen nodes (P_NODE CVs) ----
    # store reuse map: u -> (metrics, score_without)
    node_without_score: Dict[str, float] = {}
    node_without_metrics: Dict[str, Dict[str, Any]] = {}

    node_probes: List[Dict[str, Any]] = []
    for u in node_targets:
        B_wo_u = _remove_feats(B, [u])
        m_wo, sc_wo = eval_set_to_score(
            B_wo_u,
            kg=kg,
            eval_cache=eval_cache,
            ds=ds,
            model_name=model_name,
            seed=int(eval_seed),
        )
        node_without_score[u] = sc_wo
        node_without_metrics[u] = m_wo

        node_probes.append({
            "var": u,
            "score_with": float(base_score),      # base_set包含u
            "score_without": float(sc_wo),        # base_set删除u
        })
        print(f"node probe: {u} -> {sc_wo}")

    # ---- 3) pick edge targets ----
    if edge_from == "base_set":
        edge_nodes = B
    else:
        edge_nodes = node_targets

    edge_targets = pick_edge_targets(
        kg=kg,
        candidate_nodes=edge_nodes,
        P_EDGE=P_EDGE,
        t0_edge_fill=float(reward_w.t0_edge_fill),
        temperature=0.0,
    )

    # [LLM] prepend LLM frontier edges if executable under strict reuse
    if use_llm_probes and agenda is not None and getattr(agenda, "frontier_edges", None):
        desired = []
        # strict reuse: both endpoints must be in node_targets so y10/y01 can reuse node_without_score
        node_set = set(node_targets)
        for (u, v) in agenda.frontier_edges:
            if u in node_set and v in node_set and u != v:
                a, b = (u, v) if u < v else (v, u)
                desired.append((a, b))

        # merge: desired first, then baseline-picked, unique, cut to P_EDGE
        if desired:
            seen = set()
            merged = []
            for e in desired + list(edge_targets):
                if e in seen:
                    continue
                seen.add(e)
                merged.append(e)
                if len(merged) >= P_EDGE:
                    break
            edge_targets = merged

    # [OPTIONAL] re-rank edge_targets by domain priority weight (still no extra CV)
    if use_llm_probes and agenda is not None and getattr(agenda, "domain_priority", None) and edge_targets:
        wmap = agenda.domain_weight_map()
        domain_map = getattr(ds, "var_info_map", {}) or {}

        def edge_w(e):
            u, v = e
            du = _safe_domain(domain_map, u)
            dv = _safe_domain(domain_map, v)
            k = _pair_key(du, dv)
            return float(wmap.get(k, 1.0))

        edge_targets = sorted(edge_targets, key=lambda e: edge_w(e), reverse=True)

    # ---- 4) evaluate y00 for each edge (P_EDGE CVs) + build edge probes ----
    edge_probes: List[Dict[str, Any]] = []
    edge_00_score: Dict[Tuple[str, str], float] = {}

    # -----------------------------
    # [20260210] helper: compute score for B \ {x} if missing in node_without_score
    # -----------------------------
    def _ensure_node_without(x: str) -> None:
        """
        Ensure node_without_score[x] exists.
        If x already has a node probe, reuse it.
        Otherwise, run ONE extra CV to compute score(base_set \\ {x}) and cache it.
        This enables edge probes even when endpoints are not in node_targets.
        """
        if x in node_without_score:
            return
        B_wo_x = _remove_feats(B, [x])
        m_wo, sc_wo = eval_set_to_score(
            B_wo_x,
            kg=kg,
            eval_cache=eval_cache,
            ds=ds,
            model_name=model_name,
            seed=int(eval_seed),
        )
        node_without_score[x] = float(sc_wo)
        node_without_metrics[x] = m_wo  # keep for debugging/inspection if needed

    # -----------------------------
    # [20260210]: statistics for relaxed reuse
    # -----------------------------
    n_edges_total = 0
    n_edges_skipped = 0
    n_edges_relaxed = 0
    extra_cv_node_without = 0  # counts how many extra CVs we did for missing y10/y01

    for (u, v) in edge_targets:
        n_edges_total += 1
        # ------------------------------------------------------------
        # KEY CHANGE (ablation via edge_from):
        #
        #   edge_from == "node_targets" (strict reuse):
        #       require both endpoints already have node_without_score; else skip.
        #
        #   edge_from == "base_set" (relaxed reuse):
        #       allow endpoints NOT in node_targets by computing missing y10/y01
        #       via extra CV(s): score(base_set \\ {u}) and/or score(base_set \\ {v}).
        #
        # This keeps the edge-probe definition consistent (cells 00/10/01/11),
        # but changes compute budget per edge (1~3 CV/edge).
        # ------------------------------------------------------------
        if (u not in node_without_score) or (v not in node_without_score):
            if edge_from == "node_targets":
                # strict reuse: skip edges that cannot reuse node_without_score
                n_edges_skipped += 1
                continue

            # relaxed reuse: compute missing node-without scores (y10/y01 prerequisites)
            before = len(node_without_score)
            _ensure_node_without(u)
            _ensure_node_without(v)
            after = len(node_without_score)

            # each newly added entry corresponds to 1 extra CV
            extra_cv_node_without += int(after - before)
            n_edges_relaxed += 1

        # Now we are guaranteed to have node_without_score[u] and node_without_score[v]
        B_wo_uv = _remove_feats(B, [u, v])
        m00, s00 = eval_set_to_score(
            B_wo_uv,
            kg=kg,
            eval_cache=eval_cache,
            ds=ds,
            model_name=model_name,
            seed=int(eval_seed),
        )
        edge_00_score[(u, v)] = float(s00)

        # reuse / ensured:
        s11 = float(base_score)
        s10 = float(node_without_score[v])  # remove v only
        s01 = float(node_without_score[u])  # remove u only

        edge_probes.append({
            "u": u,
            "v": v,
            "cells": {
                "00": float(s00),
                "10": float(s10),
                "01": float(s01),
                "11": float(s11),
            }
        })
        print(f"edge probe: {u} {v} -> {s00}")

    # update dbg with new statistics
    dbg = {
        "base_k": int(len(B)),
        "node_targets": list(node_targets),
        "edge_targets": list(edge_targets),
        "node_probe_n": int(len(node_probes)),
        "edge_probe_n": int(len(edge_probes)),

        # old meaning kept but now depends on edge_from
        "skipped_edges_due_to_no_node_probe": int(n_edges_skipped),

        # NEW: relaxed-reuse diagnostics
        "edge_total": int(n_edges_total),
        "edge_relaxed_count": int(n_edges_relaxed),
        "extra_cv_for_missing_node_without": int(extra_cv_node_without),

        "agenda_mode": getattr(agenda, "mode", "NA") if agenda else "NA",
        "agenda_anchor_n": len(getattr(agenda, "anchor_nodes", [])) if agenda else 0,
        "agenda_frontier_edge_n": len(getattr(agenda, "frontier_edges", [])) if agenda else 0,
    }
    return node_probes, edge_probes, dbg


# -----------------------------
# Control Agenda (LLM -> Executor)
# -----------------------------
@dataclass
class DomainPriority:
    """A domain-pair routing weight, e.g., (CT, Smoking) -> 1.8"""
    domain_a: str
    domain_b: str
    weight: float = 1.0

    def key(self) -> Tuple[str, str]:
        a = (self.domain_a or "").strip()
        b = (self.domain_b or "").strip()
        if a <= b:
            return (a, b)
        return (b, a)


@dataclass
class ControlAgenda:
    """
    Batch-level probing agenda that will be EXECUTED inside build_shared_paired_probes()
    in the NEXT generation.

    Core idea:
    - "must_include", "prefer_include", "prefer_exclude" bias node_targets to include/exclude some nodes, so edge reuse becomes feasible.
    - "frontier_edges" are the edges LLM wants to probe next (if endpoints present).
    - "domain_priority" weights bias node and edge selection toward certain domain pairs.
    """
    mode: str = "NA"  # "edge_accel" | "balance" | "explore" | "NA"

    # (A) content
    must_include: List[str] = field(default_factory=list)
    prefer_include: List[str] = field(default_factory=list)
    prefer_exclude: List[str] = field(default_factory=list)
    frontier_edges: List[Tuple[str, str]] = field(default_factory=list)       # size <= E_MAX
    domain_priority: List[DomainPriority] = field(default_factory=list)

    # (B) policy knobs (executor uses these to bias baseline selection)
    # edge_from: str = "node_targets"        # keep reuse version stable

    # (C) safety
    max_include_nodes:  int = 10           # pick paired node probes 使用的就是10
    max_include_ratio: float = 0.4         # must and prefer inlcude在note_probes里的比例
    max_exclude_nodes:  int = 5
    max_frontier_edges: int = 20           # pick paired edge probes 使用的就是20
    max_domain_priority: int = 4

    # (D) bookkeeping / debug
    rationale: str = ""

    def domain_weight_map(self) -> Dict[Tuple[str, str], float]:
        out: Dict[Tuple[str, str], float] = {}
        for dp in self.domain_priority:
            k = dp.key()
            if not k[0] or not k[1] or k[0] == k[1]:
                continue
            w = float(dp.weight)
            if not math.isfinite(w):
                continue
            out[k] = float(max(0.1, min(5.0, w)))  # clamp for safety
        return out


def _clean_str_list(xs: Any, *, allow_empty: bool = False) -> List[str]:
    if not isinstance(xs, list):
        return []
    out: List[str] = []
    seen: Set[str] = set()
    for z in xs:
        if not isinstance(z, str):
            continue
        s = z.strip()
        if not s and not allow_empty:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _clean_edge_list(xs: Any) -> List[Tuple[str, str]]:
    """
    Accept formats:
      - [["u","v"], ["a","b"]]
      - [{"u":"...","v":"..."}, ...]
    Return list of (u,v) with u < v.
    """
    if not isinstance(xs, list):
        return []
    out: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()
    for item in xs:
        u = v = ""
        if isinstance(item, list) and len(item) >= 2:
            u = str(item[0]).strip()
            v = str(item[1]).strip()
        elif isinstance(item, dict):
            u = str(item.get("u", "")).strip()
            v = str(item.get("v", "")).strip()
        else:
            continue
        if not u or not v or u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _clean_domain_priority(xs: Any) -> List[DomainPriority]:
    """
    Accept formats:
      - [{"pair":["CT","Smoking"], "weight":1.8}, ...]
      - [{"d1":"CT","d2":"Smoking","weight":1.8}, ...]
    """
    if not isinstance(xs, list):
        return []
    out: List[DomainPriority] = []
    for item in xs:
        if not isinstance(item, dict):
            continue
        w = utils.safe_float(item.get("weight", 1.0), 1.0)
        if "pair" in item and isinstance(item["pair"], list) and len(item["pair"]) >= 2:
            d1 = str(item["pair"][0]).strip()
            d2 = str(item["pair"][1]).strip()
        else:
            d1 = str(item.get("d1", "")).strip()
            d2 = str(item.get("d2", "")).strip()
        if not d1 or not d2 or d1 == d2:
            continue
        out.append(DomainPriority(d1=d1, d2=d2, weight=w))
    return out


def sanitize_controller_output(
    controller: Any,
    *,
    all_features: Set[str],
    domain_map: Dict[str, str],
    allowed_modes: Tuple[str, ...] = ("edge_accel", "balance", "explore", "NA"),
) -> Optional[ControlAgenda]:
    """
    Turn raw LLM JSON/dict -> ControlAgenda (or None).

    Expected (NEW) LLM schema (top-level, NO nesting):
      {
        "mode": "edge_accel|balance|explore|NA",
        "must_include": ["<feature_name>", "..."],
        "prefer_include": ["<feature_name>", "..."],
        "prefer_exclude": ["<feature_name>", "..."],
        "frontier_edges": [["<u>", "<v>"], ["<u>", "<v>"]],
        "domain_priority": [{"domain_a": "<string>", "domain_b": "<string>", "weight": 1.0}],
        "rationale": "<string>"
      }

    Important:
    - STRICT: invalid items are DROPPED (not invented).
    - Reproducible: fixed defaults + clamping + deterministic truncation.
    - edge_from is FIXED by executor as "node_targets" (NOT read from LLM).
    """
    if controller is None:
        return None

    # controller might be JSON string
    if isinstance(controller, str):
        s = controller.strip()
        if not s:
            return None
        try:
            controller = json.loads(s)
        except Exception:
            return None

    if not isinstance(controller, dict) or not controller:
        return None

    # -------------------------
    # 0) mode (strict)
    # -------------------------
    mode = str(controller.get("mode", "NA")).strip()
    if mode not in allowed_modes:
        mode = "NA"

    # rationale (optional)
    rationale = str(controller.get("rationale", "")).strip()

    # -------------------------
    # 1) must_include (top-level)
    # -------------------------
    must_include = _clean_str_list(controller.get("must_include", []))
    must_include = [x for x in must_include if x in all_features]
    must_include = must_include[:controller.get("max_include_nodes", 10)]  # ControlAgenda.max_include_nodes

    # -------------------------
    # 2) prefer_include (top-level)
    # -------------------------
    prefer_include = _clean_str_list(controller.get("prefer_include", []))
    prefer_include = [x for x in prefer_include if x in all_features]
    prefer_include = prefer_include[:controller.get("max_include_nodes", 10)]  # ControlAgenda.max_include_nodes

    # -------------------------
    # 3) prefer_exclude (top-level)
    # -------------------------
    prefer_exclude = _clean_str_list(controller.get("prefer_exclude", []))
    prefer_exclude = [x for x in prefer_exclude if x in all_features]
    prefer_exclude = prefer_exclude[:controller.get("max_exclude_nodes", 5)]  # ControlAgenda.max_include_nodes

    # -------------------------
    # 4) frontier edges (top-level)
    # -------------------------
    frontier_edges = _clean_edge_list(controller.get("frontier_edges", []))
    frontier_edges = [(u, v) for (u, v) in frontier_edges if (u in all_features and v in all_features)]
    frontier_edges = frontier_edges[:controller.get("max_frontier_edges", 20)]  # ControlAgenda.max_frontier_edges

    # -------------------------
    # 5) domain priority (top-level)
    # -------------------------
    # Your helper _clean_domain_priority() should ideally accept:
    #   - {"domain_a","domain_b","weight"} (new schema)
    # If it currently expects {"d1","d2","weight"}, update that helper once.
    dp_list = _clean_domain_priority(controller.get("domain_priority", []))

    # Optional: validate against observed domains
    observed_domains = {d.strip() for d in domain_map.values() if isinstance(d, str) and d.strip()}
    if observed_domains:
        cleaned: List[DomainPriority] = []
        for dp in dp_list:
            a, b = dp.d1.strip(), dp.d2.strip()
            # 要求两个域都合法（更严格、更可解释；避免半合法导致偏置奇怪）
            if a in observed_domains and b in observed_domains and a != b:
                cleaned.append(dp)
        dp_list = cleaned

    dp_list = dp_list[:controller.get("max_domain_priority", 4)]  # ControlAgenda.max_domain_priority

    # -------------------------
    # 5) FIXED executor rule: edge_from always node_targets
    # -------------------------
    out = ControlAgenda(
        mode=mode,
        must_include=must_include,
        prefer_include=prefer_include,
        prefer_exclude=prefer_exclude,
        frontier_edges=frontier_edges,
        domain_priority=dp_list,
        rationale=rationale[:1000],
    )
    return out


def _safe_domain(domain_map: Dict[str, str], f: str) -> str:
    d = domain_map.get(f, "Other")
    if not isinstance(d, str) or not d.strip():
        return "Other"
    return d.strip()


def _pair_key(d1: str, d2: str) -> Tuple[str, str]:
    a, b = (d1, d2) if d1 <= d2 else (d2, d1)
    return (a, b)


def build_batch_summary(
    *,
    kg: KnowledgeGraph,  # KnowledgeGraph
    domain_map: Dict[str, str],  # feature -> domain
    all_features: List[str],
    # light knobs (reduced defaults to shrink payload)
    top_nodes: int = 8,
    top_edges: int = 12,
    frontier_nodes: int = 12,
    frontier_edges: int = 20,
    domain_pair_max_rows: int = 10,   # keep prompt small
) -> Dict[str, Any]:
    """
    Minimal batch summary for LLM planning (NO CV, snapshot-only).

    Key changes (for speed):
      - Fewer items in top/frontier lists (defaults reduced).
      - All floats rounded by utils.sig4() to reduce payload size.
      - domain_pair_coverage truncated (default 16 rows).
    """
    def f4(x: Any) -> float:
        # utils.sig4 keeps 4 significant digits; fallback to safe float
        try:
            return float(utils.sig4(float(x)))
        except Exception:
            try:
                return float(x)
            except Exception:
                return 0.0

    # -------------------------
    # Node table
    # -------------------------
    node_rows: List[Tuple[str, int, float]] = []  # (f, n_pair, abs_t)
    for f in all_features:
        st = kg.node_stat.get(f, None)
        n = int(st.n_pair()) if st is not None else 0
        try:
            t = float(st.effect_t()) if st is not None else 0.0
        except Exception:
            t = 0.0
        node_rows.append((f, n, float(abs(t))))

    top_probed_nodes = sorted(node_rows, key=lambda x: x[1], reverse=True)[:top_nodes]
    top_signal_nodes = sorted(node_rows, key=lambda x: x[2], reverse=True)[:top_nodes]

    # Frontier nodes: low support but non-trivial |t|
    abs_ts = np.array([x[2] for x in node_rows], dtype=float)
    t_med = float(np.median(abs_ts)) if abs_ts.size else 0.0

    node_front: List[Tuple[str, int, float, float]] = []  # (f, n, abs_t, score)
    for (f, n, at) in node_rows:
        if at <= max(0.5 * t_med, 0.5):
            continue
        score = at / (1.0 + float(n))
        node_front.append((f, n, at, float(score)))
    node_front = sorted(node_front, key=lambda x: x[3], reverse=True)[:frontier_nodes]

    # -------------------------
    # Edge table (only edges in edge_stat)
    # -------------------------
    edge_rows: List[Tuple[str, str, int, float]] = []  # (u, v, n_pair, abs_t)
    for (u, v), est in getattr(kg, "edge_stat", {}).items():
        if not isinstance(u, str) or not isinstance(v, str) or u == v:
            continue
        n = int(est.n_pair()) if est is not None else 0
        try:
            t = float(est.interaction_t()) if est is not None else 0.0
        except Exception:
            t = 0.0
        edge_rows.append((u, v, n, float(abs(t))))

    top_probed_edges = sorted(edge_rows, key=lambda x: x[2], reverse=True)[:top_edges]
    top_signal_edges = sorted(edge_rows, key=lambda x: x[3], reverse=True)[:top_edges]

    e_abs_ts = np.array([x[3] for x in edge_rows], dtype=float)
    e_t_med = float(np.median(e_abs_ts)) if e_abs_ts.size else 0.0

    edge_front: List[Tuple[str, str, int, float, float]] = []  # (u, v, n, abs_t, score)
    for (u, v, n, at) in edge_rows:
        if at <= max(0.5 * e_t_med, 0.5):
            continue
        score = at / (1.0 + float(n))
        edge_front.append((u, v, n, at, float(score)))
    edge_front = sorted(edge_front, key=lambda x: x[4], reverse=True)[:frontier_edges]

    # -------------------------
    # Domain-pair coverage (support-based)
    # -------------------------
    cov: Dict[Tuple[str, str], Dict[str, float]] = {}
    pair_to_ns: Dict[Tuple[str, str], List[int]] = {}

    for (u, v, n, at) in edge_rows:
        du = _safe_domain(domain_map, u)
        dv = _safe_domain(domain_map, v)
        k = _pair_key(du, dv)

        if k not in cov:
            cov[k] = {"edge_cnt": 0.0, "edge_cnt_npos": 0.0, "n_pair_sum": 0.0}

        cov[k]["edge_cnt"] += 1.0
        if n > 0:
            cov[k]["edge_cnt_npos"] += 1.0
        cov[k]["n_pair_sum"] += float(n)

        pair_to_ns.setdefault(k, []).append(int(n))

    domain_pair_coverage = []
    for k, st in cov.items():
        ns = pair_to_ns.get(k, [])
        p50 = float(np.median(np.asarray(ns, dtype=float))) if ns else 0.0

        domain_pair_coverage.append({
            "pair": [k[0], k[1]],
            "edges_tracked": int(st["edge_cnt"]),
            "edges_with_support": int(st["edge_cnt_npos"]),
            "support_sum": f4(st["n_pair_sum"]),
            "support_p50": f4(p50),
        })

    # sort: lowest support coverage first (gap first)
    domain_pair_coverage.sort(
        key=lambda x: (x["edges_with_support"] / max(1, x["edges_tracked"]), x["support_p50"])
    )
    domain_pair_coverage = domain_pair_coverage[:domain_pair_max_rows]

    # -------------------------
    # Progress scalars
    # -------------------------
    node_support_p50 = float(np.median(np.asarray([x[1] for x in node_rows], dtype=float))) if node_rows else 0.0
    edge_support_p50 = float(np.median(np.asarray([x[2] for x in edge_rows], dtype=float))) if edge_rows else 0.0

    return {
        "top_probed_nodes": [
            {"feature": f, "n_pair": int(n), "abs_t": f4(at), "domain": _safe_domain(domain_map, f)}
            for (f, n, at) in top_probed_nodes
        ],
        "top_signal_nodes": [
            {"feature": f, "n_pair": int(n), "abs_t": f4(at), "domain": _safe_domain(domain_map, f)}
            for (f, n, at) in top_signal_nodes
        ],
        "frontier_nodes": [
            {"feature": f, "n_pair": int(n), "abs_t": f4(at), "score": f4(sc), "domain": _safe_domain(domain_map, f)}
            for (f, n, at, sc) in node_front
        ],

        "top_probed_edges": [
            {"u": u, "v": v, "n_pair": int(n), "abs_t": f4(at),
             "du": _safe_domain(domain_map, u), "dv": _safe_domain(domain_map, v)}
            for (u, v, n, at) in top_probed_edges
        ],
        "top_signal_edges": [
            {"u": u, "v": v, "n_pair": int(n), "abs_t": f4(at),
             "du": _safe_domain(domain_map, u), "dv": _safe_domain(domain_map, v)}
            for (u, v, n, at) in top_signal_edges
        ],
        "frontier_edges": [
            {"u": u, "v": v, "n_pair": int(n), "abs_t": f4(at), "score": f4(sc),
             "du": _safe_domain(domain_map, u), "dv": _safe_domain(domain_map, v)}
            for (u, v, n, at, sc) in edge_front
        ],

        "domain_pair_coverage": domain_pair_coverage,
        "progress": {
            "node_support_p50": f4(node_support_p50),
            "edge_support_p50": f4(edge_support_p50),
            "n_edges_tracked": int(len(edge_rows)),
            "n_nodes": int(len(node_rows)),
            "t_med_node_abs": f4(t_med),
            "t_med_edge_abs": f4(e_t_med),
        }
    }


# =============================================================================
# 1) Structural snapshot (static bottleneck / gaps) — SHORT + sig4
# =============================================================================
def _sig4(x: Any) -> Any:
    """4 significant digits (scalar / nested). Falls back if utils.sig4 not available."""
    try:
        return utils.sig4(x)
    except Exception:
        pass

    def _one(v):
        try:
            v = float(v)
        except Exception:
            return v
        if not math.isfinite(v):
            return v
        if v == 0.0:
            return 0.0
        # 4 significant digits
        return float(f"{v:.4g}")

    if isinstance(x, dict):
        return {k: _sig4(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [ _sig4(v) for v in x ]
    return _one(x)


def build_structural_snapshot(
    *,
    kg: Any,                               # KnowledgeGraph-like
    domain_map: Dict[str, str],
    all_features: List[str],
    # short knobs
    top_nodes: int = 10,                   # kept for API compatibility; NOT emitted (see notes below)
    top_edges: int = 16,                   # kept for API compatibility; NOT emitted (see notes below)
    frontier_nodes: int = 10,
    frontier_edges: int = 12,
    domain_pair_gaps: int = 8,
) -> Dict[str, Any]:
    """
    Structural snapshot: bottleneck + gaps (NO dynamics).
    Output is compact, LLM-readable, and rounded (sig4) for prompt efficiency.

    Control-theoretic view (paper-style rationale)
    ----------------------------------------------
    The probe scheduler (LLM) is a controller. What it needs is NOT raw parameters (theta/phi),
    but a compact observation of the *structural uncertainty* of the current graph:
      1) Frontier nodes/edges: where additional probes have high expected value because
         confidence is non-trivial but statistical support is still low.
      2) Domain-pair gaps: which cross-domain interaction subspaces are under-supported,
         *normalized by the number of possible pairs*, to avoid bias from domain size imbalance.
      3) Progress scalars: coarse stage indicators (median supports, tracked edge count).
      4) Frontier mass: how large the remaining uncertainty set is (counts over ALL candidates).

    IMPORTANT DESIGN CHOICE (LLM robustness)
    ----------------------------------------
    We intentionally DO NOT emit "top_probed_*" / "top_signal_*" lists here (even though we still
    accept top_nodes/top_edges parameters for backward compatibility), because LLMs exhibit:
      - primacy/frequency bias (over-focusing on the first / most frequent tokens),
      - semantic salience bias (e.g., repeatedly selecting familiar tokens like "score", "age", "diag_*").
    Emitting only *frontier* items reduces the risk of the controller collapsing onto the same anchors
    (e.g., always proposing agatston_score) and improves exploration of evidence-starved interactions.

    Returns (schema)
    ----------------
    Dict with:
      - frontier_nodes: table {cols, rows}
      - frontier_edges: table {cols, rows}
      - domain_pair_gaps: table {cols, rows} (normalized gaps)
      - progress: scalars
      - frontier_mass: scalars

    Notes on statistics
    -------------------
    - We treat node_stat[f].effect_confidence() and edge_stat[(u,v)].interaction_confidence()
      as already being |t|-like magnitudes (absolute confidence).
    - Frontier ranking uses a simple "promise / support" heuristic:
          score = abs_t / (1 + n_pair)
      gated by a robust threshold derived from the median abs_t (prevents noise-dominated frontier).
    - Domain-pair gaps are normalized by the number of possible cross-domain pairs:
          possible_edges = n_dom_a * n_dom_b  (for a != b)
      and we rank gaps by low normalized support ratio.

    """
    # -------------------------
    # sanitize feature universe
    # -------------------------
    feats = [f.strip() for f in all_features if isinstance(f, str) and f.strip()]
    feats_set = set(feats)

    # Helper: build domain -> list(features) for normalization of domain gaps
    dom_to_feats: Dict[str, List[str]] = {}
    for f in feats:
        d = _safe_domain(domain_map, f)
        dom_to_feats.setdefault(d, []).append(f)

    # -------------------------
    # nodes: build frontier
    # -------------------------
    node_rows: List[Tuple[str, int, float]] = []  # (feature, n_pair, abs_t)
    node_stat = getattr(kg, "node_stat", {})
    for f in feats:
        st = node_stat.get(f, None)
        n = int(st.n_pair()) if st is not None and hasattr(st, "n_pair") else 0
        # effect_confidence() is assumed to be |t| (or monotone proxy of confidence)
        try:
            at = float(st.effect_confidence()) if st is not None else 0.0
        except Exception:
            at = 0.0
        node_rows.append((f, n, at))

    # coarse stage scalar (controller uses it as "where are we in learning")
    node_support_p50 = float(np.median([n for _, n, _ in node_rows])) if node_rows else 0.0

    # robust gate: ignore tiny signals likely to be noise (prevents frontier explosion)
    abs_ts = np.asarray([at for *_, at in node_rows], dtype=float) if node_rows else np.asarray([], dtype=float)
    t_med = float(np.median(abs_ts)) if abs_ts.size else 0.0
    node_gate = max(0.5 * t_med, 0.5)

    node_front_all: List[Tuple[str, int, float, float]] = []  # (f, n, abs_t, score)
    for (f, n, at) in node_rows:
        if at <= node_gate:
            continue
        score = at / (1.0 + float(n))
        node_front_all.append((f, n, at, float(score)))

    node_front_all.sort(key=lambda x: x[3], reverse=True)
    node_front_list = node_front_all[:max(0, int(frontier_nodes))]
    node_frontier_cnt = int(len(node_front_all))

    # -------------------------
    # edges: build frontier
    # -------------------------
    edge_rows: List[Tuple[str, str, int, float]] = []  # (u, v, n_pair, abs_t)
    edge_stat = getattr(kg, "edge_stat", {})
    for (u, v), est in edge_stat.items():
        if not (isinstance(u, str) and isinstance(v, str)) or u == v:
            continue
        if u not in feats_set or v not in feats_set:
            continue

        n = int(est.n_pair()) if est is not None and hasattr(est, "n_pair") else 0
        try:
            at = float(est.interaction_confidence()) if est is not None else 0.0
        except Exception:
            at = 0.0
        edge_rows.append((u, v, n, at))

    edge_support_p50 = float(np.median([n for *_, n, _ in edge_rows])) if edge_rows else 0.0

    e_abs_ts = np.asarray([at for *_, at in edge_rows], dtype=float) if edge_rows else np.asarray([], dtype=float)
    e_t_med = float(np.median(e_abs_ts)) if e_abs_ts.size else 0.0
    edge_gate = max(0.5 * e_t_med, 0.5)

    edge_front_all: List[Tuple[str, str, int, float, float]] = []  # (u, v, n, abs_t, score)
    for (u, v, n, at) in edge_rows:
        if at <= edge_gate:
            continue
        score = at / (1.0 + float(n))
        edge_front_all.append((u, v, n, at, float(score)))

    edge_front_all.sort(key=lambda x: x[4], reverse=True)
    edge_front_list = edge_front_all[:max(0, int(frontier_edges))]
    edge_frontier_cnt = int(len(edge_front_all))

    # -------------------------
    # domain-pair gaps (NORMALIZED)
    # -------------------------
    # Motivation (paper-style):
    # If domains are imbalanced (e.g., comorbidity dominates feature count), raw "support_sum"
    # or "edges_with_support" will mechanically favor large domains. For a controller, we need
    # a *normalized* measurement of how well a cross-domain interaction subspace has been probed,
    # relative to its opportunity size.
    #
    # For domain pair (A,B), A!=B:
    #   possible_edges = |A| * |B|
    #   supported_edges = count{ (u in A, v in B) : n_pair(u,v) > 0 }
    #   support_ratio = supported_edges / possible_edges
    #   support_p50   = median n_pair over edges in that subspace (0 if empty)
    #
    # We then rank gaps by low support_ratio (primary) and low support_p50 (secondary).

    pair_stats: Dict[Tuple[str, str], Dict[str, Any]] = {}
    pair_ns: Dict[Tuple[str, str], List[int]] = {}

    for (u, v, n, _at) in edge_rows:
        du = _safe_domain(domain_map, u)
        dv = _safe_domain(domain_map, v)
        if not du or not dv or du == dv:
            continue  # only cross-domain gaps for controller's domain_priority

        k = _pair_key(du, dv)  # canonical order (d_small, d_large)
        st = pair_stats.setdefault(
            k,
            {"edges_tracked": 0, "edges_with_support": 0, "support_sum": 0.0},
        )
        st["edges_tracked"] += 1
        if n > 0:
            st["edges_with_support"] += 1
        st["support_sum"] += float(n)
        pair_ns.setdefault(k, []).append(int(n))

    gaps_rows = []
    # enumerate *observed* domain pairs from edge_stat; if you prefer "all possible domain pairs",
    # you can enumerate dom_to_feats keys and fill missing with zeros, but that may increase output size.
    for (da, db), st in pair_stats.items():
        na = len(dom_to_feats.get(da, []))
        nb = len(dom_to_feats.get(db, []))
        possible = int(max(0, na) * max(0, nb))

        # If possible==0 (shouldn't happen), skip.
        if possible <= 0:
            continue

        ns = pair_ns.get((da, db), [])
        support_p50 = float(np.median(np.asarray(ns, dtype=float))) if ns else 0.0
        support_ratio = float(st["edges_with_support"]) / float(possible)
        # normalized "support_sum per possible edge" (helps distinguish sparse-but-deep vs broad-but-shallow)
        support_sum_norm = float(st["support_sum"]) / float(possible)

        gaps_rows.append((
            da, db,
            float(support_ratio),
            float(support_p50),
            float(support_sum_norm),
            int(possible),
            int(st["edges_with_support"]),
        ))

    # sort: smallest support_ratio first, then smallest support_p50, then smallest support_sum_norm
    gaps_rows.sort(key=lambda x: (x[2], x[3], x[4]))
    gaps_rows = gaps_rows[:max(0, int(domain_pair_gaps))]

    # -------------------------
    # Build compact, readable tables (cols + rows)
    # -------------------------
    # NOTE: Use long, human-readable keys. Do not use cryptic abbreviations.
    frontier_nodes = {
        "cols": ["feature", "n_pair", "t_abs", "score", "domain"],
        "rows": [
            [f, int(n), float(at), float(sc), _safe_domain(domain_map, f)]
            for (f, n, at, sc) in node_front_list
        ],
    }
    frontier_edges = {
        "cols": ["u", "v", "n_pair", "t_abs", "score"],
        "rows": [
            [u, v, int(n), float(at), float(sc)]
            for (u, v, n, at, sc) in edge_front_list
        ],
    }
    domain_pair_gaps = {
        "cols": ["domain_a", "domain_b", "support_ratio", "support_p50", "support_sum_norm", "possible_edges", "edges_with_support"],
        "rows": [
            [da, db, float(r), float(p50), float(ssn), int(possible), int(nsup)]
            for (da, db, r, p50, ssn, possible, nsup) in gaps_rows
        ],
    }

    out = {
        "frontier_nodes": frontier_nodes,
        "frontier_edges": frontier_edges,
        "domain_pair_gaps": domain_pair_gaps,

        # compact scalars (controller "phase awareness" without full time-series)
        "progress": {
            "n_nodes": int(len(node_rows)),
            "n_edges_tracked": int(len(edge_rows)),
            "node_support_p50": float(node_support_p50),
            "edge_support_p50": float(edge_support_p50),
        },

        # how much remains uncertain (counts over ALL frontier candidates, not just the displayed rows)
        "frontier_mass": {
            "node_frontier_cnt": int(node_frontier_cnt),
            "edge_frontier_cnt": int(edge_frontier_cnt),
        },
    }

    # round floats to reduce prompt length without changing semantics
    return _sig4(out)
# =============================================================================
# 2) Phase scalars from logs (dynamic reachability + risk + trends)
# =============================================================================
def _lin_slope(y: np.ndarray) -> float:
    """
    Robust tiny helper: slope of y vs t (t=0..n-1) using least squares.
    Returns 0 if not enough points.
    """
    if y is None:
        return 0.0
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    n = int(y.size)
    if n < 3:
        return 0.0
    t = np.arange(n, dtype=float)
    t = t - t.mean()
    denom = float((t * t).sum())
    if denom <= 1e-12:
        return 0.0
    slope = float((t * (y - y.mean())).sum() / denom)
    return slope


def build_phase_scalars_from_logs(
    iters_log: list,
    *,
    window: int = 10,
    use_median_for_sparse: bool = True,
) -> dict:
    """
    Build compact *phase / trend scalars* from in-memory iteration logs (list[dict]).

    ----------------------------
    Control-theoretic motivation
    ----------------------------
    In a closed-loop graph-learning system, the LLM controller should NOT be fed raw,
    high-dimensional internal states. Instead, it should be given a *low-dimensional
    observation* that is:
      (1) Sufficient: captures where the optimization is (phase) and how it is moving (trend);
      (2) Robust: insensitive to per-run noise (aggregate across runs);
      (3) Actionable: directly informs probe scheduling (evidence acquisition) rather than
          micro-managing model parameters.

    This function therefore compresses the batch into:
      - latest scalars: "where we are now"
      - trend scalars:  "where we are heading" (slope over a short tail window)
      - composite indicators: stable_mass / support / metric variance (phase summary)

    Output format is designed for prompt efficiency:
      - 'latest_table' and 'trend_table' are small 2-column tables (list-of-lists),
        which are more compact than large JSON dicts and reduce token overhead.
      - 'latest' keeps a minimal dict for programmatic access (only key indicators).

    -----------------------
    Behavior for short logs
    -----------------------
    - n_iters == 0: return empty
    - n_iters < 3: return latest only, no trend (not enough points to estimate slope)
    - 3 <= n_iters < window: compute slope using the available tail (w = n_iters)

    Assumes your iteration_log rows contain at least:
      - iteration (int), traj_id (int)
    and optionally many numeric columns. Non-numeric columns are ignored in aggregation.
    """

    def _safe_get_last(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
        if col not in df.columns:
            return float(default)
        try:
            v = float(df[col].iloc[-1])
            return v if np.isfinite(v) else float(default)
        except Exception:
            return float(default)

    # -------------------------
    # 1) DataFrame + aggregation (batch -> per-iteration mean)
    # -------------------------
    df = pd.DataFrame(iters_log or [])
    if df.empty or ("iteration" not in df.columns):
        return {
            "latest": {},
            "latest_table": [],
            "trend_table": [],
            "window": int(window),
            "n_iters": 0,
        }

    # Keep numeric columns only (groupby mean needs numeric types)
    # Exclude heavy JSON text columns explicitly.
    drop_cols = {"current_set_json"}
    num_cols = [
        c for c in df.columns
        if (c not in drop_cols) and pd.api.types.is_numeric_dtype(df[c])
    ]
    # Guarantee 'iteration' included for grouping
    if "iteration" not in num_cols:
        num_cols = ["iteration"] + num_cols

    g = df.groupby("iteration", as_index=False)[num_cols].mean(numeric_only=True)
    g = g.sort_values("iteration").reset_index(drop=True)

    iters = g["iteration"].astype(int).tolist()
    n_iters = int(len(iters))
    if n_iters == 0:
        return {
            "latest": {},
            "latest_table": [],
            "trend_table": [],
            "window": int(window),
            "n_iters": 0,
        }

    # -------------------------
    # 2) Tail window selection
    # -------------------------
    # We do NOT early-return when n_iters < window.
    # Instead we compute trends on the available tail (>=3 points), which is the
    # most natural “online control” behavior early in optimization.
    if n_iters < 3:
        # Not enough points to estimate a meaningful trend; return latest only.
        latest_min = {}
        for k in ("R", "auc_mean", "auprc_mean", "edge_supp_probe_p50", "node_supp_probe_p50",
                  "node_w_p50", "edge_w_p50", "flip_rate"):
            if k in g.columns:
                latest_min[k] = _safe_get_last(g, k, 0.0)

        latest_table = [[k, latest_min.get(k, 0.0)] for k in latest_min.keys()]
        return _sig4({
            "latest": latest_min,
            "latest_table": latest_table,
            "trend_table": [],
            "window": int(min(window, n_iters)),
            "n_iters": n_iters,
        })

    w = int(max(3, min(window, n_iters)))
    tail = g.tail(w).reset_index(drop=True)

    # -------------------------
    # 3) Choose a compact set of observable scalars
    # -------------------------
    # Align to your real columns in iteration_log.csv:
    #   - perf: R, auc_mean, auprc_mean, (std)
    #   - evidence: node_confirm, edge_confirm, node_supp_probe_p50, edge_supp_probe_p50
    #   - stability: node_w_p50, edge_w_p50, flip_rate
    #   - probes: node_probe_n, edge_probe_n
    #   - params: bias0, Gibbs_T, mean_abs_theta/phi (if present)
    #
    # NOTE: Keep this list short; it's the "controller observation", not a full telemetry dump.
    core_cols = [
        # performance
        "R", "auc_mean", "auprc_mean", "auc_std", "auprc_std",
        # evidence / confirmation
        "node_confirm", "edge_confirm",
        "node_supp_probe_p50", "edge_supp_probe_p50",
        # stability / exploration
        "node_w_p50", "edge_w_p50", "flip_rate",
        # probe budget accounting
        "node_probe_n", "edge_probe_n",
        # schedule / knobs (observed)
        # "bias0", "Gibbs_T",
        # optional parameter magnitude indicators (often correlate with “sharpness”)
        # "mean_abs_theta", "mean_abs_phi",
    ]

    present_cols = [c for c in core_cols if c in g.columns]

    # -------------------------
    # 4) Latest values + slopes (trend)
    # -------------------------
    latest_out = {c: _safe_get_last(g, c, 0.0) for c in present_cols}
    trend_out = {f"{c}_slope": _lin_slope(tail[c].to_numpy(dtype=float)) for c in present_cols}

    # -------------------------
    # 5) Composite “phase” indicators (low-dim state)
    # -------------------------
    # These are intentionally *interpretable* summary variables that the LLM can reason about:
    # - stable_mass_p50: proxy for “how stabilized” the evidence weights are (node+edge)
    # - support_p50_mean: proxy for “how much evidence has been accumulated”
    # - metric_std_mean: proxy for “run-to-run instability” (safety signal)
    comp = {}

    if ("node_w_p50" in g.columns) and ("edge_w_p50" in g.columns):
        comp["stable_mass_p50"] = 0.5 * _safe_get_last(g, "node_w_p50", 0.0) + 0.5 * _safe_get_last(g, "edge_w_p50", 0.0)
        comp["stable_mass_p50_slope"] = 0.5 * _lin_slope(tail["node_w_p50"].to_numpy(float)) + 0.5 * _lin_slope(tail["edge_w_p50"].to_numpy(float))

    # Prefer *_supp_probe_p50 as “support observed in probes” (what your paired probing actually accumulates)
    if ("node_supp_probe_p50" in g.columns) and ("edge_supp_probe_p50" in g.columns):
        comp["support_p50_mean"] = 0.5 * _safe_get_last(g, "node_supp_probe_p50", 0.0) + 0.5 * _safe_get_last(g, "edge_supp_probe_p50", 0.0)
        comp["support_p50_mean_slope"] = 0.5 * _lin_slope(tail["node_supp_probe_p50"].to_numpy(float)) + 0.5 * _lin_slope(tail["edge_supp_probe_p50"].to_numpy(float))

    if ("auc_std" in g.columns) and ("auprc_std" in g.columns):
        comp["metric_std_mean"] = 0.5 * _safe_get_last(g, "auc_std", 0.0) + 0.5 * _safe_get_last(g, "auprc_std", 0.0)
        comp["metric_std_mean_slope"] = 0.5 * _lin_slope(tail["auc_std"].to_numpy(float)) + 0.5 * _lin_slope(tail["auprc_std"].to_numpy(float))

    # add light “stage counters” that are genuinely useful:
    # - current iteration (last)
    # - w used for slope
    stage = {
        "iteration_last": int(iters[-1]) if iters else 0,
        "tail_window": int(w),
    }

    # Update latest_out with composites (for controller convenience)
    latest_out.update(comp)
    latest_out.update(stage)

    # -------------------------
    # 6) Compact table serialization (prompt-efficient)
    # -------------------------
    # Table rows are [name, latest, slope], which is compact and still explicit.
    latest_table = []
    trend_table = []
    for c in present_cols:
        latest_table.append([c, latest_out.get(c, 0.0)])
        trend_table.append([f"{c}_slope", trend_out.get(f"{c}_slope", 0.0)])

    # Put composites in tables too (so the LLM “sees” them even if it ignores dict keys)
    for k in ["stable_mass_p50", "support_p50_mean", "metric_std_mean",
              "stable_mass_p50_slope", "support_p50_mean_slope", "metric_std_mean_slope",
              "iteration_last", "tail_window"]:
        if k in latest_out:
            latest_table.append([k, latest_out[k]])
        if k.endswith("_slope") and k in latest_out:
            # already included above as latest values, but keep consistency if you prefer
            pass

    # -------------------------
    # 7) Return (sig4-rounded)
    # -------------------------
    return _sig4({
        # minimal dict (keep small on purpose)
        "latest": latest_out,
        "trend": trend_out,   # kept for backward compatibility / debugging
        # prompt-efficient views
        "latest_table": latest_table,
        "trend_table": trend_table,
        # bookkeeping
        # "window": int(w),
        # "n_iters": int(n_iters),
    })

# =============================================================================
# 3) Feature glossary subset (feasibility / robustness) — keep SHORT
# =============================================================================
def build_feature_glossary_subset(
    *,
    feature_glossary: List[Dict[str, Any]],
    current_set: List[str],
    structural_snapshot: Dict[str, Any],
    max_items: Optional[int] = None,   # <-- if None: ONLY include "needed" features (no fill)
    include_domains: bool = True,
    # optional fill policy (only used when max_items is not None)
    missing_rate_cut: float = 0.20,
    max_meaning_chars: int = 80,       # truncate meaning to keep payload compact
) -> Dict[str, Any]:
    """
    Build a compact feature glossary subset for the LLM payload, aligned with the
    NEW structural_snapshot schema (tables with cols/rows).

    Key design (control/estimation view; paper-style)
    -------------------------------------------------
    The controller should only see *action-relevant* feature metadata:
      - features currently present (current_set),
      - features on the current structural frontier (frontier_nodes),
      - endpoints of frontier_edges (so proposed edges/anchors are feasible),
    because extra rows dilute attention and amplify LLM primacy/frequency biases.

    Therefore:
      - By default (max_items=None), we output ONLY the features required by the snapshot/current_set.
      - If you explicitly set max_items, we optionally add a small deterministic "fill" set
        (low-missingness + mild domain diversity) to give the LLM minimal flexibility.

    Output format (table)
    ---------------------
    Returns:
      {
        "cols": ["feature", "domain", "missing_rate", "meaning"],
        "rows": [[...], ...]
      }

    Assumptions
    -----------
    - feature_glossary entries contain:
        feature (or var fallback), domain, meaning, missing_rate
      and you have already removed heavy fields like notes upstream.
    - structural_snapshot has tables:
        structural_snapshot["frontier_nodes"] = {"cols":[...], "rows":[...]}
        structural_snapshot["frontier_edges"] = {"cols":[...], "rows":[...]}
    """
    if not feature_glossary:
        return {"cols": ["feature", "domain", "missing_rate", "meaning"], "rows": []}

    # -------------------------
    # 1) normalize glossary entries -> index by feature
    # -------------------------
    norm: List[Dict[str, Any]] = []
    for it in feature_glossary:
        if not isinstance(it, dict):
            continue

        f = it.get("feature", None)
        if not isinstance(f, str) or not f.strip():
            f = it.get("var", "")
        if not isinstance(f, str) or not f.strip():
            continue

        row = dict(it)
        row["feature"] = f.strip()
        # avoid duplicate naming keys
        if "var" in row and row.get("var", None) == row["feature"]:
            row.pop("var", None)

        # normalize missing_rate if possible
        if "missing_rate" in row:
            try:
                row["missing_rate"] = float(row["missing_rate"])
            except Exception:
                row.pop("missing_rate", None)

        # truncate meaning to keep payload small
        if "meaning" in row and isinstance(row["meaning"], str) and max_meaning_chars > 0:
            m = row["meaning"].strip()
            if len(m) > max_meaning_chars:
                row["meaning"] = m[: max_meaning_chars - 1] + "…"
            else:
                row["meaning"] = m

        norm.append(row)

    by_feat = {it["feature"]: it for it in norm if isinstance(it.get("feature", None), str)}
    if not by_feat:
        return {"cols": ["feature", "domain", "missing_rate", "meaning"], "rows": []}

    # helper: safe missing rate
    def _mr(it: Dict[str, Any]) -> float:
        try:
            return float(it.get("missing_rate", 0.0))
        except Exception:
            return 0.0

    # -------------------------
    # 2) collect "needed" features from (current_set + structural frontier)
    #    structural_snapshot uses table schema now (cols/rows)
    # -------------------------
    need: List[str] = []

    # (a) current_set
    for f in current_set or []:
        if isinstance(f, str):
            f = f.strip()
            if f and f in by_feat:
                need.append(f)

    # (b) frontier_nodes table: must locate "feature" column index
    fn_tbl = structural_snapshot.get("frontier_nodes", {}) or {}
    if isinstance(fn_tbl, dict):
        cols = fn_tbl.get("cols", []) or []
        rows = fn_tbl.get("rows", []) or []
        if isinstance(cols, list) and isinstance(rows, list) and "feature" in cols:
            fi = cols.index("feature")
            for r in rows:
                if not isinstance(r, (list, tuple)) or fi >= len(r):
                    continue
                f = r[fi]
                if isinstance(f, str) and f in by_feat:
                    need.append(f)

    # (c) frontier_edges table endpoints: locate "u","v" indices
    fe_tbl = structural_snapshot.get("frontier_edges", {}) or {}
    if isinstance(fe_tbl, dict):
        cols = fe_tbl.get("cols", []) or []
        rows = fe_tbl.get("rows", []) or []
        if isinstance(cols, list) and isinstance(rows, list) and ("u" in cols) and ("v" in cols):
            ui = cols.index("u")
            vi = cols.index("v")
            for r in rows:
                if not isinstance(r, (list, tuple)):
                    continue
                if ui < len(r):
                    u = r[ui]
                    if isinstance(u, str) and u in by_feat:
                        need.append(u)
                if vi < len(r):
                    v = r[vi]
                    if isinstance(v, str) and v in by_feat:
                        need.append(v)

    # unique, keep order
    seen: Set[str] = set()
    core: List[str] = []
    for f in need:
        if f not in seen:
            seen.add(f)
            core.append(f)

    # -------------------------
    # 3) optional fill (ONLY if max_items is set)
    # -------------------------
    picked: List[str] = core[:]

    if max_items is not None:
        max_items = int(max_items)
        if max_items <= 0:
            max_items = len(picked)

        # domain diversity bookkeeping
        dom_used: Set[str] = set()
        if include_domains:
            for f in picked:
                d = by_feat[f].get("domain", "")
                if isinstance(d, str) and d.strip():
                    dom_used.add(d.strip())

        # deterministic remaining candidates: low missing first, then name
        rest = [f for f in by_feat.keys() if f not in seen]
        rest.sort(key=lambda f: (_mr(by_feat[f]), f))

        # Fill policy: prefer low-missing and (lightly) new domains early
        for f in rest:
            if len(picked) >= max_items:
                break
            it = by_feat[f]
            if _mr(it) > float(missing_rate_cut):
                continue

            if include_domains:
                d = it.get("domain", "")
                d = d.strip() if isinstance(d, str) else ""
                if d and (d not in dom_used) and len(dom_used) < 10:
                    picked.append(f)
                    dom_used.add(d)
                    continue

            picked.append(f)

        picked = picked[:max_items]

    # -------------------------
    # 4) emit table (compact, stable ordering)
    # -------------------------
    cols_out = ["feature", "domain", "missing_rate", "meaning"]
    rows_out: List[List[Any]] = []
    for f in picked:
        it = by_feat.get(f, {})
        domain = it.get("domain", "") if include_domains else ""
        meaning = it.get("meaning", "")
        mr = it.get("missing_rate", None)
        rows_out.append([
            f,
            domain if isinstance(domain, str) else "",
            float(mr) if mr is not None else None,
            meaning if isinstance(meaning, str) else "",
        ])

    out = {"cols": cols_out, "rows": rows_out}
    return _sig4(out)

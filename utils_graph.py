from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional, Iterator, TYPE_CHECKING
import numpy as np
import math
import random
import json
from dataclasses import dataclass, field
from collections import defaultdict, Counter

from config_graph import GraphUpdateConfig
from utils import sigmoid, trust_weight
if TYPE_CHECKING:
    from utils_control import ControlAgenda

# -------------------------------------------------------------------
# helper functions
# -------------------------------------------------------------------
Edge = Tuple[str, str]  # sorted (u, v)

def _sorted_edge(a: str, b: str) -> Edge:
    return (a, b) if a <= b else (b, a)


def set_to_bitvec(current_set: List[str], feat_index: Dict[str, int]) -> np.ndarray:
    x = np.zeros(len(feat_index), dtype=np.int8)
    for f in current_set:
        i = feat_index.get(f, None)
        if i is not None:
            x[i] = 1
    return x


def bitvec_to_set(x: np.ndarray, index_feat: List[str]) -> List[str]:
    return [index_feat[i] for i in range(len(index_feat)) if int(x[i]) == 1]


def _pairs_from_set(S: List[str]) -> List[Tuple[str, str]]:
    """
    只在 selected_set 内部构造边（O(k^2)），避免 full edge O(F^2)。
    返回 canonical (u,v) 形式：u <= v（字典序）
    """
    ss = sorted(set(S))
    out: List[Tuple[str, str]] = []
    L = len(ss)
    for i in range(L):
        u = ss[i]
        for j in range(i + 1, L):
            v = ss[j]
            out.append((u, v) if u <= v else (v, u))
    return out

# ============================================================
# GlobalKnowledgeGraph
# ============================================================
@dataclass
class _OnlineMoments:
    """
    Online moments for a scalar stream {z_i}:
      - n: sample count
      - s: sum(z)
      - ss: sum(z^2)

    This is sufficient to compute:
      - mean = s / n
      - unbiased sample variance (Bessel-corrected)
      - standard error of the mean (SEM) = sqrt(var / n)

    Notes:
    - We use unbiased sample variance:
        var = (ss - n * mean^2) / (n - 1),  for n >= 2
    - For n < 2, var is defined as 0.0 (no dispersion estimate).
    """
    n: int = 0
    s: float = 0.0
    ss: float = 0.0

    def update(self, z: float) -> None:
        z = float(z)
        self.n += 1
        self.s += z
        self.ss += z * z

    def mean(self) -> float:
        return self.s / self.n if self.n > 0 else 0.0

    def var_unbiased(self) -> float:
        if self.n < 2:
            return 0.0
        m = self.mean()
        # max(.,0) prevents tiny negative due to numerical errors
        return max((self.ss - self.n * m * m) / (self.n - 1), 0.0)

    def se_mean(self, eps: float = 1e-12) -> float:
        """
        Standard error of the mean:
            SE(mean) = sqrt(var / n)
        If n==0 -> inf (no evidence).
        """
        if self.n <= 0:
            return float("inf")
        return math.sqrt(self.var_unbiased() / (self.n + eps))


# =============================================================================
# Paired-probe Node evidence
# =============================================================================
@dataclass
class NodeOnlineStat:
    """
    Paired counterfactual evidence for a *single feature/node* u.

    Each probe produces a matched pair under the SAME background B:
        y_with    = score(B)
        y_without = score(B - {u})

    We record the paired difference:
        Δ_u = y_with - y_without

    Statistical meaning:
    - effect(u) = E[Δ_u]  (estimated by sample mean of deltas)
    - effect_se(u) = SE(mean(Δ_u))
    - effect_t(u) = mean(Δ_u) / SE(mean(Δ_u))

    Why this is better than present/absent averaging:
    - The background B is matched within the pair, so confounding from other
      features is controlled (to first order).
    - Variance is computed over deltas, not across heterogeneous "present" and
      "absent" contexts -> less conservative and more interpretable.

    Fields:
    - delta_mom: online moments for the delta stream {Δ_u}
    - with_mom / without_mom: optional absolute-level tracking (debug only)
      These are useful to detect drift in absolute scores, but NOT used for t.

    Practical:
    - support = n_pair = number of paired probes for this node
    """

    # Main statistic: paired differences Δ_u
    delta_mom: _OnlineMoments = field(default_factory=_OnlineMoments)

    # Optional debug: absolute scores inside the pairs
    with_mom: _OnlineMoments = field(default_factory=_OnlineMoments)     # y_with
    without_mom: _OnlineMoments = field(default_factory=_OnlineMoments)  # y_without

    def update_pair(self, *, y_with: float, y_without: float) -> None:
        """
        Update node evidence with ONE paired probe.

        Args:
            y_with:    score(B)
            y_without: score(B - {u})

        This will update:
            - delta_mom with (y_with - y_without)
            - with_mom / without_mom with absolute scores (debug)
        """
        y_with = float(y_with)
        y_without = float(y_without)
        self.with_mom.update(y_with)
        self.without_mom.update(y_without)
        self.delta_mom.update(y_with - y_without)

    # -------------------------
    # Primary effect estimates (paired)
    # -------------------------
    def n_pair(self) -> int:
        """Number of paired probes collected for this node."""
        return int(self.delta_mom.n)

    def effect(self) -> float:
        """Estimated marginal effect: mean(Δ_u)."""
        return float(self.delta_mom.mean())

    def effect_se(self, eps: float = 1e-12) -> float:
        """
        SE of mean(Δ_u). If n_pair==0 -> inf.
        """
        return float(self.delta_mom.se_mean(eps=eps))

    def effect_t(self, eps: float = 1e-12) -> float:
        """
        t-statistic for paired effect:
            t = mean(Δ_u) / SE(mean(Δ_u))
        If SE is inf or ~0 -> returns 0 (not confident / degenerate).
        """
        se = self.effect_se(eps=eps)
        if not math.isfinite(se) or se <= 0:
            return 0.0
        return float(self.effect() / (se + eps))

    def effect_confidence(self, eps: float = 1e-12) -> float:
        """
        Confidence proxy used by your pipeline: |t|.
        """
        return float(abs(self.effect_t(eps=eps)))

    # -------------------------
    # Optional debug helpers
    # -------------------------
    def mean_with(self) -> float:
        """Mean score of B ∪ {u} across probes (debug)."""
        return float(self.with_mom.mean())

    def mean_without(self) -> float:
        """Mean score of B across probes (debug)."""
        return float(self.without_mom.mean())

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize for snapshot/logging.

        We expose both:
        - paired stats (delta)
        - absolute stats (with/without) for diagnostics
        """
        return {
            # paired delta stats (primary)
            "n_pair": self.n_pair(),
            "delta_mean": self.effect(),
            "delta_se": self.effect_se(),
            "delta_t": self.effect_t(),
            "delta_conf": self.effect_confidence(),
            "delta_var": float(self.delta_mom.var_unbiased()),

            # absolute score stats (debug)
            "with_mean": self.mean_with(),
            "without_mean": self.mean_without(),
            "with_var": float(self.with_mom.var_unbiased()),
            "without_var": float(self.without_mom.var_unbiased()),
        }


# =============================================================================
# Paired-probe Edge evidence (interaction)
# =============================================================================
@dataclass
class EdgeOnlineStat:
    """
    Paired counterfactual evidence for an *edge/interaction* (u, v).

    Each probe is a matched 2x2 counterfactual block under the SAME background B:

        y00 = score(B)
        y10 = score(B ∪ {u})
        y01 = score(B ∪ {v})
        y11 = score(B ∪ {u, v})

    Interaction (difference-in-differences) sample:
        Δ_uv = y11 - y10 - y01 + y00

    Statistical meaning:
    - interaction_effect = E[Δ_uv] (estimated by mean of Δ_uv samples)
    - interaction_se     = SE(mean(Δ_uv))
    - interaction_t      = mean(Δ_uv) / SE(mean(Δ_uv))

    Why this is better than the old 2x2 independent-cells SE:
    - The four scores are not independent (same dataset, same folds, same B),
      and the old method that sums cell variances / n assumes independence,
      which tends to be conservative and also mis-calibrated.
    - Here we treat Δ_uv itself as the paired sample; this captures covariance
      implicitly and gives a cleaner confidence signal.

    Fields:
    - delta_mom: online moments for Δ_uv (primary)
    - cell_mom:  optional per-cell absolute tracking (debug / missing-cell diagnosis)
      Note: Under paired probe design, cells should NEVER be "missing" inside a
      probe. But you may still want to log if any of y** is NaN/invalid.

    Support/trust:
    - effective support = n_pair = number of paired edge probes
    """

    # Primary statistic: paired interaction deltas Δ_uv
    delta_mom: _OnlineMoments = field(default_factory=_OnlineMoments)

    # Optional debug: absolute cell scores, to inspect drift or invalid values
    c00: _OnlineMoments = field(default_factory=_OnlineMoments)
    c10: _OnlineMoments = field(default_factory=_OnlineMoments)
    c01: _OnlineMoments = field(default_factory=_OnlineMoments)
    c11: _OnlineMoments = field(default_factory=_OnlineMoments)

    # Diagnostics: count invalid observations per cell
    invalid_00: int = 0
    invalid_10: int = 0
    invalid_01: int = 0
    invalid_11: int = 0

    def update_pair(
        self,
        *,
        y00: float,
        y10: float,
        y01: float,
        y11: float,
    ) -> None:
        """
        Update edge evidence with ONE paired 2x2 probe.

        Args:
            y11: score(B)
            y01: score(B - {u})
            y10: score(B - {v})
            y00: score(B - {u, v})

        Behavior:
        - If any y** is non-finite, we still:
            * increment invalid counters for the offending cells
            * skip updating delta_mom for this probe (to avoid poisoning stats)
          (You can change this policy later if you want.)
        """
        y00 = float(y00)
        y10 = float(y10)
        y01 = float(y01)
        y11 = float(y11)

        ok00 = math.isfinite(y00)
        ok10 = math.isfinite(y10)
        ok01 = math.isfinite(y01)
        ok11 = math.isfinite(y11)

        if not ok00:
            self.invalid_00 += 1
        if not ok10:
            self.invalid_10 += 1
        if not ok01:
            self.invalid_01 += 1
        if not ok11:
            self.invalid_11 += 1

        # Update absolute cell moments where valid (debug only)
        if ok00:
            self.c00.update(y00)
        if ok10:
            self.c10.update(y10)
        if ok01:
            self.c01.update(y01)
        if ok11:
            self.c11.update(y11)

        # Only update paired interaction delta if ALL four are valid
        if ok00 and ok10 and ok01 and ok11:
            delta = (y11 - y10 - y01 + y00)
            self.delta_mom.update(delta)

    # -------------------------
    # Primary interaction estimates (paired)
    # -------------------------
    def n_pair(self) -> int:
        """Number of valid paired 2x2 probes contributing to Δ_uv."""
        return int(self.delta_mom.n)

    def interaction_effect(self) -> float:
        """Estimated interaction: mean(Δ_uv)."""
        return float(self.delta_mom.mean())

    def interaction_se(self, eps: float = 1e-12) -> float:
        """SE of mean(Δ_uv). If n_pair==0 -> inf."""
        return float(self.delta_mom.se_mean(eps=eps))

    def interaction_t(self, eps: float = 1e-12) -> float:
        """t-statistic for interaction effect based on paired Δ_uv samples."""
        se = self.interaction_se(eps=eps)
        if not math.isfinite(se) or se <= 0:
            return 0.0
        return float(self.interaction_effect() / (se + eps))

    def interaction_confidence(self, eps: float = 1e-12) -> float:
        """Confidence proxy: |t|."""
        return float(abs(self.interaction_t(eps=eps)))

    # -------------------------
    # Diagnostics requested earlier: "which cells are missing"
    # -------------------------
    def missing_cell_flags(self) -> Dict[str, int]:
        """
        In the paired design, 'missing cell' typically means:
        - we did NOT get a finite score for that counterfactual condition.

        Return:
            counts of invalid scores per cell.
        """
        return {
            "missing_00": int(self.invalid_00),
            "missing_10": int(self.invalid_10),
            "missing_01": int(self.invalid_01),
            "missing_11": int(self.invalid_11),
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize for snapshot/logging.

        Includes:
        - paired interaction delta stats (primary)
        - absolute cell stats (debug)
        - invalid/missing diagnostics
        """
        return {
            # paired delta stats (primary)
            "n_pair": self.n_pair(),
            "delta_mean": self.interaction_effect(),
            "delta_se": self.interaction_se(),
            "delta_t": self.interaction_t(),
            "delta_conf": self.interaction_confidence(),
            "delta_var": float(self.delta_mom.var_unbiased()),

            # absolute cell stats (debug)
            "n00": int(self.c00.n), "mean00": float(self.c00.mean()), "var00": float(self.c00.var_unbiased()),
            "n10": int(self.c10.n), "mean10": float(self.c10.mean()), "var10": float(self.c10.var_unbiased()),
            "n01": int(self.c01.n), "mean01": float(self.c01.mean()), "var01": float(self.c01.var_unbiased()),
            "n11": int(self.c11.n), "mean11": float(self.c11.mean()), "var11": float(self.c11.var_unbiased()),

            # missing/invalid diagnostics
            **self.missing_cell_flags(),
        }


def _edge_to_key(u: str, v: str) -> str:
    # canonical key for JSON
    return f"{u}|||{v}"


def _uniq_keep_order(xs: List[str]) -> List[str]:
    out = []
    seen = set()
    for x in xs:
        if isinstance(x, str):
            x = x.strip()
            if x and x not in seen:
                out.append(x)
                seen.add(x)
    return out

@dataclass
class KnowledgeGraph:
    """
    KnowledgeGraph = OnlineEvidenceGraph (证据账本) + IsingGraphDistribution (可采样分布)

    ┌──────────────────────────────────────────────────────────────────┐
    │  Evidence Layer (online ledger, what we observed)                 │
    │    - node_stat: 每个 feature 的 present/absent 计数 + reward 累积   │
    │    - edge_stat: 被“纳入追踪的候选边 edge_set”上的 2x2 计数 + reward │
    │    - edge_frontier: 从每次 sampled selected_set 内部共现提取的新边池 │
    │    - propose_edge_candidates(): 混合 stat/frontier/random/cold 产出 E│
    │                                                                  │
    │  Distribution Layer (Ising distribution, what we sample/optimize) │
    │    - theta: 节点势 (node potentials)                               │
    │    - phi: 边势 (edge potentials)                                   │
    │    - edge_set: 当前 Ising 模型“使用的边集合”                        │
    │    - neighbors: 由 edge_set + phi 构建的邻接表                      │
    │    - gibbs_sample/sample_set: Gibbs 采样生成 feature set            │
    │    - reinforce_update_theta_phi: REINFORCE 风格更新 theta/phi        │
    └─────────────────────────────────────────────────────────────────┘

    设计动机（为什么合并）：
      - 避免 edge_set / canonical / feat_index / neighbors 在两个对象之间同步出错
      - 把“证据更新→提边→更新分布→采样”变成一个对象的状态机，便于调试与持久化

    注意：
      - edge key 全程使用 Tuple[str,str]，且必须 canonical（_sorted_edge）
      - snapshot json 中 edges 用字符串 key（_edge_to_key）仅用于输出
    """

    # -------------------------
    # Shared feature universe
    # -------------------------
    update_cfg=GraphUpdateConfig()  # start with Stage A

    all_features: List[str]

    # 如果 True：分布层 edge_set 在初始化时直接全连接（O(F^2)）
    # F 很大时必须 False，改用 propose_edge_candidates 产生稀疏 edge_set
    full_edge_init: bool = True

    # -------------------------
    # Distribution layer params
    # -------------------------
    theta: Dict[str, float] = field(default_factory=dict)
    phi: Dict[Edge, float] = field(default_factory=dict)
    edge_set: List[Edge] = field(default_factory=list)

    feat_index: Dict[str, int] = field(default_factory=dict)
    index_feat: List[str] = field(default_factory=list)
    neighbors: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)

    # -------------------------
    # Evidence layer stats
    # -------------------------
    node_stat: Dict[str, Any] = field(default_factory=lambda: defaultdict(NodeOnlineStat))
    edge_stat: Dict[Edge, Any] = field(default_factory=lambda: defaultdict(EdgeOnlineStat))
    edge_frontier: Counter = field(default_factory=Counter)

    # confirm（保留接口占位）
    _node_confirm: Dict[str, Any] = field(default_factory=dict)
    _edge_confirm: Dict[str, Any] = field(default_factory=dict)

    # 内部：快速 membership check
    _all_set: set = field(default_factory=set, init=False)

    # 用于 REINFORCE 的 reward和advantage 统计
    baseline_reward_ema: float = 0.0
    baseline_initialized: bool = False

    running_mean_A: float = 0.0
    running_var_A: float = 1.0   # 用方差维护 std，更稳定
    adv_initialized: bool = False

    bias0_ema_err: float = 0.0   # 用于bias0修正的滑动平均误差
    kmean_ema: float = 0.0       # 用于k mean修正的滑动平均值

    # RNG (allow injection from caller for reproducibility)
    rng: Optional[random.Random] = None

    control_agenda: Optional[ControlAgenda] = None

    last_x: Optional[np.ndarray] = None   # Gibbs 采样的最后一个 feature set

    # ---------------------------------------------------------------------
    # Init
    # ---------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = random.Random()

        # ---- feature universe canonicalization ----
        self.index_feat = [str(x).strip() for x in self.all_features if isinstance(x, str) and str(x).strip()]
        self._all_set = sorted(set(self.index_feat))
        self.feat_index = {f: i for i, f in enumerate(self.index_feat)}

        # ---- theta init ----
        # θ 全 0 时，每个 feature 完全等价，链很容易表现成“随机抛硬币”
        # 给 θ 一个很小的扰动就能显著改善早期行为（不会引入偏置到某些临床特征，只是打破对称性）
        # 让某些 feature 稍微更常出现、某些稍微更常缺席
        for f in self.index_feat:
            self.theta.setdefault(f, float(self.rng.gauss(0.0, 0.02)))

        # ---- distribution edge_set init ----
        if bool(self.full_edge_init):
            F = len(self.index_feat)
            self.edge_set = []
            for i in range(F):
                u = self.index_feat[i]
                for j in range(i + 1, F):
                    v = self.index_feat[j]
                    e = _sorted_edge(u, v)
                    self.edge_set.append(e)
                    self.phi.setdefault(e, 0.0)
        else:
            self.edge_set = []

        # ---- neighbors ----
        self._rebuild_neighbors()

        # ---- (optional) touch node_stat keys for stability ----
        for f in self.index_feat:
            _ = self.node_stat[f]

    # ---------------------------------------------------------------------
    # Distribution statistics helpers
    # ---------------------------------------------------------------------
    def get_param_stats(self) -> dict:
        """
        Extended parameter + (trust, support, effective coupling) stats for logging.

        Keeps backward-compatible keys:
        mean_theta, std_theta, mean_abs_theta,
        mean_phi, std_phi, mean_abs_phi,
        n_theta, n_phi

        Adds:
        - quantiles / norms / max_abs for theta/phi
        - trust-weight statistics (node/edge)
        - effective coupling stats: tilde_phi = w_edge * phi
        - support/balance quantiles (node min-side, edge min-cell, edge 2x2 balance)
        """
        def _finite_array(xs):
            arr = np.asarray([float(x) for x in xs if math.isfinite(float(x))], dtype=float)
            return arr

        def _q(arr: np.ndarray, q: float) -> float:
            if arr.size == 0:
                return 0.0
            return float(np.quantile(arr, q, method="nearest"))

        def _l2(arr: np.ndarray) -> float:
            if arr.size == 0:
                return 0.0
            return float(np.sqrt(np.sum(arr * arr)))

        def _l1(arr: np.ndarray) -> float:
            if arr.size == 0:
                return 0.0
            return float(np.sum(np.abs(arr)))

        # ---------- theta ----------
        theta_vals = []
        for f in getattr(self, "index_feat", []) or []:
            theta_vals.append(self.theta.get(f, 0.0))
        arr_theta = _finite_array(theta_vals)

        if arr_theta.size == 0:
            mean_theta = std_theta = mean_abs_theta = 0.0
        else:
            mean_theta = float(arr_theta.mean())
            std_theta = float(arr_theta.std(ddof=0))
            mean_abs_theta = float(np.abs(arr_theta).mean())

        # ---------- phi ----------
        phis = []
        edge_set = getattr(self, "edge_set", None)
        if isinstance(edge_set, list) and len(edge_set) > 0:
            edges_iter = edge_set
        else:
            edges_iter = list(getattr(self, "phi", {}).keys())

        for e in edges_iter:
            phis.append(self.phi.get(e, 0.0))
        arr_phi = _finite_array(phis)

        if arr_phi.size == 0:
            mean_phi = std_phi = mean_abs_phi = 0.0
        else:
            mean_phi = float(arr_phi.mean())
            std_phi = float(arr_phi.std(ddof=0))
            mean_abs_phi = float(np.abs(arr_phi).mean())

        # ------------------------------------------------------------------
        # Trust weights + effective couplings (tilde_phi)
        #   Use project's canonical _trust_weight(support, tau, min_w=0.02)
        # ------------------------------------------------------------------
        tau_node = float(getattr(self.update_cfg, "tau_node", 12.0))
        tau_edge = float(getattr(self.update_cfg, "tau_edge", 8.0))
        min_w_node = float(getattr(self.update_cfg, "trust_min_w_node", 0.02))
        min_w_edge = float(getattr(self.update_cfg, "trust_min_w_edge", 0.02))

        node_w = []
        node_supp = []   # node_support

        # Evidence distribution over nodes that have been seen in ledger
        for f, st in getattr(self, "node_stat", {}).items():
            try:
                ns = float(st.n_pair())
            except Exception:
                ns = 0
            node_supp.append(ns)
            # canonical trust mapping
            w = trust_weight(ns, tau_node, min_w=min_w_node)
            node_w.append(w)

        edge_w = []
        edge_supp = []      # edge_support
        tilde_phi = []

        edge_stat = getattr(self, "edge_stat", {})
        for e in edges_iter:
            st = edge_stat.get(e, None)
            if st is None:
                es = 0
            else:
                try:
                    es = float(st.n_pair())
                except Exception:
                    es = 0

            edge_supp.append(es)

            w = trust_weight(es, tau_edge, min_w=min_w_edge)
            edge_w.append(w)

            v = float(self.phi.get(e, 0.0))
            if math.isfinite(v):
                tilde_phi.append(w * v)

        arr_node_w = _finite_array(node_w)
        arr_edge_w = _finite_array(edge_w)
        arr_tphi = _finite_array(tilde_phi)

        arr_node_supp = np.asarray(node_supp, dtype=float) if len(node_supp) > 0 else np.asarray([], dtype=float)
        arr_edge_supp = np.asarray(edge_supp, dtype=float) if len(edge_supp) > 0 else np.asarray([], dtype=float)

        # ------------------------------------------------------------------
        # Package results (backward compatible + new fields)
        # ------------------------------------------------------------------
        out = {
            # ---- original keys (DO NOT CHANGE) ----
            "mean_theta": mean_theta,
            "std_theta": std_theta,
            "mean_abs_theta": mean_abs_theta,
            "mean_phi": mean_phi,
            "std_phi": std_phi,
            "mean_abs_phi": mean_abs_phi,
            "n_theta": int(arr_theta.size),
            "n_phi": int(arr_phi.size),

            # ---- new: theta distribution shape ----
            "theta_p10": _q(arr_theta, 0.10),
            "theta_p50": _q(arr_theta, 0.50),
            "theta_p90": _q(arr_theta, 0.90),
            "abs_theta_p10": _q(np.abs(arr_theta), 0.10) if arr_theta.size else 0.0,
            "abs_theta_p50": _q(np.abs(arr_theta), 0.50) if arr_theta.size else 0.0,
            "abs_theta_p90": _q(np.abs(arr_theta), 0.90) if arr_theta.size else 0.0,
            "theta_l1": _l1(arr_theta),
            "theta_l2": _l2(arr_theta),
            "theta_max_abs": float(np.max(np.abs(arr_theta))) if arr_theta.size else 0.0,

            # ---- new: phi distribution shape ----
            "phi_p10": _q(arr_phi, 0.10),
            "phi_p50": _q(arr_phi, 0.50),
            "phi_p90": _q(arr_phi, 0.90),
            "abs_phi_p10": _q(np.abs(arr_phi), 0.10) if arr_phi.size else 0.0,
            "abs_phi_p50": _q(np.abs(arr_phi), 0.50) if arr_phi.size else 0.0,
            "abs_phi_p90": _q(np.abs(arr_phi), 0.90) if arr_phi.size else 0.0,
            "phi_l1": _l1(arr_phi),
            "phi_l2": _l2(arr_phi),
            "phi_max_abs": float(np.max(np.abs(arr_phi))) if arr_phi.size else 0.0,

            # ---- new: trust weight stats (node/edge) ----
            "node_w_mean": float(arr_node_w.mean()) if arr_node_w.size else 0.0,
            "node_w_p10": _q(arr_node_w, 0.10),
            "node_w_p50": _q(arr_node_w, 0.50),
            "node_w_p90": _q(arr_node_w, 0.90),

            "edge_w_mean": float(arr_edge_w.mean()) if arr_edge_w.size else 0.0,
            "edge_w_p10": _q(arr_edge_w, 0.10),
            "edge_w_p50": _q(arr_edge_w, 0.50),
            "edge_w_p90": _q(arr_edge_w, 0.90),

            # ---- new: effective coupling tilde_phi = w_edge * phi ----
            "tphi_mean": float(arr_tphi.mean()) if arr_tphi.size else 0.0,
            "tphi_std": float(arr_tphi.std(ddof=0)) if arr_tphi.size else 0.0,
            "tphi_mean_abs": float(np.abs(arr_tphi).mean()) if arr_tphi.size else 0.0,
            "tphi_p50_abs": _q(np.abs(arr_tphi), 0.50) if arr_tphi.size else 0.0,
            "tphi_p90_abs": _q(np.abs(arr_tphi), 0.90) if arr_tphi.size else 0.0,
            "tphi_max_abs": float(np.max(np.abs(arr_tphi))) if arr_tphi.size else 0.0,

            # ---- new: evidence support/balance quantiles (ledger diagnostics) ----
            "node_supp_p10": _q(arr_node_supp, 0.10) if arr_node_supp.size else 0.0,
            "node_supp_p50": _q(arr_node_supp, 0.50) if arr_node_supp.size else 0.0,
            "node_supp_p90": _q(arr_node_supp, 0.90) if arr_node_supp.size else 0.0,

            "edge_supp_p10": _q(arr_edge_supp, 0.10) if arr_edge_supp.size else 0.0,
            "edge_supp_p50": _q(arr_edge_supp, 0.50) if arr_edge_supp.size else 0.0,
            "edge_supp_p90": _q(arr_edge_supp, 0.90) if arr_edge_supp.size else 0.0,
        }

        return out

    # ---------------------------------------------------------------------
    # Distribution helpers
    # ---------------------------------------------------------------------
    def _rebuild_neighbors(self) -> None:
        nb = defaultdict(list)
        for (u, v) in self.edge_set:
            w = float(self.phi.get((u, v), 0.0))
            nb[u].append((v, w))
            nb[v].append((u, w))
        self.neighbors = dict(nb)

    def set_edge_set(self, edge_set: List[Edge]) -> None:
        """
        Replace distribution-layer edge_set (Ising sampler uses THIS edge set).
        Ensure phi entries exist for all edges and rebuild neighbors.

        phi被重新初始化为0. 不应该短期多次调用
        """
        self.edge_set = [_sorted_edge(u, v) for (u, v) in edge_set]
        for e in self.edge_set:
            self.phi.setdefault(e, 0.0)
        self._rebuild_neighbors()


    def _local_field(self, f: str, x: np.ndarray) -> float:
        """
        local logit for x_f = 1 given others fixed:
            h = bias0 + trust_weight_node*θ_f + Σ_{g in N(f)} ( trust_weight_edge(mcs_fg) * φ_fg ) * x_g

        trust is computed from online edge evidence:
            mcs_fg = edge_stat[(min(f,g), max(f,g))].min_cell_support()
            tw = _trust_weight(mcs_fg, tau_edge, min_w=...)
        """
        # --- node trust for theta ---
        st = self.node_stat.get(f, None)
        node_supp = float(st.n_pair()) if st is not None else 0.0
        tau_node = float(self.update_cfg.tau_node)
        bias0 = float(self.update_cfg.bias0)

        tw_node = trust_weight(node_supp, tau_node, min_w=0.02)  # 0.02~0.05 都行
        # h = bias0 + tw_node * float(self.theta.get(f, 0.0))
        h = bias0 + float(self.theta.get(f, 0.0))  # reward中使用了gate，这里就不需要 gate theta

        # tau for trust gating (add as KG attribute; pick conservative default)
        tau_edge = self.update_cfg.tau_edge
        # keep a tiny minimum weight so early stage isn't completely "phi-dead"
        min_w = self.update_cfg.trust_min_w_edge

        for (g, w) in self.neighbors.get(f, []):
            j = self.feat_index[g]
            xj = float(x[j])
            if xj <= 0.0:
                continue

            # edge key for undirected edge stats
            u, v = (f, g) if f < g else (g, f)

            # if edge_stat missing or no evidence, treat support=0
            est = self.edge_stat.get((u, v), None)
            if est is None:
                edge_supp = 0.0
            else:
                try:
                    edge_supp = float(est.n_pair())
                except Exception:
                    edge_supp = 0.0

            tw = trust_weight(edge_supp, tau_edge, min_w=min_w)

            # effective phi = raw phi * trust
            h += float(w) * float(tw) * xj

        return h


    def update_bias0(
        self,
        *,
        k_mean: float,
    ) -> dict:
        """
        Adjust bias0 to keep expected k within [k_low, k_high].

        - If k_mean > k_high: decrease bias0 (more zeros)
        - If k_mean < k_low : increase bias0 (more ones)
        - If within band: do nothing (or slowly decay integral term)

        PI/EMA makes it smooth and avoids oscillation.
        """
        k_low = float(self.update_cfg.bias_k_low)
        k_high = float(self.update_cfg.bias_k_high)
        lr_p = float(self.update_cfg.bias_lr_p)
        lr_i = float(self.update_cfg.bias_lr_i)
        ema_beta = float(self.update_cfg.bias_ema_beta)
        clamp_min = float(self.update_cfg.bias_clamp_min)
        clamp_max = float(self.update_cfg.bias_clamp_max)

        # convert to normalized error in [-1,1]-ish scale
        err = 0.0
        if k_mean > k_high:
            err = (k_mean - k_high) / max(1.0, k_high)
        elif k_mean < k_low:
            err = (k_mean - k_low) / max(1.0, k_low)
        else:
            err = 0.0

        # update integral (EMA of error)
        self.bias0_ema_err = float(ema_beta) * float(self.bias0_ema_err) + (1.0 - float(ema_beta)) * float(err)

        # PI control
        delta = - float(lr_p) * float(err) - float(lr_i) * float(self.bias0_ema_err)

        old = float(self.update_cfg.bias0)
        self.update_cfg.bias0 = float(np.clip(old + delta, float(clamp_min), float(clamp_max)))

        return {
            "bias0_old": old,
            "bias0_new": float(self.update_cfg.bias0),
            "bias0_delta": float(delta),
            "k_mean": float(k_mean),
            "k_low": int(k_low),
            "k_high": int(k_high),
            "err": float(err),
            "ema_err": float(self.bias0_ema_err),
        }


    def apply_sampling_agenda(
        self,
        *,
        # NOTE: we keep the function signature unchanged for backward compatibility.
        # The "lambda_*" inputs are now interpreted as *odds multipliers* (not raw Δb),
        # i.e., they specify how many times we want to multiply the odds of inclusion.
        #
        # odds(统计学里更常说“胜算”或“几率”， 含义：发生的可能性相对于不发生的可能性有多大)
        # odds = p/(1-p) = exp(log(p/(1-p)))
        # Example (temperature ~1):
        #   lambda_must=7   -> Δb ≈ log(7)=1.95    -> odds ×7
        #   lambda_prefer=3 -> Δb ≈ log(3)=1.1    -> odds ×3
        #   lambda_avoid=7  -> Δb ≈ -log(7)=-1.95   -> odds ×(1/7)
        #   lambda_frontier_endpoint=2 -> Δb ≈ log(2) -> odds ×2
        #
        # Keep them moderate (2~10) to avoid destabilizing sampling.
        # lamda就是odds的倍数
        lambda_must: float = 5.0,    # 3.0,
        lambda_prefer: float = 3.0,  # 1.5,
        lambda_avoid: float = 3.0,   # 2.0,  # applied with a minus sign
        lambda_frontier_endpoint: float = 3.0,  # 2.0,
        # clamp Δb for safety (in *logit* units before division by temperature in your code)
        clamp_abs: float = 5.0,
        # optional: downweight shaping on high-missing features if you have var_info_map
        use_missing_rate: bool = True,
    ) -> Dict[str, float]:
        """
        Build a per-feature soft bias map Δb[f] from the controller agenda, used to shape Gibbs sampling.

        Paper-friendly (control + odds-calibrated shaping)
        ---------------------------------------------------
        We inject an *additive bias* Δb[f] into the Ising/Gibbs conditional:
            p(x_f=1 | x_{¬f}) = σ( (h_f(x) + Δb[f]) / T )

        where h_f(x) is the baseline local field (bias0 + θ + Σ φ·x), and T is temperature.
        Since the log-odds of inclusion is:
            log( p/(1-p) ) = h_f(x)/T,
        adding Δb shifts log-odds by Δb/T. Therefore the odds ratio (OR) is:
            OR = exp(Δb/T).

        This provides a *scale-consistent* way to choose Δb without relying on θ/φ magnitudes:
        we specify controller intent as an odds multiplier m (e.g., "prefer include" = odds ×3),
        then set:
            Δb = T * log(m).

        Control-theoretic interpretation:
        - agenda is a control input u_t that *softly reshapes* the sampling distribution (base-set),
        enabling semantic guidance to affect which states are visited and thus which probes become feasible.
        - This is a low-level distribution shaping (soft constraints), not a hard rule; it preserves Gibbs form.

        Practical design choices:
        - must_include: strong positive odds multiplier
        - frontier endpoints: medium positive multiplier (to increase feasibility of probing desired edges)
        - prefer_include: weak positive multiplier
        - prefer_exclude: negative multiplier (odds ×(1/m_avoid)), but never overrides must_include
        - missing-rate attenuation: reduce shaping strength for high-missing variables to avoid fragile selections
        - clamp_abs: safety bound on Δb to prevent extreme logits and sampling collapse

        Returns:
        delta_bias: Dict[feature -> Δb] to be added to the local field h(f).
        """
        agenda = getattr(self, "control_agenda", None)
        if agenda is None:
            return {}

        # Universe: only shape features that exist in this KG sampling space
        feats = [f for f in getattr(self, "index_feat", []) if isinstance(f, str) and f.strip()]
        feat_set = set(feats)
        if not feat_set:
            return {}

        # Temperature enters the odds-calibration: Δb = T * log(m)
        # We read the *active* sampler temperature from update_cfg to keep calibration consistent across stages.
        T_samp = float(self.update_cfg.temperature)
        T_samp = max(1e-6, T_samp)

        # 20260216: lamda参数也全部从config中读取
        lambda_must = float(self.update_cfg.lambda_must)
        lambda_prefer = float(self.update_cfg.lambda_prefer)
        lambda_avoid = float(self.update_cfg.lambda_avoid)
        lambda_frontier_endpoint = float(self.update_cfg.lambda_frontier_endpoint)

        # -------- helper: convert multiplier -> Δb contribution --------
        # We treat lambda_* as multipliers (>=1). Values <=1 mean "no shaping" by default.
        def mult_to_db(mult: float) -> float:
            try:
                m = float(mult)
            except Exception:
                return 0.0
            if (not math.isfinite(m)) or (m <= 1.0):
                return 0.0
            return float(T_samp * math.log(m))

        db_must = mult_to_db(lambda_must)
        db_pref = mult_to_db(lambda_prefer)
        db_front = mult_to_db(lambda_frontier_endpoint)
        db_avoid = mult_to_db(lambda_avoid)  # applied with a minus sign

        # -------- read controller lists (agenda may be dataclass or dict-like) --------
        must = getattr(agenda, "must_include", []) or []
        pin  = getattr(agenda, "prefer_include", []) or []
        pex  = getattr(agenda, "prefer_exclude", []) or []

        must = [x for x in _uniq_keep_order(must) if isinstance(x, str) and x in feat_set]
        pin  = [x for x in _uniq_keep_order(pin)  if isinstance(x, str) and x in feat_set]
        pex  = [x for x in _uniq_keep_order(pex)  if isinstance(x, str) and x in feat_set]

        # Frontier endpoints act like "prefer include" but for feasibility:
        # they raise the chance that desired frontier edges have both endpoints present in the sampled base set.
        frontier_edges = getattr(agenda, "frontier_edges", None) or []
        f_end: List[str] = []
        for e in frontier_edges:
            try:
                u, v = e
            except Exception:
                continue
            if isinstance(u, str) and u in feat_set:
                f_end.append(u)
            if isinstance(v, str) and v in feat_set:
                f_end.append(v)
        f_end = [x for x in _uniq_keep_order(f_end) if x in feat_set]

        # -------- build Δb (additive in logit space) --------
        delta: Dict[str, float] = {}

        # 1) must_include: strong positive odds shaping
        if abs(db_must) > 0:
            for f in must:
                delta[f] = delta.get(f, 0.0) + db_must

        # 2) frontier endpoints: medium positive shaping (feasibility)
        if abs(db_front) > 0:
            for f in f_end:
                delta[f] = delta.get(f, 0.0) + db_front

        # 3) prefer_include: weaker positive shaping
        if abs(db_pref) > 0:
            for f in pin:
                delta[f] = delta.get(f, 0.0) + db_pref

        # 4) prefer_exclude: negative shaping (unless protected by must_include)
        # Interpretation: odds multiplier m_avoid => multiply odds by 1/m_avoid => Δb = -T log(m_avoid).
        must_set = set(must)
        if abs(db_avoid) > 0:
            for f in pex:
                if f in must_set:
                    continue
                delta[f] = delta.get(f, 0.0) - db_avoid

        # -------- clamp for safety --------
        # IMPORTANT: in gibbs_sample you later divide (h + Δb) by temperature again.
        # Here Δb is already scaled by T_samp, so Δb/T_samp = log(m) (the intended odds shift).
        # Clamping should be done in Δb units to prevent extreme logits.
        if clamp_abs is not None and clamp_abs > 0:
            cap = float(clamp_abs)
            for f, v in list(delta.items()):
                if (not math.isfinite(v)) or abs(v) < 1e-12:
                    delta.pop(f, None)
                    continue
                delta[f] = float(max(-cap, min(cap, v)))

        return delta


    def gibbs_sample(
        self,
        *,
        x0: Optional[np.ndarray] = None,
        ablation_cfg: Dict[str, Any] = {},
    ) -> np.ndarray:
        """
        Gibbs sampling with T sweeps. Returns x ∈ {0,1}^F.

        Warm-start (x0) is critical:
        - With small |theta| and trust-gated small |phi|, conditionals can be near 0.5.
        Then even 1 sweep can flip many bits.
        - Warm start + small T makes samples correlated across iterations (helps flip rate).

        LLM-compatible:
        - If self.agenda is not None, we apply *soft bias shaping* Δb[f] to the local field,
        so agenda can actually influence the base_set distribution.
        """
        use_llm_gibbs = bool(ablation_cfg.get("use_llm_gibbs", True))

        if self.rng is None:
            self.rng = random.Random()

        bias0 = float(self.update_cfg.bias0)
        T = int(self.update_cfg.gibbs_T)
        temperature = float(self.update_cfg.temperature)

        F = len(self.index_feat)

        # -------------------------
        # [20260209] soft shaping from agenda (baseline: empty dict)
        # -------------------------
        delta_bias = {}
        agenda = getattr(self, "control_agenda", None)
        if use_llm_gibbs and agenda is not None:
            # You can tune lambdas later; keep moderate for now.
            delta_bias = self.apply_sampling_agenda()   # 全部用默认参数，需要修改的话修改函数的默认参数

        def db(f: str) -> float:
            return float(delta_bias.get(f, 0.0))

        # -------------------------
        # init
        # -------------------------
        if x0 is None:
            x = np.zeros(F, dtype=np.int8)
            for i, f in enumerate(self.index_feat):
                # p = sigmoid((bias0 + float(self.theta.get(f, 0.0))) / max(1e-6, float(temperature)))
                # [20260209] add db(f) into the initial logit
                logit = (bias0 + float(self.theta.get(f, 0.0)) + db(f)) / max(1e-6, float(temperature))
                p = sigmoid(logit)
                x[i] = 1 if (self.rng.random() < p) else 0
        else:
            x = x0.copy().astype(np.int8)

        # -------------------------
        # sweeps
        # -------------------------
        for _ in range(T):
            idxs = list(range(F))
            self.rng.shuffle(idxs)
            for i in idxs:
                f = self.index_feat[i]
                # h = self._local_field(f, x) / max(1e-6, float(temperature))
                # [20260209] add db(f) to the local field (do NOT change theta/phi)
                h = (self._local_field(f, x) + db(f)) / max(1e-6, float(temperature))
                p1 = sigmoid(h)
                x[i] = 1 if (self.rng.random() < p1) else 0

        return x

    def sample_set(
        self,
        *,
        x0: Optional[np.ndarray] = None,
        ablation_cfg: Dict[str, Any] = {},
    ) -> List[str]:
        x = self.gibbs_sample(x0=x0, ablation_cfg=ablation_cfg)
        self.last_x = x.copy()
        return bitvec_to_set(x, self.index_feat)

    # 更新theta，phi的时候， run太少， 增加一些空采样， 降低theta，phi梯度估计的误差
    def sample_multiple_sets(self, M: int) -> List[List[str]]:
        """
        Draw M additional Gibbs samples for stabilizing batch statistics.
        No random restarts: always warm-start from last_x (or the first sample).
        """
        if self.rng is None:
            self.rng = random.Random()

        out: List[List[str]] = []

        x = None if (getattr(self, "last_x", None) is None) else self.last_x.copy()

        for _ in range(int(M)):
            x = self.gibbs_sample(x0=x)   # if x is None, gibbs_sample will do its own init
            out.append(bitvec_to_set(x, self.index_feat))

        self.last_x = x.copy()
        return out

    # ---------------------------------------------------------------------
    # Distribution update (REINFORCE)
    # ---------------------------------------------------------------------
    def _update_baseline_and_adv_stats(
        self,
        R: np.ndarray,
        *,
        ema: float,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """
        Update:
        - baseline_reward_ema  (EMA of batch mean reward)
        - running_mean_A       (EMA of advantage mean)
        - running_var_A        (EMA of advantage variance)

        Return:
        A_norm = (A_raw - running_mean_A) / sqrt(running_var_A)
        """
        R = R.astype(np.float32)
        R_mean = float(R.mean())

        # --- baseline: EMA of reward mean ---
        if not self.baseline_initialized:
            self.baseline_reward_ema = R_mean
            self.baseline_initialized = True
        else:
            self.baseline_reward_ema = float(ema) * float(self.baseline_reward_ema) + (1.0 - float(ema)) * R_mean

        A_raw = R - float(self.baseline_reward_ema)  # shape (N,)

        # --- advantage mean/std (EMA) ---
        batch_mean_A = float(A_raw.mean())
        # batch var around batch mean (more stable than around 0 when baseline lags)
        batch_var_A = float(np.mean((A_raw - batch_mean_A) ** 2))

        if not self.adv_initialized:
            self.running_mean_A = batch_mean_A
            self.running_var_A = max(batch_var_A, eps)
            self.adv_initialized = True
        else:
            self.running_mean_A = float(ema) * float(self.running_mean_A) + (1.0 - float(ema)) * batch_mean_A
            self.running_var_A  = float(ema) * float(self.running_var_A)  + (1.0 - float(ema)) * max(batch_var_A, eps)

        std_A = math.sqrt(max(float(self.running_var_A), eps))
        A_norm = (A_raw - float(self.running_mean_A)) / std_A
        return A_norm


    def reinforce_update_theta_phi(
        self,
        *,
        batch_sets: List[List[str]],
        batch_rewards: List[float]
    ) -> Dict[str, Any]:
        """
        One generation update using REINFORCE-style gradient estimate.

        Key fix (graph-aware):
        - Your sampler is Gibbs and DOES depend on phi via local fields.
        Therefore the REINFORCE score should be consistent with the Ising/Gibbs policy.
        - For Ising with binary x in {0,1}:
            d/dtheta_i log pi(x)   = (x_i - E[x_i]) / temp
            d/dphi_ij   log pi(x)  = trust_ij * (x_i x_j - E[x_i x_j]) / temp
        where expectations are approximated by batch sample means (self-consistent, low-variance).
        """
        ema = float(self.update_cfg.ema_baseline)
        temperature = float(self.update_cfg.temperature)
        adv_clip = float(self.update_cfg.adv_clip)
        ent_coef = float(self.update_cfg.ent_coef)
        lr_theta = float(self.update_cfg.lr_theta)
        lr_phi = float(self.update_cfg.lr_phi)
        bias0 = float(self.update_cfg.bias0)
        tau_edge = float(self.update_cfg.tau_edge)


        F = len(self.index_feat)
        N = len(batch_sets)
        if N == 0:
            return {"updated": False}

        # -------- (A) Build batch X (N,F) --------
        X = np.zeros((N, F), dtype=np.float32)  # X中包括每个run的结果和extra sample的结果，extra的结果没有reward
        for n, S in enumerate(batch_sets):
            X[n] = set_to_bitvec(S, self.feat_index).astype(np.float32)

        # -------- (B) Rewards -> normalized advantage --------
        R_in = np.array([float(r) for r in batch_rewards], dtype=np.float32)  # len = N_real
        A_in = self._update_baseline_and_adv_stats(R_in, ema=ema)
        A_in = np.clip(A_in, -adv_clip, adv_clip)

        A = np.zeros((N,), dtype=np.float32)
        A[:len(A_in)] = A_in   # extra samples keep A=0

        # =====================================================
        # (C) Graph-aware REINFORCE for theta (Ising sufficient statistic)
        #     score_theta_i(x) = (x_i - E[x_i]) / temp
        #     We approximate E[x_i] with batch mean under current Gibbs policy.
        # =====================================================
        mx = X.mean(axis=0)  # (F,)  approx E[x]

        # Control-variate form: cov(A, x_i) / temp  (robust even if A mean != 0)
        grad_theta = (A.reshape(-1, 1) * (X - mx.reshape(1, -1))).mean(axis=0) / temperature

        # Optional entropy regularizer (approx using marginal mx as Bernoulli prob)
        # NOTE: This is only a marginal proxy for an Ising policy, but works as a light stabilizer.
        if ent_coef != 0.0:
            p_hat = np.clip(mx, 1e-6, 1.0 - 1e-6)
            ent_grad = (1.0 - 2.0 * p_hat) * (p_hat * (1.0 - p_hat)) / temperature
            grad_theta = grad_theta + ent_coef * ent_grad

        for i, f in enumerate(self.index_feat):
            self.theta[f] = float(self.theta.get(f, 0.0)) + lr_theta * grad_theta[i]

        # =====================================================
        # (D) Graph-aware REINFORCE for phi (Ising sufficient statistic)
        #     IMPORTANT: your sampler uses an *effective* coupling:
        #         w_eff_ij = trust_ij * phi_ij
        #     Therefore gradient w.r.t raw phi must be multiplied by trust_ij.
        # =====================================================
        min_w = float(getattr(self.update_cfg, "trust_min_w_edge", 0.02))

        grad_phi: Dict[Edge, float] = {}

        for (u, v) in self.edge_set:
            if u not in self.feat_index or v not in self.feat_index:
                continue
            e = _sorted_edge(u, v)
            iu = self.feat_index[e[0]]
            iv = self.feat_index[e[1]]

            # x_i x_j sufficient statistic
            xij = X[:, iu] * X[:, iv]
            mxij = float(xij.mean())  # approx E[x_i x_j]

            # trust_ij from online evidence (MCS)
            est = self.edge_stat.get(e, None)
            if est is None:
                edge_supp = 0.0
            else:
                try:
                    edge_supp = float(est.n_pair())
                except Exception:
                    edge_supp = 0.0
            tw = trust_weight(edge_supp, tau_edge, min_w=min_w)

            # score_phi_ij(x) = tw * (xij - E[xij]) / temp
            g = float((A * (xij - mxij)).mean()) / temperature
            g = g * float(tw)

            grad_phi[e] = g

        for e, g in grad_phi.items():
            self.phi[e] = float(self.phi.get(e, 0.0)) + lr_phi * float(g)

        # -----------------------------------------------------
        # (E) Debias theta drift (ONLY valid if node logit is bias0 + theta_i)
        #     If you later gate theta inside _local_field by tw_node, this trick
        #     will distort logits for low-trust nodes. In that case:
        #       - either remove tw_node gating in _local_field (recommended), or
        #       - disable this centering and use theta weight decay instead.
        # -----------------------------------------------------
        theta_vals = np.array([float(self.theta.get(f, 0.0)) for f in self.index_feat], dtype=np.float32)
        theta_mean = float(theta_vals.mean())
        if abs(theta_mean) > 1e-12:
            for f in self.index_feat:
                self.theta[f] = float(self.theta.get(f, 0.0)) - theta_mean
            self.update_cfg.bias0 = bias0 + theta_mean

        # (F) Theta weight decay (apply AFTER centering)
        lam = float(getattr(self.update_cfg, "theta_decay", 0.0))
        if lam > 0.0:
            lam = min(max(lam, 0.0), 0.5)
            decay = 1.0 - lam
            for f in self.index_feat:
                self.theta[f] = float(self.theta.get(f, 0.0)) * decay

        # -----------------------------------------------------
        # (G) Light regularization for phi (weight decay), no centering
        # -----------------------------------------------------
        lam = float(getattr(self.update_cfg, "phi_decay", 0.0))
        if lam > 0.0:
            lam = min(max(lam, 0.0), 0.5)
            decay = 1.0 - lam
            for e in list(self.phi.keys()):
                self.phi[e] = float(self.phi[e]) * decay

        self._rebuild_neighbors()

        return {
            "updated": True,
            "batch_N": int(N),
            "R_mean": float(R_in.mean()),
            "baseline_reward_ema": float(self.baseline_reward_ema),
            "running_mean_A": float(self.running_mean_A),
            "running_std_A": float(math.sqrt(max(self.running_var_A, 1e-6))),
            "A_mean": float(A.mean()),
            "A_min": float(A.min()),
            "A_max": float(A.max()),
            "theta_grad_norm": float(np.linalg.norm(grad_theta)),
            "phi_grad_mean_abs": float(np.mean([abs(v) for v in grad_phi.values()])) if grad_phi else 0.0,
        }


    # ---------------------------------------------------------------------
    # Evidence update (paired counterfactual probes)
    # ---------------------------------------------------------------------
    def update_online_paired(
        self,
        *,
        selected_set: List[str],
        base_score: float,
        node_probes: List[Dict[str, Any]],
        edge_probes: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Update KG evidence using *paired counterfactual probes*.

        This replaces the old "update all nodes/edges in the sampled set" logic.

        Inputs
        ------
        selected_set:
            The background set B used to construct probes this iteration.
            We use it for:
              (1) sanitization / membership checks
              (2) updating edge_frontier (co-occurrence frontier for future edge proposals)

        base_score:
            score(B). This is NOT strictly required if every node/edge probe carries
            its own y_without / y00, but we keep it for:
              - sanity checks
              - optional logging/debug

        node_probes:
            List of paired node probes. Each item should provide:
              - var: str
              - y_with: float    score(B ∪ {u})
              - y_without: float score(B)
            We update NodeOnlineStat[u] with update_pair(y_with, y_without).

        edge_probes:
            List of paired edge probes. Each item should provide:
              - u, v: str
              - y00, y10, y01, y11: float
            We update EdgeOnlineStat[(u,v)] with update_pair(y00,y10,y01,y11).

        Returns
        -------
        dbg dict:
            Counts useful for logging:
              - how many node/edge probes applied
              - how many skipped due to invalid/missing keys or non-finite scores
              - how many frontier edges were added
        """
        # -------------------------
        # 0) sanitize selected_set (background B)
        # -------------------------
        S = [f for f in (selected_set or []) if isinstance(f, str)]
        S = [f.strip() for f in S if f and f.strip()]
        if hasattr(self, "_all_set") and isinstance(self._all_set, set) and len(self._all_set) > 0:
            S = [f for f in S if f in self._all_set]
        B = sorted(set(S))

        # -------------------------
        # 1) Update edge_frontier using background co-occurrence only
        #    (no score needed; this is for "which edges are frequently co-selected")
        # -------------------------
        frontier_added = 0
        if hasattr(self, "edge_frontier") and self.edge_frontier is not None:
            # Reuse your existing helper if available
            try:
                pairs = _pairs_from_set(B)  # O(k^2) inside current selected_set
            except NameError:
                # fallback if helper is not in scope
                pairs = []
                for i in range(len(B)):
                    for j in range(i + 1, len(B)):
                        pairs.append(_sorted_edge(B[i], B[j]))
            for e in pairs:
                self.edge_frontier[e] += 1
                frontier_added += 1

        # -------------------------
        # 2) Node paired evidence updates
        # -------------------------
        node_applied = 0
        node_skipped = 0

        for item in (node_probes or []):
            if not isinstance(item, dict):
                node_skipped += 1
                continue

            u = item.get("var", None)
            if not isinstance(u, str) or not u.strip():
                node_skipped += 1
                continue
            u = u.strip()

            # universe membership (optional but recommended)
            if hasattr(self, "_all_set") and isinstance(self._all_set, set) and len(self._all_set) > 0:
                if u not in self._all_set:
                    node_skipped += 1
                    continue

            # required paired scores
            # (strict paired design: explicitly pass score_with/score_without)
            if "score_with" in item and "score_without" in item:
                y_with = item["score_with"]
                y_without = item["score_without"]
            else:
                node_skipped += 1
                continue

            # sanitize numeric
            try:
                y_with = float(y_with)
                y_without = float(y_without)
            except Exception:
                node_skipped += 1
                continue

            # If non-finite, skip this probe (do NOT poison the online moments)
            if (not math.isfinite(y_with)) or (not math.isfinite(y_without)):
                node_skipped += 1
                continue

            # update
            self.node_stat[u].update_pair(y_with=y_with, y_without=y_without)
            node_applied += 1

        # -------------------------
        # 3) Edge paired evidence updates
        # -------------------------
        edge_applied = 0          # "尝试更新" 的次数（循环走到 update_pair）
        edge_effective = 0        # n_pair 真正 +1 的次数（强烈建议记录这个）
        edge_skipped = 0
        edge_nonfinite = 0
        edge_key_dup = 0

        seen_edges = set()  # 用于统计本轮 edge_probes 是否有重复边

        for item in (edge_probes or []):
            if not isinstance(item, dict):
                edge_skipped += 1
                continue

            u = item.get("u", None)
            v = item.get("v", None)
            if not isinstance(u, str) or not isinstance(v, str) or (not u.strip()) or (not v.strip()):
                edge_skipped += 1
                continue
            u = u.strip()
            v = v.strip()

            # universe membership (optional)
            if hasattr(self, "_all_set") and isinstance(self._all_set, set) and len(self._all_set) > 0:
                if (u not in self._all_set) or (v not in self._all_set):
                    edge_skipped += 1
                    continue

            # parse scores: nested "cells"
            if "cells" in item and isinstance(item["cells"], dict):
                cells = item["cells"]
                y00 = cells.get("00", cells.get("y00", None))
                y10 = cells.get("10", cells.get("y10", None))
                y01 = cells.get("01", cells.get("y01", None))
                y11 = cells.get("11", cells.get("y11", None))
            else:
                edge_skipped += 1
                continue

            if y00 is None or y10 is None or y01 is None or y11 is None:
                edge_skipped += 1
                continue

            try:
                y00 = float(y00); y10 = float(y10); y01 = float(y01); y11 = float(y11)
            except Exception:
                edge_skipped += 1
                continue

            # ---- NEW: finiteness check (和 node 分支一致) ----
            if (not math.isfinite(y00)) or (not math.isfinite(y10)) or (not math.isfinite(y01)) or (not math.isfinite(y11)):
                edge_nonfinite += 1
                edge_skipped += 1
                continue

            # canonical edge key
            e = _sorted_edge(u, v)

            # ---- NEW: duplicate edge key stats (debug only) ----
            if e in seen_edges:
                edge_key_dup += 1
            else:
                seen_edges.add(e)

            # ---- NEW: effective applied check via n_pair delta ----
            # （如果你担心性能：n_pair() 就是取计数，开销极小）
            before = float(self.edge_stat[e].n_pair())
            self.edge_stat[e].update_pair(y00=y00, y10=y10, y01=y01, y11=y11)
            after = float(self.edge_stat[e].n_pair())

            edge_applied += 1
            if after > before + 1e-12:
                edge_effective += 1

        # -------------------------
        # 4) Optional sanity: base_score finiteness (debug only)
        # -------------------------
        try:
            base_score = float(base_score)
        except Exception:
            base_score = float("nan")

        dbg = {
            "background_k": int(len(B)),
            "base_score_finite": int(math.isfinite(base_score)),
            "paired_node_applied": int(node_applied),
            "paired_node_skipped": int(node_skipped),
            "frontier_edges_added": int(frontier_added),
            "paired_edge_applied": int(edge_applied),
            "paired_edge_effective": int(edge_effective),   # 关键：真正写进 n_pair 的次数
            "paired_edge_skipped": int(edge_skipped),
            "paired_edge_nonfinite": int(edge_nonfinite),
            "paired_edge_unique": int(len(seen_edges)),
            "paired_edge_dup": int(edge_key_dup),
        }
        return dbg


    # ---------------------------------------------------------------------
    # Evidence -> edge_set proposal (unchanged logic, tuple edge keys)
    # ---------------------------------------------------------------------
    def propose_edge_candidates(
        self,
        *,
        M: int,
        anchors: Optional[List[str]] = None,
        prefer_uncertain: float = 0.6,
        prefer_synergy: float = 0.4,
        min_support: int = 8,
        frontier_frac: float = 0.30,
        random_frac: float = 0.10,
        node_pool_k: int = 120,
        per_anchor_k: int = 120,
    ) -> Tuple[List[Edge], int]:
        """
        Return E(size≈M) and a branch_code for debugging.

        Branch codes（新的含义）：
          1: stat_only
          2: stat+frontier
          3: stat+frontier+random
          4: cold_start_fill   (edge_stat 很少，主要靠 frontier/random/cold pairs)
          5: edge_stat empty + no nodes (极端异常)

        关键点：
          - 即使 edge_stat 足够填满 M，也强制保留 frontier/random 配额，
            这样 edge_stat 不会被早期固定边集“锁死”。
        """
        if self.rng is None:
            self.rng = random.Random()

        M = int(M)
        if M <= 0:
            return [], 0

        anchors = [a.strip() for a in (anchors or []) if isinstance(a, str) and a.strip() in self._all_set]
        A = set(anchors)

        # quotas
        frontier_quota = max(0, int(round(float(frontier_frac) * M)))
        random_quota = max(0, int(round(float(random_frac) * M)))
        frontier_quota = min(frontier_quota, M)
        random_quota = min(random_quota, M - frontier_quota)
        stat_quota = max(0, M - frontier_quota - random_quota)

        E: List[Edge] = []
        seen = set()

        # ---------------------------
        # 1) Take from edge_stat (exploit)
        # ---------------------------
        if stat_quota > 0 and self.edge_stat:
            scored: List[Tuple[float, Edge]] = []
            for e, st in self.edge_stat.items():
                u, v = e
                if (u not in self._all_set) or (v not in self._all_set) or (u == v):
                    continue

                s = int(st.support())

                # soft support weighting：不做硬门控，但支持越高越可信
                denom = max(1.0, float(min_support))
                support_w = 1.0 - math.exp(-float(s) / denom)
                support_w = max(0.15, min(1.0, support_w))

                w_unc = float(st.uncertainty())          # [0,0.25]
                w_syn = abs(float(st.synergy_proxy()))   # scale depends on reward
                w_anchor = 0.10 if (u in A or v in A) else 0.0

                score = support_w * (prefer_uncertain * w_unc + prefer_synergy * w_syn) + w_anchor
                scored.append((float(score), _sorted_edge(u, v)))

            scored.sort(key=lambda t: t[0], reverse=True)

            for _, e in scored:
                if e in seen:
                    continue
                seen.add(e)
                E.append(e)
                if len(E) >= stat_quota:
                    break

        # ---------------------------
        # 2) Take from frontier (discover new edges)
        # ---------------------------
        if frontier_quota > 0 and self.edge_frontier:
            frontier_scored: List[Tuple[float, Edge]] = []
            for e, c in self.edge_frontier.items():
                u, v = e
                if (u not in self._all_set) or (v not in self._all_set) or (u == v):
                    continue
                w_anchor = 0.20 if (u in A or v in A) else 0.0
                frontier_scored.append((float(c) + w_anchor, _sorted_edge(u, v)))

            frontier_scored.sort(key=lambda t: t[0], reverse=True)

            need = frontier_quota
            for _, e in frontier_scored:
                if e in seen:
                    continue
                seen.add(e)
                E.append(e)
                need -= 1
                if need <= 0:
                    break

        # ---------------------------
        # 3) Random exploration edges (explore)
        # ---------------------------
        if random_quota > 0:
            nodes_ranked = sorted(
                [f for f in self.index_feat],
                key=lambda f: float(self.node_stat[f].exposure_gap()),
                reverse=True,
            )[: int(max(20, node_pool_k))]

            if not nodes_ranked:
                return E, 5

            self.rng.shuffle(nodes_ranked)

            need = random_quota
            if anchors:
                cand = nodes_ranked[:]
                self.rng.shuffle(cand)
                for a in anchors:
                    for b in cand[: int(per_anchor_k)]:
                        if a == b:
                            continue
                        e = _sorted_edge(a, b)
                        if e in seen:
                            continue
                        seen.add(e)
                        E.append(e)
                        need -= 1
                        if need <= 0:
                            break
                    if need <= 0:
                        break

            while need > 0 and len(nodes_ranked) >= 2:
                u = nodes_ranked[self.rng.randrange(0, len(nodes_ranked))]
                v = nodes_ranked[self.rng.randrange(0, len(nodes_ranked))]
                if u == v:
                    continue
                e = _sorted_edge(u, v)
                if e in seen:
                    continue
                seen.add(e)
                E.append(e)
                need -= 1

        # ---------------------------
        # 4) Cold-start fill (pairs among top nodes)
        # ---------------------------
        if len(E) < M:
            nodes_ranked = sorted(
                [f for f in self.index_feat],
                key=lambda f: float(self.node_stat[f].exposure_gap()),
                reverse=True,
            )[: int(max(30, node_pool_k))]

            if not nodes_ranked:
                return E, 5

            cand_nodes = list(nodes_ranked)
            self.rng.shuffle(cand_nodes)

            L = len(cand_nodes)
            for i in range(L):
                u = cand_nodes[i]
                for j in range(i + 1, L):
                    v = cand_nodes[j]
                    e = _sorted_edge(u, v)
                    if e in seen:
                        continue
                    seen.add(e)
                    E.append(e)
                    if len(E) >= M:
                        break
                if len(E) >= M:
                    break

        E = E[:M]

        used_frontier = 1 if (frontier_quota > 0 and bool(self.edge_frontier)) else 0
        used_random = 1 if (random_quota > 0) else 0
        used_stat = 1 if (len(self.edge_stat) > 0 and stat_quota > 0) else 0

        if (not used_stat) and (used_frontier or used_random):
            return E, 4
        if used_stat and (not used_frontier) and (not used_random):
            return E, 1
        if used_stat and used_frontier and (not used_random):
            return E, 2
        return E, 3

    # ---------------------------------------------------------------------
    # Snapshot (evidence layer → json)
    # ---------------------------------------------------------------------
    def save_snapshot(self, path_json: str) -> str:
        nodes = {str(k): v.to_dict() for k, v in self.node_stat.items()}
        edges = {
            _edge_to_key(str(u), str(v)): st.to_dict()
            for (u, v), st in self.edge_stat.items()
        }
        snap = {
            "meta": {
                "n_features": int(len(self.index_feat)),
                "n_edges_tracked": int(len(self.edge_stat)),
                "n_frontier": int(len(self.edge_frontier)),
                "dist_edge_set_size": int(len(self.edge_set)),
                "full_edge_init": bool(self.full_edge_init),
            },
            "nodes": nodes,
            "edges": edges,
        }

        with open(path_json, "w", encoding="utf-8") as f:
            json.dump(snap, f, ensure_ascii=True, indent=2)
        return path_json

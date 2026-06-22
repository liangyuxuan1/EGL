# knowledge graph utils for computing evidence metrics
import json, os, re, glob
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utils_plot import set_paper_style
set_paper_style()

Edge = Tuple[str, str]

# -------------------------
# basic utils
# -------------------------
def _trust(supp: float, tau: float) -> float:
    supp = max(0.0, float(supp))
    tau = max(1e-6, float(tau))
    return 1.0 - math.exp(-supp / tau)

def _tanh01(x: float) -> float:
    return float(math.tanh(abs(float(x))))

def _finite(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)

def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read field from dict OR object(method/attr)."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    if hasattr(obj, key) and callable(getattr(obj, key)):
        try:
            return getattr(obj, key)()
        except Exception:
            return default
    if hasattr(obj, key):
        return getattr(obj, key)
    return default


def _edge_key_to_tuple(k: Any) -> Optional[Edge]:
    """Support (u,v) tuple or 'u|||v' string."""
    if isinstance(k, tuple) and len(k) == 2:
        u, v = str(k[0]), str(k[1])
        return (u, v) if u <= v else (v, u)
    if isinstance(k, str) and "|||" in k:
        a, b = k.split("|||", 1)
        u, v = a.strip(), b.strip()
        return (u, v) if u <= v else (v, u)
    return None


# -------------------------
# adapters
# -------------------------
class KGAdapter:
    """Adapter for online KnowledgeGraph."""
    def __init__(self, kg: Any):
        self.kg = kg

    def iter_nodes(self) -> Iterator[Tuple[str, Any]]:
        node_stat = getattr(self.kg, "node_stat", {})
        for f, st in node_stat.items():
            yield str(f), st

    def iter_edges(self) -> Iterator[Tuple[Edge, Any]]:
        edge_stat = getattr(self.kg, "edge_stat", {})
        for k, est in edge_stat.items():
            e = _edge_key_to_tuple(k)
            if e is None:
                try:
                    u, v = k
                    e = _edge_key_to_tuple((u, v))
                except Exception:
                    continue
            yield e, est


class SnapshotAdapter:
    """
    Adapter for your kg_snapshot.json format.

    Expected:
      snapshot["nodes"] : dict {feat: node_dict}
      snapshot["edges"] : dict {"u|||v": edge_dict}   (or snapshot["edge_stat"] similarly)
    """
    def __init__(self, snapshot: Dict[str, Any]):
        self.s = snapshot

    def iter_nodes(self) -> Iterator[Tuple[str, Any]]:
        nodes = self.s.get("nodes", None)
        if isinstance(nodes, dict):
            for f, nd in nodes.items():
                yield str(f), nd

    def iter_edges(self) -> Iterator[Tuple[Edge, Any]]:
        edges = self.s.get("edges", self.s.get("edge_stat", None))
        if isinstance(edges, dict):
            for k, ed in edges.items():
                e = _edge_key_to_tuple(k)
                if e is not None:
                    yield e, ed


def load_snapshot(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------
# configs & outputs
# -------------------------
@dataclass
class KGMetricConfig:
    # confirmed definition
    t0_node: float = 2.0
    t0_edge: float = 2.0
    mss0: int = 5          # node min_side_support gate
    mcs0: int = 3          # edge min_cell_support gate

    # trust shaping
    tau_node: float = 12.0
    tau_edge: float = 8.0

    # ranking for strength/balance (topK by |t|)
    topk_nodes: int = 20
    topk_edges: int = 20

    # optional scan cap (for very large sparse graphs)
    max_edges_scan: Optional[int] = None


def compute_kg_evidence_metrics(
    source: Union[Any, Dict[str, Any], KGAdapter, SnapshotAdapter],
    *,
    cfg: KGMetricConfig = KGMetricConfig(),
) -> Dict[str, Any]:
    """
    Compute per-generation evidence metrics:
      1) Confirmed count @ fixed threshold (with support gates)
      2) Evidence balance (support distributions)
      3) Trust-weighted evidence strength (topK)
    """

    # choose adapter
    if isinstance(source, KGAdapter) or isinstance(source, SnapshotAdapter):
        adp = source
    elif isinstance(source, dict):
        adp = SnapshotAdapter(source)
    else:
        adp = KGAdapter(source)

    # -------------------------
    # nodes
    # -------------------------
    node_rows: List[Tuple[str, float, int]] = []
    n_node_conf = 0

    for f, st in adp.iter_nodes():
        t = _finite(_get(st, "effect_t", 0.0), 0.0)
        mss = int(_get(st, "min_side_support", 0))
        node_rows.append((f, t, mss))
        if abs(t) >= float(cfg.t0_node) and mss >= int(cfg.mss0):
            n_node_conf += 1

    node_rows_sorted = sorted(node_rows, key=lambda x: abs(x[1]), reverse=True)
    top_nodes = node_rows_sorted[: int(cfg.topk_nodes)]

    mss_all = [mss for _, _, mss in node_rows]
    mss_top = [mss for _, _, mss in top_nodes]

    node_tw_strength = 0.0
    for _, t, mss in top_nodes:
        node_tw_strength += _trust(mss, cfg.tau_node) * _tanh01(t)

    # -------------------------
    # edges
    # -------------------------
    edge_rows: List[Tuple[Edge, float, int, float]] = []
    n_edge_conf = 0

    scanned = 0
    for (u, v), est in adp.iter_edges():
        scanned += 1
        if cfg.max_edges_scan is not None and scanned > int(cfg.max_edges_scan):
            break

        t = _finite(_get(est, "interaction_t", 0.0), 0.0)
        mcs = int(_get(est, "min_cell_support", 0))
        bs = _finite(_get(est, "balanced_support", 0.0), 0.0)

        edge_rows.append(((u, v), t, mcs, bs))
        if abs(t) >= float(cfg.t0_edge) and mcs >= int(cfg.mcs0):
            n_edge_conf += 1

    edge_rows_sorted = sorted(edge_rows, key=lambda x: abs(x[1]), reverse=True)
    top_edges = edge_rows_sorted[: int(cfg.topk_edges)]

    mcs_all = [mcs for _, _, mcs, _ in edge_rows]
    mcs_top = [mcs for _, _, mcs, _ in top_edges]
    bs_all = [bs for _, _, _, bs in edge_rows]
    bs_top = [bs for _, _, _, bs in top_edges]

    edge_tw_strength = 0.0
    for _, t, mcs, _ in top_edges:
        edge_tw_strength += _trust(mcs, cfg.tau_edge) * _tanh01(t)

    # -------------------------
    # return dict
    # -------------------------
    out = {
        # confirmed counts
        "n_node_conf": int(n_node_conf),
        "n_edge_conf": int(n_edge_conf),

        # balance stats
        "node_mss_mean": float(np.mean(mss_all)) if mss_all else 0.0,
        "node_mss_median": float(np.median(mss_all)) if mss_all else 0.0,
        "node_mss_top_median": float(np.median(mss_top)) if mss_top else 0.0,

        "edge_mcs_mean": float(np.mean(mcs_all)) if mcs_all else 0.0,
        "edge_mcs_median": float(np.median(mcs_all)) if mcs_all else 0.0,
        "edge_mcs_top_median": float(np.median(mcs_top)) if mcs_top else 0.0,

        "edge_bs_mean": float(np.mean(bs_all)) if bs_all else 0.0,
        "edge_bs_median": float(np.median(bs_all)) if bs_all else 0.0,
        "edge_bs_top_median": float(np.median(bs_top)) if bs_top else 0.0,

        # trust-weighted strength
        "node_tw_strength_topk": float(node_tw_strength),
        "edge_tw_strength_topk": float(edge_tw_strength),

        # optional small previews (debug/prompt)
        "top_nodes": [{"feature": f, "effect_t": float(t), "min_side_support": int(mss)}
                      for f, t, mss in top_nodes],
        "top_edges": [{"u": u, "v": v, "interaction_t": float(t),
                       "min_cell_support": int(mcs), "balanced_support": float(bs)}
                      for (u, v), t, mcs, bs in top_edges],

        # echo config
        "cfg": {
            "t0_node": float(cfg.t0_node),
            "t0_edge": float(cfg.t0_edge),
            "mss0": int(cfg.mss0),
            "mcs0": int(cfg.mcs0),
            "tau_node": float(cfg.tau_node),
            "tau_edge": float(cfg.tau_edge),
            "topk_nodes": int(cfg.topk_nodes),
            "topk_edges": int(cfg.topk_edges),
            "max_edges_scan": cfg.max_edges_scan,
        }
    }
    return out


def _collect_metrics_over_generations(out_root: str, method_name: str) -> pd.DataFrame:
    snap_dir = os.path.join(out_root, method_name, "kg_snapshot")
    pattern = os.path.join(snap_dir, "kg_snapshot_*.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No snapshots found: {pattern}")

    rows = []
    for p in paths:
        fn = os.path.basename(p)
        m = re.match(r"kg_snapshot_(\d+)\.json$", fn)
        if not m:
            continue
        gen = int(m.group(1))
        snap = load_snapshot(p)
        met = compute_kg_evidence_metrics(snap)  # 用你现有默认阈值/参数
        met["generation"] = gen
        rows.append(met)

    df = pd.DataFrame(rows).sort_values("generation").reset_index(drop=True)
    df["method"] = method_name
    return df

def plot_kg_metrics_compare(out_root: str, method_a: str = "baseline", method_b: str = "Qwen", save_dir: str | None = None):
    df_a = _collect_metrics_over_generations(out_root, method_a)
    df_b = _collect_metrics_over_generations(out_root, method_b)

    # 统一 generation 轴（外连接）
    # df = pd.concat([df_a, df_b], ignore_index=True)

    # def _get(method: str) -> pd.DataFrame:
    #     return df[df["method"] == method].sort_values("generation")

    # A = _get(method_a)
    # B = _get(method_b)
    A = df_a.sort_values("generation")
    B = df_b.sort_values("generation")

    if save_dir is None:
        save_dir = out_root
    os.makedirs(save_dir, exist_ok=True)

    # =========================
    # (1) Confirmed count
    # =========================
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharex=True)
    ax = axes[0]
    ax.plot(A["generation"], A["n_node_conf"], marker="o", label=method_a)
    ax.plot(B["generation"], B["n_node_conf"], marker="o", label=method_b)
    ax.set_title("Confirmed nodes")
    ax.set_xlabel("Generation")
    ax.set_ylabel("n_node_conf")
    ax.legend()

    ax = axes[1]
    ax.plot(A["generation"], A["n_edge_conf"], marker="o", label=method_a)
    ax.plot(B["generation"], B["n_edge_conf"], marker="o", label=method_b)
    ax.set_title("Confirmed edges")
    ax.set_xlabel("Generation")
    ax.set_ylabel("n_edge_conf")
    ax.legend()

    fig.suptitle("Confirmed count @ fixed threshold (nodes / edges)")
    fig.tight_layout()
    p1 = os.path.join(save_dir, "01_confirmed_count.png")
    fig.savefig(p1, dpi=300)
    plt.close(fig)

    # =========================
    # (2) Evidence balance
    # =========================
    fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharex=True)

    ax = axes[0]
    ax.plot(A["generation"], A["node_mss_median"], marker="o", label=method_a)
    ax.plot(B["generation"], B["node_mss_median"], marker="o", label=method_b)
    ax.set_title("Node balance")
    ax.set_xlabel("Generation")
    ax.set_ylabel("node_mss_median (= median min_side_support)")
    ax.legend()

    ax = axes[1]
    ax.plot(A["generation"], A["edge_mcs_top_median"], marker="o", label=method_a)
    ax.plot(B["generation"], B["edge_mcs_top_median"], marker="o", label=method_b)
    ax.set_title("Edge balance (top edges)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("edge_mcs_top_median (= topK median min_cell_support)")
    ax.legend()

    ax = axes[2]
    ax.plot(A["generation"], A["edge_bs_top_median"], marker="o", label=method_a)
    ax.plot(B["generation"], B["edge_bs_top_median"], marker="o", label=method_b)
    ax.set_title("Edge 2x2 balance (top edges)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("edge_bs_top_median (= topK median balanced_support)")
    ax.set_ylim(0.0, 1.0)
    ax.legend()

    fig.suptitle("Evidence balance over generations")
    fig.tight_layout()
    p2 = os.path.join(save_dir, "02_evidence_balance.png")
    fig.savefig(p2, dpi=300)
    plt.close(fig)

    # =========================
    # (3) Trust-weighted strength
    # =========================
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharex=True)

    ax = axes[0]
    ax.plot(A["generation"], A["node_tw_strength_topk"], marker="o", label=method_a)
    ax.plot(B["generation"], B["node_tw_strength_topk"], marker="o", label=method_b)
    ax.set_title("Node trust-weighted strength (topK)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("node_tw_strength_topk")
    ax.legend()

    ax = axes[1]
    ax.plot(A["generation"], A["edge_tw_strength_topk"], marker="o", label=method_a)
    ax.plot(B["generation"], B["edge_tw_strength_topk"], marker="o", label=method_b)
    ax.set_title("Edge trust-weighted strength (topK)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("edge_tw_strength_topk")
    ax.legend()

    fig.suptitle("Trust-weighted evidence strength over generations")
    fig.tight_layout()
    p3 = os.path.join(save_dir, "03_trust_weighted_strength.png")
    fig.savefig(p3, dpi=300)
    plt.close(fig)

    print("Saved plots to:")
    print(" ", p1)
    print(" ", p2)
    print(" ", p3)

# 使用json文件
def compute_kg_evidence_metrics_from_path(method_name: str, out_root: str):
    out_root = os.path.join(out_root, method_name)
    kg_path = os.path.join(out_root, "kg_snapshot.json")
    snap = load_snapshot(kg_path)
    m = compute_kg_evidence_metrics(snap)
    return m

# =========================
# plot iteration log
# =========================
def _load_iter_log(out_root: str, method: str) -> pd.DataFrame:
    path = os.path.join(out_root, method, "iteration_log.csv")
    df = pd.read_csv(path)
    if "iteration" not in df.columns:
        for alt in ["generation", "gen", "iter", "iter_idx"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "iteration"})
                break
    if "iteration" not in df.columns:
        raise KeyError(f"'iteration' column not found in {path}. Columns={list(df.columns)[:30]}")
    df["_source_path"] = path
    return df

def _mean_by_iteration(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = [c for c in df.columns if c not in ("iteration",) and pd.api.types.is_numeric_dtype(df[c])]
    g = df.groupby("iteration", as_index=False)[num_cols].mean()
    return g.sort_values("iteration").reset_index(drop=True)


def plot_iteration_log_reward_items(
    out_root: str,
    method_a: str | None = "baseline",
    method_b: str | None = "Qwen",
    *,
    cols: list[str] | None = None,
    per_fig: int = 25,
    fig_cols: int = 5,
    save_dir: str | None = None,
):
    df_raw_a = df_raw_b = None
    if method_a:
        df_raw_a = _load_iter_log(out_root, method_a)
    if method_b:
        df_raw_b = _load_iter_log(out_root, method_b)

    print("Loaded:")
    if df_raw_a is not None:
        print(" -", method_a, "from", df_raw_a["_source_path"].iloc[0])
    if df_raw_b is not None:
        print(" -", method_b, "from", df_raw_b["_source_path"].iloc[0])

    # plot所有trajectory的平均值
    df_a = _mean_by_iteration(df_raw_a) if df_raw_a is not None else None
    df_b = _mean_by_iteration(df_raw_b) if df_raw_b is not None else None

    # traj = 3  # 只plot一个traj_id
    # df_a = _mean_by_iteration(df_raw_a[df_raw_a["traj_id"] == traj])
    # df_b = _mean_by_iteration(df_raw_b[df_raw_b["traj_id"] == traj])

    deny = set(["gstep", "traj_id", "iteration", "n_edge_set_size",
                "temperature", "thr_auc_std", "floor_spec", "floor_sens",
                "floor_prec", "n_features_total", "n_theta", "n_phi",
                "dom_unknown_not_in_meta_count", "dom_unknown_not_in_meta_ratio",
                ]
    )
    if cols is None:
        # cols = df_a.columns.difference(deny)
        if df_a is not None:
            cols = [c for c in df_a.columns if c not in deny]
        elif df_b is not None:
            cols = [c for c in df_b.columns if c not in deny]
        else:
            cols = []

    priority_cols = ["R", "R_raw", "n_features", "bias0", "flip_rate",
                     "node_confirm", "edge_confirm", "node_fill", "edge_fill", "confirm_term",
                     "fill_term", "perf_score", "perf_term", "size_cost", "n_edge_frontier_size",
                     "node_supp_probe_p10", "node_supp_probe_p50", "edge_supp_probe_p10", "edge_supp_probe_p50", "Gibbs_T",
                     ]
    cols = priority_cols + [c for c in cols if c not in priority_cols]

    if save_dir is None:
        save_dir = out_root
    os.makedirs(save_dir, exist_ok=True)

    n = len(cols)
    pages = math.ceil(n / per_fig)
    paths = []

    for page in range(pages):
        chunk = cols[page * per_fig : (page + 1) * per_fig]
        nplots = len(chunk)
        rows = math.ceil(nplots / fig_cols)

        fig, axes = plt.subplots(rows, fig_cols, figsize=(6 * fig_cols, 3.6 * rows), sharex=True)
        axes = np.array(axes).reshape(-1)

        for i, col in enumerate(chunk):
            ax = axes[i]
            if df_a is not None and col in df_a.columns:
                ax.plot(df_a["iteration"], df_a[col], marker="o", markersize=3, label="Baseline")
            if df_b is not None and col in df_b.columns:
                ax.plot(df_b["iteration"], df_b[col], marker="o", markersize=3, label="LLM")
            ax.set_title(col)
            ax.set_xlabel("Generation (= iteration)")
            ax.grid(True, alpha=0.2)
            if ax.lines:
                ax.legend(loc="best")

        for j in range(nplots, len(axes)):
            axes[j].axis("off")

        fig.suptitle(f"Iteration-log items (mean over traj_id) — page {page+1}/{pages}", y=1.02)
        fig.tight_layout()

        out_path = os.path.join(save_dir, f"reward_items_page_{page+1:02d}.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(out_path)

    return paths, cols

if __name__ == "__main__":
    from plot_all_from_config import main
    main(default_sections=["iterations"])

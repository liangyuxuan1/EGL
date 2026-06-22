from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import re, json
import subprocess
import matplotlib.pyplot as plt

from utils_plot import set_paper_style
set_paper_style()


EDGE_KEY_SEP = "|||"
REQUIRED_TOP_KEYS = ("meta", "nodes", "edges")


# =============================================================================
# IO: KG snapshot
# =============================================================================
def load_kg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        kg = json.load(f)
    missing = [k for k in REQUIRED_TOP_KEYS if k not in kg]
    if missing:
        raise KeyError(
            f"[SCHEMA ERROR] KG snapshot missing top-level keys: {missing}. "
            f"Found keys: {sorted(list(kg.keys()))}"
        )
    if not isinstance(kg["meta"], dict):
        raise TypeError(f"[SCHEMA ERROR] kg['meta'] must be dict, got {type(kg['meta'])}")
    if not isinstance(kg["nodes"], dict):
        raise TypeError(f"[SCHEMA ERROR] kg['nodes'] must be dict, got {type(kg['nodes'])}")
    if not isinstance(kg["edges"], dict):
        raise TypeError(f"[SCHEMA ERROR] kg['edges'] must be dict, got {type(kg['edges'])}")
    return kg


def parse_node_df(kg: dict) -> pd.DataFrame:
    rows = []
    for feat, st in kg["nodes"].items():
        if not isinstance(feat, str) or not feat:
            raise ValueError(f"[SCHEMA ERROR] Invalid feature key in nodes: {feat!r}")
        if not isinstance(st, dict):
            raise TypeError(f"[SCHEMA ERROR] nodes[{feat}] must be dict, got {type(st)}")
        row = {"feature": feat}
        row.update(st)
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # enforce types
    if "n_pair" in df.columns:
        df["n_pair"] = df["n_pair"].fillna(0).astype(int)

    # numeric to float
    for c in df.columns:
        if c == "feature":
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(float)

    return df


def parse_edge_df(kg: dict) -> pd.DataFrame:
    rows = []
    for ek, st in kg["edges"].items():
        if not isinstance(ek, str) or EDGE_KEY_SEP not in ek:
            raise ValueError(f"[SCHEMA ERROR] Invalid edge key in edges (expect 'u{EDGE_KEY_SEP}v'): {ek!r}")
        u, v = ek.split(EDGE_KEY_SEP, 1)
        if not (isinstance(u, str) and isinstance(v, str) and u and v and u != v):
            raise ValueError(f"[SCHEMA ERROR] Invalid edge endpoints parsed from {ek!r}: u={u!r}, v={v!r}")
        if not isinstance(st, dict):
            raise TypeError(f"[SCHEMA ERROR] edges[{ek}] must be dict, got {type(st)}")
        row = {"u": u, "v": v, "edge_key": ek}
        row.update(st)
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # enforce int columns if present
    for c in ["n_pair", "n11", "n10", "n01", "n00"]:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(int)

    # numeric to float
    for c in df.columns:
        if c in ["u", "v", "edge_key"]:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(float)

    return df


_SNAP_RE = re.compile(r"kg_snapshot_(\d+)\.json$")

def _parse_iter_from_snapshot_name(p: Path) -> Optional[int]:
    """
    Parse iteration index from snapshot filename:
        kg_snapshot_000.json -> 0
    Return None if not matched.
    """
    m = _SNAP_RE.search(p.name)
    if not m:
        return None
    return int(m.group(1))



def list_snapshots(snapshot_dir: Path) -> List[Tuple[int, Path]]:
    """
    List snapshot files under snapshot_dir and sort by iteration index.
    Expected files: kg_snapshot_000.json, kg_snapshot_001.json, ...
    """
    if not snapshot_dir.exists():
        raise FileNotFoundError(f"[SNAPSHOT] dir not found: {snapshot_dir}")

    items: List[Tuple[int, Path]] = []
    for p in snapshot_dir.iterdir():
        if not p.is_file():
            continue
        it = _parse_iter_from_snapshot_name(p)
        if it is None:
            continue
        items.append((it, p))

    items.sort(key=lambda x: x[0])
    if not items:
        raise FileNotFoundError(f"[SNAPSHOT] no files matched kg_snapshot_###.json in: {snapshot_dir}")
    return items


# =============================================================================
# Domain mapping
# =============================================================================
def load_domain_map(summary_csv: Path) -> Dict[str, str]:
    """
    Expect columns:
      - var_name
      - clinical_domain
    Missing -> "Other"
    """
    df = pd.read_csv(summary_csv)
    if "var_name" not in df.columns or "clinical_domain" not in df.columns:
        raise KeyError(
            f"[DOMAIN MAP] summary_csv must contain columns var_name and clinical_domain. "
            f"Got: {list(df.columns)}"
        )
    m = {}
    for _, r in df.iterrows():
        k = str(r["var_name"]).strip()
        v = str(r["clinical_domain"]).strip() if pd.notna(r["clinical_domain"]) else "Other"
        if k:
            m[k] = v if v else "Other"
    return m


def attach_domains_nodes(node_df: pd.DataFrame, domain_map: Dict[str, str]) -> pd.DataFrame:
    df = node_df.copy()
    df["domain"] = df["feature"].map(domain_map).fillna("Other")
    return df


def attach_domains_edges(edge_df: pd.DataFrame, domain_map: Dict[str, str]) -> pd.DataFrame:
    df = edge_df.copy()
    df["domain_u"] = df["u"].map(domain_map).fillna("Other")
    df["domain_v"] = df["v"].map(domain_map).fillna("Other")
    # convenient domain-pair label for aggregation / ordering
    df["domain_pair"] = df["domain_u"].astype(str) + " | " + df["domain_v"].astype(str)
    return df


# =============================================================================
# Paper-facing plots (matplotlib)
# =============================================================================
def plot_top_nodes_bar(node_df: pd.DataFrame, out_png: Path, *, tag: str, top_k: int = 30) -> None:
    """
    Show top nodes by effect sign (paired delta_mean), ranked by delta_confidence (|t|).
    We keep labels readable by limiting to top_k.
    """
    df = node_df.sort_values("delta_conf", ascending=False).copy()
    df = df.head(int(top_k)).copy()

    # sort for horizontal bar readability
    df = df.sort_values("delta_mean", ascending=True)

    df["label"] = df["feature"].astype(str).str.replace(r"^diag_", "", regex=True)

    plt.figure(figsize=(6, 4))
    plt.barh(df["label"], df["delta_mean"].astype(float))
    plt.xlabel("Node effect")
    # plt.title(f"[{tag}] Stable nodes (effect Δ) — ranked by StableScore")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_top_edges_bar(edge_df: pd.DataFrame, out_png: Path, *, tag: str, top_k: int = 30) -> None:
    """
    Show top edges by interaction sign (paired delta_mean), ranked by delta_confidence (|t|).
    """
    df = edge_df.sort_values("delta_conf", ascending=False).copy()
    df = df.head(int(top_k)).copy()
    u_lab = df["u"].astype(str).str.replace(r"^diag_", "", regex=True)
    v_lab = df["v"].astype(str).str.replace(r"^diag_", "", regex=True)
    df["label"] = u_lab + "—" + v_lab

    df = df.sort_values("delta_mean", ascending=True)

    plt.figure(figsize=(6, 4))
    plt.barh(df["label"], df["delta_mean"].astype(float))
    plt.xlabel("Edge interaction")
    # plt.title(f"[{tag}] Stable edges (interaction Δ) — ranked by StableScore")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()


# =============================================================================
# Export for R/circlize
# =============================================================================
def export_for_circlize(
    node_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    out_nodes_csv: Path,
    out_edges_csv: Path,
    *,
    max_nodes: int = 20,
    max_edges: int = 80,
    min_edge_score: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    circlize becomes unreadable if there are too many sectors/links.
    We therefore:
      1) pick top nodes by delta_confidence (|t|) (max_nodes)
      2) keep edges whose endpoints are in selected nodes
      3) rank edges by delta_confidence (|t|), keep max_edges
    """
    nodes_sel = node_df.sort_values("delta_conf", ascending=False).head(int(max_nodes)).copy()

    # keep = set(nodes_sel["feature"].astype(str).tolist())
    # edges_f = edge_df[(edge_df["u"].isin(keep)) & (edge_df["v"].isin(keep))].copy()
    # if float(min_edge_score) > 0.0:
    #     edges_f = edges_f[edges_f["stable_score"] >= float(min_edge_score)].copy()
    # edges_sel = edges_f.sort_values("stable_score", ascending=False).head(int(max_edges)).copy()

    edges_sel = edge_df.sort_values("delta_conf", ascending=False).copy()
    edges_sel = edges_sel.head(int(max_edges)).copy()

    # Circos inputs: keep only necessary columns to make R script stable
    nodes_out = nodes_sel[[
        "feature", "domain", "n_pair", "delta_mean", "delta_t", "delta_conf"
    ]].copy()

    edges_out = edges_sel[[
        "u", "v", "domain_u", "domain_v", "n_pair", "delta_mean", "delta_t", "delta_conf"
    ]].copy()
    nodes_out["edge_only"] = False

    # Add edge endpoints not in nodes_out
    edge_nodes = set(edges_out["u"].astype(str).tolist()) | set(edges_out["v"].astype(str).tolist())
    missing_nodes = sorted(edge_nodes - set(nodes_out["feature"].astype(str).tolist()))
    if missing_nodes:
        extra = node_df[node_df["feature"].astype(str).isin(missing_nodes)][[
            "feature", "domain", "n_pair", "delta_mean", "delta_t", "delta_conf"
        ]].copy()
        extra["edge_only"] = True
        nodes_out_both = pd.concat([nodes_out, extra], ignore_index=True)
    else:
        nodes_out_both = nodes_out

    nodes_out_both.to_csv(out_nodes_csv, index=False)
    edges_out.to_csv(out_edges_csv, index=False)

    return nodes_out_both, edges_out


def run_r_circlize(r_script: Path, nodes_csv: Path, edges_csv: Path, out_png: Path, *, tag: str) -> None:
    """
    Call Rscript to render chord diagram.
    """
    cmd = [
        "Rscript",
        str(r_script),
        "--nodes_csv", str(nodes_csv),
        "--edges_csv", str(edges_csv),
        "--out_png", str(out_png),
        "--tag", str(tag),
    ]
    print("[R] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


# =============================================================================
# Pipeline for last snapshot
# =============================================================================
def process_one(
    kg_path: Path,
    out_dir: Path,
    *,
    tag: str,
    domain_map: Dict[str, str],
    top_k_bar: int,
    circlize_max_nodes: int,
    circlize_max_edges: int,
    circlize_min_edge_score: float,
    r_script: Path | None,
    run_r: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    kg = load_kg(kg_path)

    node_df = parse_node_df(kg)
    edge_df = parse_edge_df(kg)

    node_df = attach_domains_nodes(node_df, domain_map)
    edge_df = attach_domains_edges(edge_df, domain_map)

    # Save full evidence tables (with stable scores)
    out_dir.mkdir(parents=True, exist_ok=True)
    node_df.to_csv(out_dir / f"{tag}_nodes_evidence.csv", index=False)
    edge_df.to_csv(out_dir / f"{tag}_edges_evidence.csv", index=False)

    # Paper-facing bar plots
    plot_top_nodes_bar(
        node_df=node_df,
        out_png=out_dir / f"{tag}_top_nodes_bar.png",
        tag=tag,
        top_k=int(top_k_bar),
    )
    plot_top_edges_bar(
        edge_df=edge_df,
        out_png=out_dir / f"{tag}_top_edges_bar.png",
        tag=tag,
        top_k=int(top_k_bar),
    )

    # Export for R/circlize
    nodes_csv = out_dir / f"{tag}_circlize_nodes.csv"
    edges_csv = out_dir / f"{tag}_circlize_edges.csv"
    export_for_circlize(
        node_df=node_df,
        edge_df=edge_df,
        out_nodes_csv=nodes_csv,
        out_edges_csv=edges_csv,
        max_nodes=int(circlize_max_nodes),
        max_edges=int(circlize_max_edges),
        min_edge_score=float(circlize_min_edge_score),
    )

    # Optional: call Rscript to generate chord diagram
    if run_r == "yes":
        if r_script is None:
            raise ValueError("--run_r is set but --r_script is not provided.")
        out_png = out_dir / f"{tag}_circlize_chord.png"
        run_r_circlize(r_script=r_script, nodes_csv=nodes_csv, edges_csv=edges_csv, out_png=out_png, tag=tag)

    return node_df, edge_df


def process_both(
    kg_path_a: Path,
    kg_path_b: Path,
    out_dir: Path,
    *,
    tag_a: str,
    tag_b: str,
    domain_map: Dict[str, str],
    top_k_bar: int,
) -> None:
    kg_a = load_kg(kg_path_a)
    kg_b = load_kg(kg_path_b)

    node_df_a = parse_node_df(kg_a)
    edge_df_a = parse_edge_df(kg_a)
    node_df_b = parse_node_df(kg_b)
    edge_df_b = parse_edge_df(kg_b)

    node_df_a = attach_domains_nodes(node_df_a, domain_map)
    edge_df_a = attach_domains_edges(edge_df_a, domain_map)
    node_df_b = attach_domains_nodes(node_df_b, domain_map)
    edge_df_b = attach_domains_edges(edge_df_b, domain_map)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- combined nodes bar ----
    # 按照confirmatioion排序，取top_k_bar个，按照delta_t画图
    def _prep_nodes(df: pd.DataFrame) -> pd.DataFrame:
        d = df.sort_values("delta_conf", ascending=False).head(int(top_k_bar)).copy()
        d = d.sort_values("delta_mean", ascending=True)
        d["label"] = d["feature"].astype(str).str.replace(r"^diag_", "", regex=True)
        return d

    df_a_nodes = _prep_nodes(node_df_a)
    df_b_nodes = _prep_nodes(node_df_b)
    xlim_nodes = float(max(
        df_a_nodes["delta_mean"].abs().max() if not df_a_nodes.empty else 0.0,
        df_b_nodes["delta_mean"].abs().max() if not df_b_nodes.empty else 0.0,
        1e-6,
    ))

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 4))
    ax_l.barh(df_a_nodes["label"], df_a_nodes["delta_mean"].astype(float))
    ax_l.set_title(f"{tag_a}")
    ax_l.set_xlabel("Node effect")
    # ax_l.set_xlim(-xlim_nodes, xlim_nodes)

    ax_r.barh(df_b_nodes["label"], df_b_nodes["delta_mean"].astype(float))
    ax_r.set_title(f"{tag_b}")
    ax_r.set_xlabel("Node effect")
    # ax_r.set_xlim(-xlim_nodes, xlim_nodes)

    fig.tight_layout()
    out_png_nodes = out_dir / "both_top_nodes_bar.png"
    fig.savefig(out_png_nodes, dpi=300)
    plt.close(fig)

    # ---- combined edges bar ----
    def _prep_edges(df: pd.DataFrame) -> pd.DataFrame:
        d = df.sort_values("delta_conf", ascending=False).head(int(top_k_bar)).copy()
        u_lab = d["u"].astype(str).str.replace(r"^diag_", "", regex=True)
        v_lab = d["v"].astype(str).str.replace(r"^diag_", "", regex=True)
        d["label"] = u_lab + "—" + v_lab
        d = d.sort_values("delta_mean", ascending=True)
        return d

    df_a_edges = _prep_edges(edge_df_a)
    df_b_edges = _prep_edges(edge_df_b)
    xlim_edges = float(max(
        df_a_edges["delta_mean"].abs().max() if not df_a_edges.empty else 0.0,
        df_b_edges["delta_mean"].abs().max() if not df_b_edges.empty else 0.0,
        1e-6,
    ))

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 4))
    ax_l.barh(df_a_edges["label"], df_a_edges["delta_mean"].astype(float))
    ax_l.set_title(f"{tag_a}")
    ax_l.set_xlabel("Edge interaction")
    # ax_l.set_xlim(-xlim_edges, xlim_edges)

    ax_r.barh(df_b_edges["label"], df_b_edges["delta_mean"].astype(float))
    ax_r.set_title(f"{tag_b}")
    ax_r.set_xlabel("Edge interaction")
    # ax_r.set_xlim(-xlim_edges, xlim_edges)

    fig.tight_layout()
    out_png_edges = out_dir / "both_top_edges_bar.png"
    fig.savefig(out_png_edges, dpi=300)
    plt.close(fig)


def process_both_compact(
    kg_path_a: Path,
    kg_path_b: Path,
    out_dir: Path,
    *,
    tag_a: str,
    tag_b: str,
    domain_map: Dict[str, str],
    top_k_bar: int,
) -> None:
    kg_a = load_kg(kg_path_a)
    kg_b = load_kg(kg_path_b)

    node_df_a = parse_node_df(kg_a)
    edge_df_a = parse_edge_df(kg_a)
    node_df_b = parse_node_df(kg_b)
    edge_df_b = parse_edge_df(kg_b)

    node_df_a = attach_domains_nodes(node_df_a, domain_map)
    edge_df_a = attach_domains_edges(edge_df_a, domain_map)
    node_df_b = attach_domains_nodes(node_df_b, domain_map)
    edge_df_b = attach_domains_edges(edge_df_b, domain_map)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- combined nodes bar ----
    def _prep_nodes(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
        d = df.sort_values("delta_conf", ascending=False).head(int(top_k)).copy()
        d = d.sort_values("delta_mean", ascending=True)
        d["label"] = d["feature"].astype(str).str.replace(r"^diag_", "", regex=True)
        return d

    df_a_nodes = _prep_nodes(node_df_a, top_k=top_k_bar)
    df_b_nodes = _prep_nodes(node_df_b, top_k=top_k_bar)
    a_map = dict(zip(df_a_nodes["label"], df_a_nodes["delta_mean"].astype(float)))
    b_map = dict(zip(df_b_nodes["label"], df_b_nodes["delta_mean"].astype(float)))
    labels = list(set(a_map.keys()) | set(b_map.keys()))
    labels.sort(key=lambda lbl: a_map.get(lbl, b_map.get(lbl, 0.0)))

    y = np.arange(len(labels))
    h = 0.38

    # MICCAI 论文使用
    fig, ax = plt.subplots(1, 1, figsize=(2.3, 2.0))
    ax.barh(
        y - h/2,
        [a_map.get(lbl, np.nan) for lbl in labels],
        height=h,
        label=tag_a,
    )
    ax.barh(
        y + h/2,
        [b_map.get(lbl, np.nan) for lbl in labels],
        height=h,
        label=tag_b,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Node effect")
    ax.grid(True, axis="x", linewidth=0.5, alpha=0.5)
    leg = ax.legend(loc="best", frameon=True, edgecolor="lightgray", fontsize=5, framealpha=1.0)
    leg.get_frame().set_linewidth(0.5)

    fig.tight_layout()
    out_png_nodes = out_dir / "both_top_nodes_bar_compact.png"
    fig.savefig(out_png_nodes, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- combined edges bar ----
    def _prep_edges(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
        d = df.sort_values("delta_conf", ascending=False).head(int(top_k)).copy()
        u_lab = d["u"].astype(str).str.replace(r"^diag_", "", regex=True)
        v_lab = d["v"].astype(str).str.replace(r"^diag_", "", regex=True)
        d["label"] = u_lab + "—" + v_lab
        d = d.sort_values("delta_mean", ascending=True)
        return d

    df_a_edges = _prep_edges(edge_df_a, top_k=top_k_bar)
    df_b_edges = _prep_edges(edge_df_b, top_k=top_k_bar)

    # Draw part of top positive and negative edges.
    def _filter_edges_by_delta(df: pd.DataFrame, pos_keep: int = 5, neg_keep: int = 5) -> pd.DataFrame:
        if df.empty:
            return df
        pos = df[df["delta_mean"] >= 0].nlargest(pos_keep, "delta_mean")
        neg = df[df["delta_mean"] < 0].nsmallest(neg_keep, "delta_mean")
        return pd.concat([neg, pos], axis=0).sort_values("delta_mean", ascending=True)

    df_a_edges = _filter_edges_by_delta(df_a_edges)
    df_b_edges = _filter_edges_by_delta(df_b_edges)
    a_map = dict(zip(df_a_edges["label"], df_a_edges["delta_mean"].astype(float)))
    b_map = dict(zip(df_b_edges["label"], df_b_edges["delta_mean"].astype(float)))
    labels = list(set(a_map.keys()) | set(b_map.keys()))
    labels.sort(key=lambda lbl: a_map.get(lbl, b_map.get(lbl, 0.0)))

    y = np.arange(len(labels))
    h = 0.38

    fig, ax = plt.subplots(1, 1, figsize=(2.6, 2.0))
    ax.barh(
        y - h/2,
        [a_map.get(lbl, np.nan) for lbl in labels],
        height=h,
        label=tag_a,
    )
    ax.barh(
        y + h/2,
        [b_map.get(lbl, np.nan) for lbl in labels],
        height=h,
        label=tag_b,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Edge interaction")
    ax.grid(True, axis="x", linewidth=0.5, alpha=0.5)
    leg = ax.legend(loc="best", frameon=True, edgecolor="lightgray", fontsize=5, framealpha=0.8)
    leg.get_frame().set_linewidth(0.5)

    fig.tight_layout()
    out_png_edges = out_dir / "both_top_edges_bar_compact.png"
    fig.savefig(out_png_edges, dpi=300, bbox_inches="tight")
    plt.close(fig)

# -------------------------
# Convergence analysis over snapshots
# -------------------------
def _canon_edge_key(u: str, v: str, sep: str = EDGE_KEY_SEP) -> str:
    a, b = str(u), str(v)
    return f"{a}{sep}{b}" if a <= b else f"{b}{sep}{a}"


def _ensure_edge_id(edf: pd.DataFrame) -> pd.DataFrame:
    """Add `edge_id` column: prefer edge_key else canonical u|||v (undirected)."""
    if edf is None or edf.empty:
        return edf
    dff = edf.copy()
    if "edge_key" in dff.columns and dff["edge_key"].notna().any():
        dff["edge_id"] = dff["edge_key"].astype(str)
    else:
        if "u" not in dff.columns or "v" not in dff.columns:
            raise KeyError("[EDGE DF] missing u/v columns.")
        dff["edge_id"] = [
            _canon_edge_key(u, v) for u, v in zip(dff["u"].astype(str), dff["v"].astype(str))
        ]
    return dff


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation without scipy: corr(rank(x), rank(y))."""
    if len(x) < 2:
        return np.nan
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    sx = np.std(rx)
    sy = np.std(ry)
    if sx == 0 or sy == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def _safe_sign(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Sign with small epsilon to avoid noisy 0."""
    s = np.sign(x)
    s[np.abs(x) < eps] = 0.0
    return s


def _select_core_from_last(
    df_last: pd.DataFrame,
    *,
    id_col: str,
    top_k: int,
    score_col: str = "delta_t",
    min_n_pair: int = 0,
    min_abs_t: float = 0.0,
) -> List[str]:
    """
    Core = top_k by |score_col| in last snapshot, with optional filters.
    score_col default uses delta_t (more comparable across scales).
    """
    if df_last is None or df_last.empty:
        return []
    dff = df_last.copy()

    if id_col not in dff.columns:
        raise KeyError(f"[CORE SELECT] missing id_col={id_col}")
    if score_col not in dff.columns:
        raise KeyError(f"[CORE SELECT] missing score_col={score_col}")
    if "n_pair" not in dff.columns:
        raise KeyError("[CORE SELECT] missing n_pair")

    dff = dff[dff["n_pair"].fillna(0).astype(float) >= float(min_n_pair)].copy()
    if min_abs_t > 0:
        dff = dff[dff[score_col].abs().fillna(0) >= float(min_abs_t)].copy()

    if dff.empty:
        return []
    dff["__score_abs"] = dff[score_col].abs()
    dff = dff.sort_values("__score_abs", ascending=False).head(int(top_k))
    return dff[id_col].astype(str).tolist()


def _compute_convergence_metrics(
    df_t: pd.DataFrame,
    df_last: pd.DataFrame,
    *,
    id_col: str,
    core_ids: List[str],
    mean_col: str = "delta_mean",
    se_col: str = "delta_se",
    t_col: str = "delta_t",
) -> Dict[str, float]:
    """
    Compute convergence metrics for snapshot t relative to last snapshot, restricted to core ids.
    """
    if len(core_ids) == 0:
        return dict(
            core_k=0,
            mae_vs_last=np.nan,
            maxae_vs_last=np.nan,
            rmse_vs_last=np.nan,
            sign_flip_rate=np.nan,
            spearman_mean=np.nan,
            spearman_abs_t=np.nan,
            mean_se=np.nan,
            mean_n_pair=np.nan,
            coverage_rate=np.nan,  # fraction of core ids present in this snapshot
        )

    # index by id
    A = df_t.set_index(id_col, drop=False)
    B = df_last.set_index(id_col, drop=False)

    present = [cid for cid in core_ids if cid in A.index and cid in B.index]
    coverage = len(present) / max(1, len(core_ids))

    if len(present) == 0:
        out = dict(
            core_k=len(core_ids),
            mae_vs_last=np.nan,
            maxae_vs_last=np.nan,
            rmse_vs_last=np.nan,
            sign_flip_rate=np.nan,
            spearman_mean=np.nan,
            spearman_abs_t=np.nan,
            mean_se=np.nan,
            mean_n_pair=np.nan,
            coverage_rate=float(coverage),
        )
        return out

    a_mean = A.loc[present, mean_col].astype(float).to_numpy()
    b_mean = B.loc[present, mean_col].astype(float).to_numpy()
    diff = a_mean - b_mean
    absdiff = np.abs(diff)

    mae = float(np.mean(absdiff))
    maxae = float(np.max(absdiff))
    rmse = float(np.sqrt(np.mean(diff * diff)))

    # sign flips: compare sign(delta_mean) vs last
    sa = _safe_sign(a_mean)
    sb = _safe_sign(b_mean)
    # count flips excluding zeros (optional; here count any mismatch when both nonzero)
    mask = (sa != 0) & (sb != 0)
    if np.any(mask):
        flip = float(np.mean((sa[mask] != sb[mask]).astype(float)))
    else:
        flip = np.nan

    # correlations: mean and abs(t)
    a_t = A.loc[present, t_col].astype(float).to_numpy()
    b_t = B.loc[present, t_col].astype(float).to_numpy()
    spearman_mean = _spearman_corr(a_mean, b_mean)
    spearman_abs_t = _spearman_corr(np.abs(a_t), np.abs(b_t))

    # summary uncertainty/support
    mean_se = float(np.mean(A.loc[present, se_col].astype(float).to_numpy())) if se_col in A.columns else np.nan
    mean_n_pair = float(np.mean(A.loc[present, "n_pair"].astype(float).to_numpy())) if "n_pair" in A.columns else np.nan

    return dict(
        core_k=int(len(core_ids)),
        mae_vs_last=mae,
        maxae_vs_last=maxae,
        rmse_vs_last=rmse,
        sign_flip_rate=flip,
        spearman_mean=float(spearman_mean),
        spearman_abs_t=float(spearman_abs_t),
        mean_se=mean_se,
        mean_n_pair=mean_n_pair,
        coverage_rate=float(coverage),
    )


def analyze_snapshot_dir_convergence(
    snapshot_dir: Path,
    *,
    tag: str,
    # core definition from LAST snapshot
    core_top_k_nodes: int = 15,
    core_top_k_edges: int = 15,
    core_score_col: str = "delta_t",   # robust default
    core_min_n_pair_node: int = 20,
    core_min_n_pair_edge: int = 4,
    core_min_abs_t: float = 0.0,
) -> pd.DataFrame:
    """
    Evidence convergence analysis over a snapshot directory.

    For each snapshot t, we compute metrics relative to the LAST snapshot, restricted to
    core nodes / core edges defined from the last snapshot.

    Node core IDs: last.topK by |delta_t| (or chosen score_col) with n_pair>=min.
    Edge core IDs: same but on edges.

    Output columns (per snapshot):
      - node_*: mae_vs_last / maxae_vs_last / rmse_vs_last / sign_flip_rate / spearman_* / mean_se / mean_n_pair / coverage_rate
      - edge_*: same
    """

    snaps = list_snapshots(snapshot_dir)
    if len(snaps) == 0:
        return pd.DataFrame([])

    # ---- load LAST snapshot ----
    it_last, p_last = snaps[-1]
    kg_last = load_kg(p_last)
    ndf_last = parse_node_df(kg_last)
    edf_last = parse_edge_df(kg_last)

    # add IDs
    ndf_last = ndf_last.copy()
    if "feature" not in ndf_last.columns:
        raise KeyError("[NODE DF] missing feature column.")
    ndf_last["node_id"] = ndf_last["feature"].astype(str)

    edf_last = _ensure_edge_id(edf_last)

    # ---- define core sets from LAST ----
    core_nodes = _select_core_from_last(
        ndf_last, id_col="node_id",
        top_k=core_top_k_nodes,
        score_col=core_score_col,
        min_n_pair=core_min_n_pair_node,
        min_abs_t=core_min_abs_t,
    )
    core_edges = _select_core_from_last(
        edf_last, id_col="edge_id",
        top_k=core_top_k_edges,
        score_col=core_score_col,
        min_n_pair=core_min_n_pair_edge,
        min_abs_t=core_min_abs_t,
    )

    rows: List[Dict[str, Any]] = []
    for it, p in snaps:
        kg = load_kg(p)
        ndf = parse_node_df(kg)
        edf = parse_edge_df(kg)

        # IDs
        ndf = ndf.copy()
        ndf["node_id"] = ndf["feature"].astype(str)
        edf = _ensure_edge_id(edf)

        node_metrics = _compute_convergence_metrics(
            ndf, ndf_last, id_col="node_id", core_ids=core_nodes
        )
        edge_metrics = _compute_convergence_metrics(
            edf, edf_last, id_col="edge_id", core_ids=core_edges
        )

        row = {"tag": tag, "iter": int(it)}
        for k, v in node_metrics.items():
            row[f"node_{k}"] = v
        for k, v in edge_metrics.items():
            row[f"edge_{k}"] = v
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("iter").reset_index(drop=True)

    # attach some meta for logging / debugging
    df.attrs["core_nodes"] = core_nodes
    df.attrs["core_edges"] = core_edges
    df.attrs["iter_last"] = int(it_last)
    return df


def plot_convergence_curves(
    df_list: List[pd.DataFrame],
    *,
    title: str = "Evidence Convergence vs Final Snapshot",
    outpath: Optional[Path] = None,
    smooth_window: int = 1,  # set >1 if you want rolling mean smoothing
) -> None:
    """
    Plot convergence curves for multiple runs (e.g., baseline vs LLM).
    Each df must contain: tag, iter, and the node_/edge_ metrics from analyze_snapshot_dir_convergence.
    """

    def _maybe_smooth(y: np.ndarray) -> np.ndarray:
        if smooth_window is None or smooth_window <= 1:
            return y
        s = pd.Series(y).rolling(window=int(smooth_window), min_periods=1, center=False).mean().to_numpy()
        return s

    # Choose metrics to plot (you can edit this list)
    panels = [
        ("node_mae_vs_last", "Node core: MAE(|Δmean(t)-Δmean(last)|)"),
        ("node_spearman_abs_t", "Node core: Spearman corr of |t| vs last"),
        ("edge_mae_vs_last", "Edge core: MAE(|Δmean(t)-Δmean(last)|)"),
        ("edge_spearman_abs_t", "Edge core: Spearman corr of |t| vs last"),
    ]

    fig = plt.figure(figsize=(11, 8))
    fig.suptitle(title)

    n = len(panels)
    nrows = 2
    ncols = 2

    for idx, (col, ylabel) in enumerate(panels, start=1):
        ax = fig.add_subplot(nrows, ncols, idx)
        for df in df_list:
            if df is None or df.empty:
                continue
            tag = str(df["tag"].iloc[0]) if "tag" in df.columns else "run"
            x = df["iter"].to_numpy()
            y = df[col].to_numpy() if col in df.columns else np.full_like(x, np.nan, dtype=float)
            y = _maybe_smooth(y)
            ax.plot(x, y, marker="o", linewidth=1.5, markersize=3, label=tag)

        ax.set_xlabel("iteration")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    if outpath is not None:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)

        outpng = outpath / "convergence_curves.png"
        plt.savefig(outpng, dpi=300)
        print(f"Saved plot to {outpng}")
        plt.close(fig)


def plot_convergence_curves_separate(
    df_list: List[pd.DataFrame],
    *,
    outpath: Optional[Path] = None,
    smooth_window: int = 1,  # set >1 if you want rolling mean smoothing
) -> None:
    """
    Plot convergence curves for multiple runs (e.g., baseline vs LLM).
    Each df must contain: tag, iter, and the node_/edge_ metrics from analyze_snapshot_dir_convergence.
    """

    def _maybe_smooth(y: np.ndarray) -> np.ndarray:
        if smooth_window is None or smooth_window <= 1:
            return y
        s = pd.Series(y).rolling(window=int(smooth_window), min_periods=1, center=False).mean().to_numpy()
        return s

    # Choose metrics to plot (you can edit this list)
    panels = [
        ("node_mae_vs_last", "Node core MAE"),
        ("node_spearman_abs_t", "Node core corr"),
        ("edge_mae_vs_last", "Edge core MAE"),
        ("edge_spearman_abs_t", "Edge core corr"),
    ]

    for idx, (col, ylabel) in enumerate(panels, start=1):
        fig, ax = plt.subplots(figsize=(2.5, 1.5))
        for df in df_list:
            if df is None or df.empty:
                continue
            tag = str(df["tag"].iloc[0]) if "tag" in df.columns else "run"
            x = df["iter"].to_numpy()
            y = df[col].to_numpy() if col in df.columns else np.full_like(x, np.nan, dtype=float)
            y = _maybe_smooth(y)
            ax.plot(x, y, linewidth=1.0, label=tag)

            ax.set_xlabel("Iteration")
            ax.set_ylabel(ylabel)
            ax.grid(True, linewidth=0.5, alpha=0.5)
            ax.legend()

        plt.tight_layout()

        outpng = Path(outpath) / f"{col}.png"
        outpng.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpng, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {outpng}")
        plt.close(fig)



def plot_convergence_figures(results_root: str, method_a: str, method_b: str, tag_a: str, tag_b: str, top_k_nodes: int, top_k_edges: int, save_dir: str):
    """
    |t|：证据强度 + 已经内含方差/样本量影响，用来定义“core evidence”最合理，也最像统计检验语言。
    delta_mean：临床解释性最好（方向、大小直观）。你把它用于展示，不会让人误以为你在用一个“搜索期启发式分数”做科学结论

    Core definition： “We define the core evidence as the final-snapshot top-K nodes/edges ranked by |t|,
    reflecting evidence strength under the controlled 2×2 probe design.”

    展示口径： “For interpretability, we visualize the core elements using signed effect sizes (Delta——mean)
    while keeping selection and convergence analysis based on |t|.”
    """
    baseline_dir = Path(f"{results_root}/{method_a}/kg_snapshot")
    llm_dir = Path(f"{results_root}/{method_b}/kg_snapshot")

    df_base = analyze_snapshot_dir_convergence(
        baseline_dir, tag=tag_a,
        core_top_k_nodes=top_k_nodes, core_top_k_edges=top_k_edges,
        core_min_n_pair_node=20, core_min_n_pair_edge=4,
        core_score_col="delta_t",
    )

    df_llm = analyze_snapshot_dir_convergence(
        llm_dir, tag=tag_b,
        core_top_k_nodes=top_k_nodes, core_top_k_edges=top_k_edges,
        core_min_n_pair_node=20, core_min_n_pair_edge=4,
        core_score_col="delta_t",
    )

    plot_convergence_curves(
        [df_base, df_llm],
        title="Convergence of Core Evidence to Final Snapshot (EGL vs EGL-LLM)",
        outpath=save_dir,
        smooth_window=1,
    )

    plot_convergence_curves_separate(
        [df_base, df_llm],
        outpath=save_dir,
        smooth_window=1,
    )


def plot_node_edge_figures(kg1: str, kg2: str, out_dir: str, tag_a: str, tag_b: str, run_r: str = "yes", domain_map_csv: str | Path = "outputs/ct_clin_integration/COPD_ct_variable_summary.csv"):
    kg1 = Path(kg1)
    kg2 = Path(kg2)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = Path(domain_map_csv)
    domain_map = load_domain_map(summary_csv)

    top_k = 15
    circlize_max_nodes = 10
    circlize_max_edges = 10
    circlize_min_edge_score = 0.0

    run_r = str(run_r)
    r_script = Path("S04S05_plot_stable_circlize.R")

    # snapshot 1
    process_one(
        kg_path=kg1,
        out_dir=out_dir,
        tag=tag_a,
        domain_map=domain_map,
        top_k_bar=top_k,
        circlize_max_nodes=circlize_max_nodes,
        circlize_max_edges=circlize_max_edges,
        circlize_min_edge_score=circlize_min_edge_score,
        r_script=r_script,
        run_r=run_r,
    )

    process_one(
        kg_path=kg2,
        out_dir=out_dir,
        tag=tag_b,
        domain_map=domain_map,
        top_k_bar=top_k,
        circlize_max_nodes=circlize_max_nodes,
        circlize_max_edges=circlize_max_edges,
        circlize_min_edge_score=circlize_min_edge_score,
        r_script=r_script,
        run_r=run_r,
    )

    process_both(
        kg_path_a=Path(kg1),
        kg_path_b=Path(kg2),
        out_dir=out_dir,
        tag_a=str(tag_a),
        tag_b=str(tag_b),
        domain_map=domain_map,
        top_k_bar=top_k,
    )

    top_k = 10  # 只画10个top nodes的union，edge更少
    process_both_compact(
        kg_path_a=Path(kg1),
        kg_path_b=Path(kg2),
        out_dir=out_dir,
        tag_a=str(tag_a),
        tag_b=str(tag_b),
        domain_map=domain_map,
        top_k_bar=top_k,
    )


if __name__ == "__main__":
    from plot_all_from_config import main
    main(default_sections=["stable_kg"])

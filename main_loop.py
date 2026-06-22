#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
main — Parallel (interleaving) with shared Global Knowledge Graph (KG)
---------------------------------------------------------------------------
"""
from __future__ import annotations

import os, json, time
import argparse
import math
import random
from dataclasses import dataclass, field, asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Sequence
from collections import Counter
from datetime import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
import hashlib

# 工程依赖
from config_cvd import CVD_EVENT_COLS
from utils_graph import KnowledgeGraph, set_to_bitvec
import utils_models
import utils_cache
import utils_dataset
from utils_dataset import DatasetCtx, ExperimentSpec

from utils_control import OnlineThresholdState, RewardWeights
import utils_control

import utils

# ================================================================
# 2026-02-06: 主程序中有用的函数，会逐步整理到这里
# ================================================================
def run_warmup_calibration(
    *,
    df_all: pd.DataFrame,
    y_all: np.ndarray,
    pool_schema: Dict[str, Any],
    pool_meta: pd.DataFrame,
    all_features: List[str],
    ots: OnlineThresholdState,
    out_root: Path,
    model_name: str = "Logistic",
    # stratified sampling plan
    k_grid: Sequence[int] = (10, 15, 20, 25),
    sets_per_k: int = 10,          # total warmup CV runs = len(k_grid) * sets_per_k
    anchor_frac: float = 0.30,     # fraction of anchors in each set
    seed: int = 123,
    # anchor pool controls
    missing_rate_max: float = 0.30,
    prefer_non_diag: bool = True,
) -> Path:
    """
    Threshold calibration stage (objective-free).

    What it does:
      1) Generate stratified-random feature sets across k_grid.
      2) Evaluate each set with your CV evaluator.
      3) Update OnlineThresholdState (GLOBAL buckets only):
           - auc_std distribution (for unstable threshold)
           - spec/sens/prec distributions (for quantile floors)
      4) Persist the calibrated warmup parameters as JSON

    Returns:
      Path to warmup_params.json
    """
    out_root.mkdir(parents=True, exist_ok=True)
    warmup_json = out_root / "warmup_params.json"

    rng = random.Random(int(seed))

    # ---------------------------
    # Build anchor pool
    # ---------------------------
    # Anchor pool aims to avoid "all garbage" feature sets during warmup.
    # We use low missingness and optionally downweight diag_* features.
    meta = pool_meta.copy()
    if "var_name" not in meta.columns:
        raise ValueError("pool_meta must have column 'var_name'")
    meta["var_name"] = meta["var_name"].astype(str)

    feat_set = set(map(str, all_features))
    meta = meta[meta["var_name"].isin(feat_set)]

    if "missing_rate" in meta.columns:
        meta["missing_rate"] = pd.to_numeric(meta["missing_rate"], errors="coerce")
    else:
        meta["missing_rate"] = np.nan

    def _is_diag(v: str) -> bool:
        return str(v).startswith("diag_")

    # base anchor candidates: low missingness if available, else everyone
    if meta["missing_rate"].notna().any():
        anchors_df = meta[meta["missing_rate"].fillna(1.0) <= float(missing_rate_max)].copy()
    else:
        anchors_df = meta.copy()

    if prefer_non_diag:
        # keep diag_* but deprioritize by dropping most of them from anchor pool
        anchors_df = anchors_df[~anchors_df["var_name"].map(_is_diag)].copy()

    anchor_pool = anchors_df["var_name"].tolist()
    if len(anchor_pool) < 30:
        # If anchor_pool is too small, fall back to full feature pool (still okay).
        anchor_pool = list(map(str, all_features))

    all_pool = list(map(str, all_features))

    # ---------------------------
    # Warmup loop
    # ---------------------------
    rows: List[Dict[str, Any]] = []
    run_id = 0

    for k in list(k_grid):
        k = int(k)
        if k <= 0:
            continue

        for j in tqdm(range(int(sets_per_k)), desc=f"Warmup k={k}"):
            run_id += 1

            # --- sample one feature set ---
            n_anchor = int(round(float(anchor_frac) * k))
            n_anchor = max(0, min(k, n_anchor))

            # sample anchors without replacement
            anchors = rng.sample(anchor_pool, k=min(n_anchor, len(anchor_pool))) if n_anchor > 0 else []
            anchors = list(dict.fromkeys(anchors))

            # fill the rest from global pool, avoiding duplicates
            remaining = k - len(anchors)
            if remaining > 0:
                rest_candidates = [f for f in all_pool if f not in set(anchors)]
                # if pool smaller than needed, allow wrap (shouldn't happen for real pool sizes)
                if len(rest_candidates) >= remaining:
                    rest = rng.sample(rest_candidates, k=remaining)
                else:
                    rest = rest_candidates
                feat_set_k = anchors + rest
            else:
                feat_set_k = anchors

            # final safety
            feat_set_k = list(dict.fromkeys([f for f in feat_set_k if f in feat_set]))
            if len(feat_set_k) < max(3, int(0.5 * k)):
                # if it collapsed too much, refill purely random
                feat_set_k = rng.sample(all_pool, k=min(k, len(all_pool)))

            # --- evaluate ---
            selected_schema = {f: pool_schema[f] for f in feat_set_k if f in pool_schema}

            # IMPORTANT: warmup is objective-free -> do NOT pass objective to ots
            # The evaluator is treated as a black box.
            metrics_summary, cv_lines, dropped_msg, hard_index_pack, fold_df = utils_models.evaluate_auc_and_more_cv(
                df=df_all,
                label=y_all,
                feature_schema=selected_schema,
                model_name=str(model_name),
                seed=int(seed),
            )
            metrics_summary = metrics_summary or {}
            metrics_summary["n_features"] = int(len(feat_set_k))

            # --- ingest into ots (GLOBAL buckets only) ---
            # objective=None or "NA" will skip obj×phase buckets by design.
            ots.update_from_metrics_summary(
                metrics_summary,
            )

            # --- log one row ---
            row = {
                "warmup_id": int(run_id),
                "k": int(k),
                "n_features": int(len(feat_set_k)),
                "features_json": json.dumps(feat_set_k, ensure_ascii=True),
                "auc_mean": float(metrics_summary.get("auc_mean", np.nan)),
                "auc_std": float(metrics_summary.get("auc_std", np.nan)),
                "specificity_mean": float(metrics_summary.get("specificity_mean", np.nan)),
                "sensitivity_mean": float(metrics_summary.get("sensitivity_mean", np.nan)),
                "precision_mean": float(metrics_summary.get("precision_mean", np.nan)),
                "auprc_mean": float(metrics_summary.get("auprc_mean", np.nan)),
                "ece_mean": float(metrics_summary.get("ece_mean", np.nan)),
                "brier_mean": float(metrics_summary.get("brier_mean", np.nan)),
                "dropped_msg": str(dropped_msg)[:200] if dropped_msg is not None else "",
            }
            rows.append(row)

    q_auc = ots.get_auc_std_threshold()
    q_spec = ots.get_spec_floor()
    q_sens = ots.get_sens_floor()
    q_prec = ots.get_prec_floor()

    # Freeze warmup-derived thresholds/floors into the fixed_* fields used by reward.
    ots.fixed_auc_std = float(q_auc.threshold)
    ots.fixed_spec_floor = float(q_spec.threshold)
    ots.fixed_sens_floor = float(q_sens.threshold)
    ots.fixed_prec_floor = float(q_prec.threshold)

    payload = {
        "warmup": {
            "model_name": str(model_name),
            "seed": int(seed),
            "k_grid": [int(x) for x in k_grid],
            "sets_per_k": int(sets_per_k),
            "anchor_frac": float(anchor_frac),
            "missing_rate_max": float(missing_rate_max),
            "prefer_non_diag": bool(prefer_non_diag),
            "n_rows_evaluated": int(len(rows)),
        },
        "ots": {
            "n_min": int(ots.n_min),
            "cap_global": int(ots.cap_global),
            "mad_k": float(ots.mad_k),
            "floor_q": float(ots.floor_q),
            "fixed_auc_std": float(ots.fixed_auc_std),
            "fixed_spec_floor": float(ots.fixed_spec_floor),
            "fixed_sens_floor": float(ots.fixed_sens_floor),
            "fixed_prec_floor": float(ots.fixed_prec_floor),
        },
        "sources": {
            "auc_std": {"source": str(q_auc.source), "n_used": int(q_auc.n_used)},
            "spec_floor": {"source": str(q_spec.source), "n_used": int(q_spec.n_used)},
            "sens_floor": {"source": str(q_sens.source), "n_used": int(q_sens.n_used)},
            "prec_floor": {"source": str(q_prec.source), "n_used": int(q_prec.n_used)},
        },
    }
    warmup_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[WARMUP] OTS snapshot:")
    print(f"  auc_std_thr={q_auc.threshold:.4f} (src={q_auc.source}, n={q_auc.n_used})")
    print(f"  spec_floor ={q_spec.threshold:.4f} (src={q_spec.source}, n={q_spec.n_used})")
    print(f"  sens_floor ={q_sens.threshold:.4f} (src={q_sens.source}, n={q_sens.n_used})")
    print(f"  prec_floor ={q_prec.threshold:.4f} (src={q_prec.source}, n={q_prec.n_used})")

    return warmup_json


def load_warmup_ots(warmup_json: Path) -> OnlineThresholdState:
    payload = json.loads(warmup_json.read_text(encoding="utf-8"))
    ots_cfg = payload.get("ots", {})
    return OnlineThresholdState(
        n_min=int(ots_cfg["n_min"]),
        cap_global=int(ots_cfg["cap_global"]),
        mad_k=float(ots_cfg["mad_k"]),
        floor_q=float(ots_cfg["floor_q"]),
        fixed_auc_std=float(ots_cfg["fixed_auc_std"]),
        fixed_spec_floor=float(ots_cfg["fixed_spec_floor"]),
        fixed_sens_floor=float(ots_cfg["fixed_sens_floor"]),
        fixed_prec_floor=float(ots_cfg["fixed_prec_floor"]),
    )


def warmup_params_match(
    warmup_json: Path,
    *,
    model_name: str,
    seed: int,
    k_grid: Sequence[int],
    sets_per_k: int,
    anchor_frac: float,
    missing_rate_max: float,
    prefer_non_diag: bool,
    n_min: int,
    cap_global: int,
    mad_k: float,
    floor_q: float,
) -> bool:
    payload = json.loads(warmup_json.read_text(encoding="utf-8"))
    warm = payload.get("warmup", {})
    ots_cfg = payload.get("ots", {})
    return (
        str(warm.get("model_name", "")) == str(model_name)
        and int(warm.get("seed", -1)) == int(seed)
        and [int(x) for x in warm.get("k_grid", [])] == [int(x) for x in k_grid]
        and int(warm.get("sets_per_k", -1)) == int(sets_per_k)
        and float(warm.get("anchor_frac", -1.0)) == float(anchor_frac)
        and float(warm.get("missing_rate_max", -1.0)) == float(missing_rate_max)
        and bool(warm.get("prefer_non_diag", False)) == bool(prefer_non_diag)
        and int(ots_cfg.get("n_min", -1)) == int(n_min)
        and int(ots_cfg.get("cap_global", -1)) == int(cap_global)
        and float(ots_cfg.get("mad_k", -1.0)) == float(mad_k)
        and float(ots_cfg.get("floor_q", -1.0)) == float(floor_q)
    )


def build_domain_map(
    *,
    vars: List[str],
    var_info_map: Dict[str, Dict[str, Any]],
) -> Dict[str, str]:
    out = []
    for v in vars:
        info = var_info_map.get(v, {})
        dom = str(info.get("domain", "NA"))
        out.append(dom)
    return dict(zip(vars, out))


def build_feature_glossary(
    *,
    vars_needed: List[str],
    var_info_map: Dict[str, Dict[str, Any]],
    max_items: int = 150,
    max_meaning_chars: int = 100,
    max_notes_chars: int = 150,
) -> List[Dict[str, Any]]:
    """
    Produce a compact glossary list for ONLY the variables that appear in the prompt.

    Design:
    - Keep prompt payload small.
    - Include domain + meaning + missing_rate (highly relevant for your controller).
    """
    uniq = []
    seen = set()
    for v in vars_needed:
        if not isinstance(v, str) or not v:
            continue
        if v in seen:
            continue
        seen.add(v)
        uniq.append(v)
        if len(uniq) >= int(max_items):
            break

    out = []
    for v in uniq:
        info = var_info_map.get(v, {})
        meaning = str(info.get("meaning", ""))[: int(max_meaning_chars)]
        # notes = str(info.get("notes", ""))[: int(max_notes_chars)]
        out.append(
            {
                "feature": v,
                "domain": str(info.get("domain", "NA")),
                "meaning": meaning,
                "missing_rate": utils.sig4(info.get("missing_rate", 0.0)),
                # "notes": notes,
            }
        )
    return out


def craft_controller_prompt_ising(
    *,
    current_set: List[str],
    structural_snapshot: Dict[str, Any],
    phase: Dict[str, Any],
    feature_glossary: List[Dict[str, Any]],
    # metrics_brief: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    LLM outputs a probing agenda compatible with ControlAgenda, but with fewer keys:
      {
        "mode": "edge_accel|balance|explore|NA",
        "anchor_nodes": [...],
        "frontier_edges": [[u,v], ...],
        "domain_priority": [{"domain_a":..., "domain_b":..., "weight":...}, ...],
        "probe_alpha_node": number,
        "probe_topk_edges": integer,
        "force_include_anchor_prob": number,
        "rationale": string
      }

    Fixed by executor (NOT output by LLM):
      - edge_from = "node_targets"
      - max_* caps (A_MAX/E_MAX/D_MAX); executor clips safely
    """

    system_text = Path("utils_prompt.md").read_text(encoding="utf-8")

    payload = {
        "feature_glossary": feature_glossary,
        "current_set": current_set,
        "structural_snapshot": structural_snapshot,
        "phase": phase,
    }
    glossary_text = json.dumps(feature_glossary, ensure_ascii=True, separators=(",", ":"))
    payload_text = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    struct_text = json.dumps(structural_snapshot, ensure_ascii=True, separators=(",", ":"))
    phase_text = json.dumps(phase, ensure_ascii=True, separators=(",", ":"))
    print(f"glossary_text length={len(glossary_text)}")
    print(f"payload_text length={len(payload_text)}")
    print(f"struct_text length={len(struct_text)}")
    print(f"phase_text length={len(phase_text)}\n")

    return [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": [{"type": "text", "text": payload_text}]},
    ]


@dataclass
class RunState:
    run_idx: int
    iter_idx: int = 0
    current_set: List[str] = field(default_factory=list)
    prev_set: Optional[List[str]] = None
    prev_metrics: Optional[Dict[str, Any]] = None
    prev_reward: float = float("nan")


def build_agenda_row(agenda: Any) -> Dict[str, Any]:
    def _encode_agenda_value(v: Any) -> Any:
        if v is None:
            return "NA"
        if isinstance(v, (dict, list, tuple, set)):
            return json.dumps(v, ensure_ascii=True)
        return v

    if is_dataclass(agenda):
        agenda_items = asdict(agenda).items()
    elif isinstance(agenda, dict):
        agenda_items = agenda.items()
    else:
        agenda_items = []

    return {f"{k}": _encode_agenda_value(v) for k, v in agenda_items}


# ================================================================
# main loop
# ================================================================
def run_parallel_interleaving(
    ds: DatasetCtx,
    exp_spec: Optional[ExperimentSpec] = None,
):
    """
    Parallel interleaving exploration with 2-layer KG.

    Core loop:
      - Evaluate current_set with paired probes.
      - Update KG ONLINE using probed evidences.
      - Build evidence-driven candidate pool.
      - Ask LLM controller -> agenda -> next_set.
      - Update state: prev_* <= current, current <= next, iter++.
    """
    run_spec = exp_spec.run
    llm_model = run_spec.llm_model
    llm_time_total = 0.0
    out_root = Path(run_spec.out_root)
    ablation_cfg = {
        "edge_from": run_spec.edge_from,
        "use_llm_probes": run_spec.use_llm_probes,
        "use_llm_gibbs": run_spec.use_llm_gibbs,
    }
    RUN_NUM = int(run_spec.run_num)
    RUN_ITERS = int(run_spec.iters)
    K_MIN = int(run_spec.k_min)
    K_MAX = int(run_spec.k_max)
    P_NODE = int(run_spec.p_node)
    P_EDGE = int(run_spec.p_edge)
    EVAL_SEED = int(run_spec.global_seed)
    CV_MODEL = str(run_spec.cv_model)
    WARMUP_SEED = EVAL_SEED + int(run_spec.warmup_seed_offset)

    # 全局随机数生成器
    global_rng = utils.set_global_seed(EVAL_SEED)
    out_root.mkdir(parents=True, exist_ok=True)

    out_root_kg_snapshot = out_root / "kg_snapshot"
    out_root_kg_snapshot.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # Global outputs
    # ---------------------------
    # log文件在运行过程中不断更新，便于追踪运行进度
    # 最后会一次性覆盖存储
    global_iter_csv = out_root / "iteration_log.csv"
    global_control_csv = out_root / "control_log.csv"

    global_iter_csv_temp = out_root / "iteration_log_temp.csv"
    global_control_csv_temp = out_root / "control_log_temp.csv"

    global_fold_csv = out_root / "per_fold_metrics.csv"
    global_fold_std_csv = out_root / "per_fold_metrics_std.csv"

    for p in [global_iter_csv_temp, global_fold_csv, global_fold_std_csv, global_control_csv_temp]:
        if p.exists():
            p.unlink()  # delete previous runs

    global_iter_rows: List[Dict[str, Any]] = []
    global_fold_rows: List[pd.DataFrame] = []
    global_fold_std_rows: List[pd.DataFrame] = []
    global_control_rows: List[Dict[str, Any]] = []

    # ---------------------------
    # Load data, moved outside loop
    # ---------------------------
    # ds, stats = utils_dataset.build_dataset_ctx(
    #     DATA_FEATHER=DATA_FEATHER,
    #     META_CSV=META_CSV,
    #     CVD_EVENT_COLS=CVD_EVENT_COLS,
    #     allow_maybe=True,
    #     mode=run_plan.get("mode"),  # 20260225: 增加mode参数，区分不同模式下的训练集划分
    # )

    # ---------------------------
    # Init global knowledge graph
    # ---------------------------
    global_kg = KnowledgeGraph(all_features=ds.all_features, rng=global_rng)

    cache_path = Path("eval_cache")
    cache_path.mkdir(exist_ok=True)

    eval_cache_file = cache_path / f"{out_root.name}.sqlite"
    sig_file = cache_path / f"{out_root.name}_FEATURE_UNIVERSE_SHA256.txt"

    eval_cache, sig = utils_cache.init_eval_cache_with_universe_guard(
        kg=global_kg,
        eval_cache_path=eval_cache_file,
        universe_sig_path=sig_file,
    )

    # ---------------------------
    # Build run states
    # ---------------------------
    # 1) init run states
    states: List[RunState] = []
    for idx in range(RUN_NUM):
        st = RunState(run_idx=idx)
        states.append(st)

    traj_ids = [st.run_idx for st in states]

    # 2) init ES state (global shared)
    # target_k = int(np.median([len(s.current_set) for s in states]))

    # 3) init control plane (default if LLM not called yet)
    reward_w=utils_control.RewardWeights()
    reward_w.prevalence = ds.prevalence

    # 4) generation buffers
    gen_sets: Dict[int, List[str]] = {}
    gen_rewards: Dict[int, float] = {}
    gen_reward_dbg: Dict[int, Dict[str, Any]] = {}
    gen_metrics: Dict[int, Dict[str, Any]] = {}
    # gen_fold_rows: List[pd.DataFrame] = []

    # ---------------------------
    # Build round-robin schedule
    # ---------------------------
    rr = []
    active = {st.run_idx: RUN_ITERS for st in states}
    while any(v > 0 for v in active.values()):
        for st in states:
            if active[st.run_idx] > 0:
                rr.append(st.run_idx)
                active[st.run_idx] -= 1

    print(f"[INFO] runs={len(states)} total_global_steps={len(rr)} (round-robin interleaving)\n")

    warmup_json = out_root / "warmup_params.json"
    if warmup_json.exists() and warmup_params_match(
        warmup_json,
        model_name=CV_MODEL,
        seed=WARMUP_SEED,
        k_grid=run_spec.warmup_k_grid,
        sets_per_k=run_spec.warmup_sets_per_k,
        anchor_frac=run_spec.warmup_anchor_frac,
        missing_rate_max=run_spec.warmup_missing_rate_max,
        prefer_non_diag=run_spec.warmup_prefer_non_diag,
        n_min=run_spec.warmup_n_min,
        cap_global=run_spec.warmup_cap_global,
        mad_k=run_spec.warmup_mad_k,
        floor_q=run_spec.warmup_floor_q,
    ):
        ots = load_warmup_ots(warmup_json)
        print(f"[INFO] loaded warmup params -> {warmup_json}")
    else:
        if warmup_json.exists():
            print(f"[INFO] warmup params mismatch; rebuilding -> {warmup_json}")
        print("[INFO] threshold calibration (warmup)...")
        ots = OnlineThresholdState(
            n_min=run_spec.warmup_n_min,
            cap_global=run_spec.warmup_cap_global,
            mad_k=run_spec.warmup_mad_k,
            floor_q=run_spec.warmup_floor_q,
        )
        warmup_json = run_warmup_calibration(
            df_all=ds.df_all,
            y_all=ds.y_all,
            pool_schema=ds.pool_schema,
            pool_meta=ds.pool_meta,
            all_features=ds.all_features,
            ots=ots,
            out_root=out_root,
            model_name=CV_MODEL,
            k_grid=run_spec.warmup_k_grid,
            sets_per_k=run_spec.warmup_sets_per_k,
            anchor_frac=run_spec.warmup_anchor_frac,
            seed=WARMUP_SEED,
            missing_rate_max=run_spec.warmup_missing_rate_max,
            prefer_non_diag=run_spec.warmup_prefer_non_diag,
        )
        print(f"[INFO] threshold calibration done -> {warmup_json}")

    # ---------------------------
    # Interleaving loop
    # ---------------------------
    generation_idx = 0
    for gstep, ridx in enumerate(rr):
        st = next(s for s in states if s.run_idx == ridx)

        if st.iter_idx >= RUN_ITERS:
            continue  # this run finished

        # --- (A) Sample next_set from Ising-ES ---
        # 下一轮评估/更新KG的 feature set”（即进入 evaluate_auc_and_more_cv 的集合），
        # 应该以 global_kg.sample_set() 为主，因为它直接来自 REINFORCE 更新过的 p_{\theta,\phi}，
        # 能保证“reward → θ/φ → 采样分布 → 下一轮集合”这条闭环是自洽的。
        if st.iter_idx == 0:  # 第一步，feature set随机初始化;
                k = global_rng.randint(K_MIN, K_MAX)
                sampled_set = global_rng.sample(ds.all_features, k)
        else: # 后续，feature set通过Gibbs采样
            x0 = set_to_bitvec(st.current_set, global_kg.feat_index)
            sampled_set = global_kg.sample_set(x0=x0, ablation_cfg=ablation_cfg)

        st.prev_set = list(st.current_set)
        st.current_set = list(sampled_set)

        flip_rate = 0.0
        if st.prev_set:
            x0 = set_to_bitvec(st.prev_set, global_kg.feat_index)
            x1 = set_to_bitvec(st.current_set, global_kg.feat_index)
            flip_rate = np.mean(np.abs(x0 - x1))

        # --- (B) Evaluate base_set (current_set) ---
        base_set = list(st.current_set)
        base_metrics, base_score = utils_control.eval_set_to_score(
            base_set,
            kg=global_kg,
            eval_cache=eval_cache,
            ds=ds,
            model_name=CV_MODEL,
            seed=EVAL_SEED,
        )

        # Shared paired probes (node first, then edge using reuse) ---
        # P_NODE: number of nodes to probe
        # P_EDGE: number of edges to probe，边数是P_NODE*(P_NODE-1)/2
        node_probes, edge_probes, dbg_probes = utils_control.build_shared_paired_probes(
            kg=global_kg,
            eval_cache=eval_cache,
            ds=ds,
            base_set=base_set,
            base_score=base_score,
            P_NODE=P_NODE,
            P_EDGE=P_EDGE,
            reward_w=reward_w,
            # edge_from="base_set",
            model_name=CV_MODEL,
            eval_seed=EVAL_SEED,
            ablation_cfg=ablation_cfg,
        )

        # --- (C) Update KG online with paired probes ---
        # Replace the old global_kg.update_online_sample(...)
        dbg = global_kg.update_online_paired(
            selected_set=base_set,
            base_score=float(base_score),
            node_probes=node_probes,
            edge_probes=edge_probes,
        )
        print(f"[Evidence Update] run={st.run_idx} iter={st.iter_idx} n_features={len(st.current_set)} dbg={dbg}\n")

        # --- (D) Compute reward (perf + evidence gain + structure gain) ---
        R, R_dbg = utils_control.compute_reward(
            metrics_summary=base_metrics,
            selected_set=st.current_set,
            kg=global_kg,
            weights=reward_w,
            ots=ots,
            pool_meta=ds.pool_meta,
            node_probes=node_probes,
            edge_probes=edge_probes,
        )
        print(f"[RUN] run={st.run_idx} iter={st.iter_idx} n_features={len(st.current_set)} n_node_probes={len(node_probes)} n_edge_probes={len(edge_probes)} R={R:.4f}\n")
        st.prev_metrics = dict(base_metrics)
        st.prev_reward = float(R)

        # --- (E) Put into generation buffer ---
        gen_sets[st.run_idx] = list(st.current_set)
        gen_rewards[st.run_idx] = float(R)
        gen_reward_dbg[st.run_idx] = dict(R_dbg)
        gen_metrics[st.run_idx] = dict(base_metrics)

        # logging row (iteration_log)
        iter_row = {
            "gstep": int(gstep),
            "traj_id": int(st.run_idx),
            "iteration": int(st.iter_idx),
            "Gibbs_T": int(global_kg.update_cfg.gibbs_T),
            "temperature": global_kg.update_cfg.temperature,
            "flip_rate": float(flip_rate),
            "bias0": float(global_kg.update_cfg.bias0),
            "n_edge_frontier_size": int(len(global_kg.edge_frontier)),
            "current_set_json": json.dumps(st.current_set, ensure_ascii=True),
            "node_probes": json.dumps(node_probes, ensure_ascii=True),
            "edge_probes": json.dumps(edge_probes, ensure_ascii=True),
            # reward中没使用的metrics
            "acc_mean": utils.as_float(base_metrics, "acc_mean"),
            "acc_std": utils.as_float(base_metrics, "acc_std"),
            "bal_acc_mean": utils.as_float(base_metrics, "bal_acc_mean"),
            "bal_acc_std": utils.as_float(base_metrics, "bal_acc_std"),
            "f1_mean": utils.as_float(base_metrics, "f1_mean"),
            "f1_std": utils.as_float(base_metrics, "f1_std"),
            "recall_mean": utils.as_float(base_metrics, "recall_mean"),
            "recall_std": utils.as_float(base_metrics, "recall_std"),
        }
        # reward related
        iter_row.update(R_dbg)
        # kg related
        pstats = global_kg.get_param_stats()
        iter_row.update(pstats)

        global_iter_rows.append(iter_row)
        if not global_iter_csv_temp.exists():
            pd.DataFrame([iter_row]).to_csv(global_iter_csv_temp, index=False, header=True)
        else:
            pd.DataFrame([iter_row]).to_csv(global_iter_csv_temp, index=False, mode="a", header=False)

        st.iter_idx += 1

        # ======================generations==========================================
        # --- (F) If we have one sample from each run => a generation is complete ---
        if len(gen_sets) == RUN_NUM:
            # --- Update graph update schedule ---
            START_FRAC = 0.40   # e.g. 200/500
            TRANS_FRAC = 0.20   # e.g. 100/500
            MIN_TRANS  = 20     # transition at least this many iters

            transition_start_iter = int(round(START_FRAC * RUN_ITERS))
            transition_len = int(round(TRANS_FRAC * RUN_ITERS))
            transition_len = max(MIN_TRANS, transition_len)

            # clamp to valid range
            transition_start_iter = max(0, min(transition_start_iter, RUN_ITERS - 1))
            transition_end_iter = min(RUN_ITERS, transition_start_iter + transition_len)

            # if end == start (very short runs), force a minimal meaningful transition window
            if transition_end_iter <= transition_start_iter:
                transition_end_iter = min(RUN_ITERS, transition_start_iter + 1)

            # --- Update graph update schedule (A -> soft B -> hard B) ---
            if generation_idx < transition_start_iter:
                # (1) hard A
                global_kg.update_cfg.change_stage(stage="A")

            elif generation_idx < transition_end_iter:
                # (2) soft transition toward B
                global_kg.update_cfg.change_stage(
                    stage="B",
                    iteration=int(generation_idx),
                    transition_len=int(transition_end_iter - transition_start_iter),
                    transition_start_iter=int(transition_start_iter),
                    use_smoothstep=True,
                )
            else:
                # (3) hard B (no iteration args; your early-return path will hard-assign)
                global_kg.update_cfg.change_stage(stage="B")


            batch_sets = [gen_sets[r] for r in sorted(gen_sets.keys())]
            batch_rewards = [gen_rewards[r] for r in sorted(gen_rewards.keys())]

            # 严格正确的 REINFORCE 逻辑：应该先用“生成样本时的分布”来更新 θ/φ，然后再更新 edge_set 给下一代用
            # Update graph distribution (theta, phi)
            extra_sets = global_kg.sample_multiple_sets(M=32)
            upd_dbg = global_kg.reinforce_update_theta_phi(
                batch_sets=batch_sets + extra_sets,
                batch_rewards=batch_rewards,
            )

            # 更新bias0
            ks = [len(s) for s in gen_sets.values() if isinstance(s, list)]
            k_mean = sum(ks) / len(ks) if ks else 0.0
            beta = 0.9
            global_kg.kmean_ema = beta*global_kg.kmean_ema + (1-beta)*k_mean
            bias_dbg = global_kg.update_bias0(k_mean=global_kg.kmean_ema)

            # 每个generation保存一次kg_snapshot
            kg_json = f"{out_root_kg_snapshot}/kg_snapshot_{generation_idx:03d}.json"
            global_kg.save_snapshot(str(kg_json))

            # ======================使用LLM进行调控===================================
            if llm_model is not None:
                # 只有full edge set size 非常大的时候，才采用稀疏化的edge set
                # 目前使用full sdge set，不需要调用propose_edge_candidates()
                # A) 你当前（F≈75，E=2775）——不需要稀疏化
                #     •	Gibbs 每次扫一遍变量只需要看该变量的邻居。
                #     •	如果 edge_set full，那么每个变量的邻居数约 F-1。
                #     •	一次 sweep 的复杂度近似：O(F*(F-1)) = O(F^2)。
                #     •	F=75 → 75²=5625 级别，哪怕 T=10 也就 ~5e4 次邻居累加，非常小。
                # 因此：
                #     •	全量 edge_set + 全量 phi + Gibbs 在你现在规模上完全可接受。
                #     •	同时你还能避免“edge_set 候选更新顺序/branch/配额”这类工程噪声对实验的干扰。
                # B）什么时候才需要重新做稀疏 edge_set？
                # 当满足任一情况时再上稀疏化（或者局部邻居化）更合理：
                #     1.	F 上千：full edge 的内存和更新都开始不舒服
                #     •	E ~ F(F-1)/2，F=2000 → E≈2,000,000（phi 就很大了）
                #     2.	你想把更新 phi 的开销压下来
                #     •	你现在的 reinforce_update_theta_phi 里 phi 更新如果按 full edge 做，会是 O(N*E)。
                #     •	你可能现在还只对 edge_set 更新 phi；若 edge_set full，就等价于更新所有边（开销会随 E 增长）。
                #     3.	你想让“图结构学习”更像 structure learning
                #     •	让 edge_set 本身成为被学习/被选择的对象（结构可解释、稀疏、易看）。
                # ================
                # Update edge_set every generation (or every K generations)
                # edge_set, branch = global_kg.propose_edge_candidates(
                #     M=edge_candidate_M,
                #     anchors=anchors,
                #     prefer_uncertain=control.edge_pref_uncertain,
                #     prefer_synergy=control.edge_pref_synergy,
                #     min_support=8,         # 仍然保留这个量，但现在是“软尺度”
                #     rng=rng,               # 打散补边顺序
                # )
                # global_kg.set_edge_set(edge_set)

                # --- (G) Call LLM once per generation to adjust policy ---
                # build small kg_snapshot (you can make this richer)
                # kg_snapshot = {
                #     "top_probed_nodes": [
                #         {"var": f, "probed": global_kg.node_stat[f].n_pair()}
                #         for f in sorted(ds.all_features, key=lambda x: global_kg.node_stat[x].n_pair(), reverse=True)[:12]
                #     ],
                #     "update_debug": upd_dbg,
                #     "reward_mean": float(np.mean(batch_rewards)),
                # }

                # failure_type: 这里可以用 batch 统计（比如 auc_std 高/auc drop）
                # 先简单：若 update_debug 表示 A 波动大，则 unstable
                # failure_type = "ok"
                # if float(upd_dbg.get("A_max", 0.0)) - float(upd_dbg.get("A_min", 0.0)) > 3.0:
                #     failure_type = "unstable"

                metrics_brief = {
                    "auc_mean_batch": utils.sig4(float(np.mean([utils.as_float(m, "auc_mean") for m in gen_metrics.values()]))),
                    "auc_std_batch": utils.sig4(float(np.mean([utils.as_float(m, "auc_std") for m in gen_metrics.values()]))),
                    "auprc_mean_batch": utils.sig4(float(np.mean([utils.as_float(m, "auprc_mean") for m in gen_metrics.values()]))),
                    "auprc_std_batch": utils.sig4(float(np.mean([utils.as_float(m, "auprc_std") for m in gen_metrics.values()]))),
                }

                # 5) Evidence-driven candidate subset
                # 确定下一轮实验（采样/对照）应该优先让哪些 feature 的证据更快变得“可确认/可否认”（从而提升 KG 的可检验性与稳定性）
                # 是一个 Expected Evidence Gain（预期证据增益） 的问题
                # 对一个 feature f，我们希望选它，是因为它：
                # 1) 缺证据（counterfactual 不平衡 / 支持度低 / incident edges 的 2×2 cell 不齐）
                # 2) 值得补（当前已经有一点信号：node 的 |t| 或一些边的 |t| 不为 0，只是证据不足；否则补它只是浪费预算）
                #
                # evidence_needed_feature_set, dbg = evidence_needed_features(
                #     kg=global_kg,
                #     top_k=20,   # 需要parameter tuning
                # )
                #
                # 寻找evidence needed到过程已经合并到build_llm_candidate_set中
                #
                # 给 LLM 的候选池/候选集合”（prompt 里让 LLM 做控制/修复/提出假设），
                # 需要显式注入“补证据/不确定/anchor”等语义信号——这是纯 Gibbs 采样不擅长的
                # KG 负责“概率正确”，LLM 候选负责“信息覆盖与对话控制”
                # 用 KG 采样作为骨架，再用 evidence_needed_feature_set 做“约束/微扰/扩展”

                # candidate_pool, cand_dbg = utils_control.build_llm_candidate_set(
                #     kg=global_kg,
                #     weights=reward_w,
                #     current_set=gen_set,
                #     candidate_subset_size=control.candidate_subset_size,
                # )

                # glossary for merged current_set across runs
                gen_set = list({v for s in gen_sets.values() for v in s})
                domain_map = build_domain_map(vars=gen_set, var_info_map=ds.var_info_map)
                # vars_needed = list(set(candidate_pool) | set(gen_set))
                feature_glossary = build_feature_glossary(vars_needed=ds.all_features, var_info_map=ds.var_info_map)

                # gen_summary = utils_control.build_batch_summary(
                #     kg=global_kg,
                #     domain_map=domain_map,
                #     all_features=gen_set,
                # )
                struct_snapshot = utils_control.build_structural_snapshot(
                    kg=global_kg,
                    domain_map=domain_map,
                    all_features=ds.all_features,
                    top_nodes=10, top_edges=16, frontier_nodes=12, frontier_edges=20, domain_pair_gaps=10,
                )

                phase = utils_control.build_phase_scalars_from_logs(
                    iters_log=global_iter_rows,
                )

                gloss_sub = utils_control.build_feature_glossary_subset(
                    feature_glossary=feature_glossary,   # 已删除notes版本
                    current_set=gen_set,
                    structural_snapshot=struct_snapshot,
                    max_items=48,
                )

                prompt = craft_controller_prompt_ising(
                    current_set=gen_set,
                    structural_snapshot=struct_snapshot,
                    phase=phase,
                    feature_glossary=gloss_sub,
                )

                t0 = time.perf_counter()
                agenda = utils_models.ask_llm(llm_model, prompt) or {}
                if not agenda:
                    print(f"[WARN] LLM returned empty output for run {st.run_idx} (iter {st.iter_idx})")
                    # print("Ask LLM again ...")
                    # agenda = utils_models.ask_llm(llm_model, prompt) or {}
                llm_time_cost = time.perf_counter() - t0
                llm_time_total += llm_time_cost
                print(f"[INFO] LLM time cost: {format_hms(llm_time_cost)}\n")

                # santize agenda
                agenda = utils_control.sanitize_controller_output(
                    controller=agenda,
                    all_features=gen_set,
                    domain_map=domain_map,
                )

                print(f"[INFO] Santized LLM output -> {agenda}\n")

                # 更新kg的agenda， 会通过pick paried probes改变evidence更新策略
                if agenda:  # LLM会返回NA，agenda为空，则不更新
                    global_kg.control_agenda = agenda

                batch_row = {
                    "generation": generation_idx,
                    "Gibbs_T": global_kg.update_cfg.gibbs_T,
                    "lr_theta": global_kg.update_cfg.lr_theta,
                    "lr_phi": global_kg.update_cfg.lr_phi,
                    "n_edge_frontier_size": int(len(global_kg.edge_frontier)),
                    "reward_mean": float(np.mean(batch_rewards)),
                    "llm_time_cost": llm_time_cost,
                }
                agenda_row = build_agenda_row(agenda)
                batch_row.update(agenda_row)

                if not global_control_csv_temp.exists():
                    pd.DataFrame([batch_row]).to_csv(global_control_csv_temp, index=False, header=True)
                else:
                    pd.DataFrame([batch_row]).to_csv(global_control_csv_temp, index=False, mode="a", header=False)

                global_control_rows.append(batch_row)

                print(f"[GEN] {batch_row}\n\n")

            # 清空 generation buffers
            gen_sets.clear()
            gen_rewards.clear()
            gen_reward_dbg.clear()
            gen_metrics.clear()

            generation_idx += 1

    # ---------------------------
    # Write GLOBAL outputs
    # ---------------------------
    if global_iter_rows:
        pd.DataFrame(global_iter_rows).to_csv(global_iter_csv, index=False)

    if global_control_rows:
        pd.DataFrame(global_control_rows).to_csv(global_control_csv, index=False)

    if global_fold_rows:
        pd.concat(global_fold_rows, axis=0, ignore_index=True).to_csv(global_fold_csv, index=False)
    if global_fold_std_rows:
        pd.concat(global_fold_std_rows, axis=0, ignore_index=True).to_csv(global_fold_std_csv, index=False)

    # ---------------------------
    # Save KG snapshot
    # ---------------------------
    # 保存最后一次kg_snapshot到out_root
    kg_json = out_root / "kg_snapshot.json"
    global_kg.save_snapshot(str(kg_json))

    # close cacher
    eval_cache.close()

    print(f"[INFO] global iteration_log -> {global_iter_csv}")
    print(f"[INFO] global per_fold_metrics -> {global_fold_csv}")
    print(f"[INFO] global per_fold_metrics_std -> {global_fold_std_csv}")
    print(f"[INFO] global control_log -> {global_control_csv}")
    print(f"[INFO] KG snapshot -> {kg_json}")
    return llm_time_total


def format_hms(seconds: float) -> str:
    """
    Convert seconds to Hh Mm Ss string.
    """
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}h {m}m {s}s"


def log_runtime(
    exp_spec: Optional[ExperimentSpec] = None,
    llm_time=None,
    total_time=None,
    out_root=None,
    llm_model=None,
):
    """
    Append runtime information to a TSV log file.

    Format:
    TIMESTAMP (YYYY-MM-DD:HH-MM-SS)   parameters   llm_time   total_time
    """
    run_spec = exp_spec.run
    llm_model = run_spec.llm_model
    mode = exp_spec.task.mode
    run_num = run_spec.run_num
    iters = run_spec.iters
    k_min = run_spec.k_min
    k_max = run_spec.k_max
    ablation_cfg = {
        "edge_from": run_spec.edge_from,
        "use_llm_probes": run_spec.use_llm_probes,
        "use_llm_gibbs": run_spec.use_llm_gibbs,
    }

    # Format timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_path = out_root / "runtime_log.tsv"
    # If file does not exist → create with header
    new_file = not os.path.exists(log_path)

    with open(log_path, "a") as f:
        if new_file:
            f.write("timestamp\tmode\tllm_model\trun_num\titers\tk_min\tk_max\tllm_time\ttotal_time\tedge_from\tuse_llm_probes\tuse_llm_gibbs\n")

        f.write(
            f"{timestamp}"
            f"\t{mode}"
            f"\t{llm_model}\t{run_num}\t{iters}"
            f"\t{k_min}\t{k_max}"
            f"\t{format_hms(llm_time)}"
            f"\t{format_hms(total_time)}"
            f"\t{ablation_cfg.get('edge_from')}"
            f"\t{ablation_cfg.get('use_llm_probes')}"
            f"\t{ablation_cfg.get('use_llm_gibbs')}"
            f"\n"
        )

    print(f"[INFO] Runtime logged to {log_path}")


# def main():
#     """
#     Entry point.

#     Key change for Ising / sparse pairwise ES:
#     - We make the population size explicit via `run_num` (N).
#     - Each run in `run_plans` corresponds to one sampled individual per ES generation.
#       (Program-side: you can interpret one "global generation" as one round-robin pass
#        that evaluates each run once, then performs a single ES update using all N rewards.)

#     Notes on parameters kept:
#     - iters: total iterations per run (controls total evaluation budget).
#     """
#     TOTAL_START = time.perf_counter()
#     llm_time = 0.0

#     parser = argparse.ArgumentParser(description="Parallel Global KG runner")
#     parser.add_argument("--mode", default="Status", help="Status/Prediction")
#     parser.add_argument("--llm-model", default="none", help="LLM model name (use 'none' to disable)")
#     parser.add_argument("--run-num", type=int, default=5, help="Population size per generation")
#     parser.add_argument("--iters", type=int, default=2, help="Total iterations per run")
#     parser.add_argument("--k-min", type=int, default=10, help="Minimum feature set size")
#     parser.add_argument("--k-max", type=int, default=20, help="Maximum feature set size")
#     parser.add_argument("--p-node", type=int, default=10, help="Paired probe node count")
#     parser.add_argument("--p-edge", type=int, default=30, help="Paired probe edge count")
#     parser.add_argument(
#         "--edge-from",
#         type=str,
#         default="base_set",
#         choices=["base_set", "node_targets"],
#         help="Edge sampling source for probes",
#     )
#     parser.add_argument(
#         "--use-llm-probes",
#         type=str,
#         default="true",
#         help="Whether to use LLM-suggested probes (true/false)",
#     )
#     parser.add_argument(
#         "--use-llm-gibbs",
#         type=str,
#         default="true",
#         help="Whether to use LLM agenda for Gibbs debias (true/false)",
#     )
#     args = parser.parse_args()

#     mode = args.mode
#     if mode not in {"Status", "Prediction"}:
#         print(f"[ERROR] Invalid mode: {mode}")
#         exit(1)

#     llm_model = args.llm_model
#     if isinstance(llm_model, str) and llm_model.strip().lower() in {"none", "null", ""}:
#         llm_model = None

#     if not llm_model:
#         args.use_llm_probes = "false"
#         args.use_llm_gibbs = "false"

#     if llm_model:
#         if not utils_models.probe_server(llm_model):
#             print(f"[ERROR] LLM probe failed for {llm_model}; exiting...")
#             exit(1)

#     def _to_bool(x):
#         if isinstance(x, (bool, np.bool_)):
#             return bool(x)
#         if x is None:
#             return False
#         if isinstance(x, (int, float)):
#             return bool(int(x))
#         s = str(x).strip().lower()
#         if s in {"1", "true", "t", "yes", "y"}:
#             return True
#         if s in {"0", "false", "f", "no", "n", ""}:
#             return False
#         return bool(s)

#     # ===== set up output directory =====
#     if llm_model:
#         out_root = OUT_ROOT / f"{llm_model}_{mode}"
#     else:
#         out_root = OUT_ROOT / f"Baseline_{mode}"

#     out_root = Path(f"{out_root}_iters{int(args.iters)}")

#     edge_from = str(args.edge_from)
#     use_llm_probes = _to_bool(args.use_llm_probes)
#     use_llm_gibbs = _to_bool(args.use_llm_gibbs)

#     out_root = Path(f"{out_root}{'_NodeT' if edge_from == 'node_targets' else '_BaseS'}")
#     if llm_model:
#         out_root = Path(f"{out_root}{'_useP' if use_llm_probes else '_notP'}")
#         out_root = Path(f"{out_root}{'_useDB' if use_llm_gibbs else '_notDB'}")

#     run_plan = {
#         # mode： Status： status/outcome modeling； Prediction: 5-year risk-prediction
#         "mode": mode,

#         # population size per generation (number of parallel trajectories)
#         "run_num": int(args.run_num),

#         # total iterations for each run (evaluation budget per individual)
#         "iters": int(args.iters),

#         # k_min, k_max 不是为了寻找最优的特征数量， 而是为了更好的累积证据
#         # 所以取值不是考虑最优的特征数量问题
#         # 这两个参数只决定第一次随机特征采样的数量区间
#         # 后续的特征采样数量完全有Gibbs采样决定
#         # minimum feature set size
#         "k_min": int(args.k_min),
#         # maximum feature set size
#         "k_max": int(args.k_max),

#         # paired probe hyperparams
#         "p_node": int(args.p_node),
#         "p_edge": int(args.p_edge),

#         # ablation config
#         "ablation_cfg": {
#             "edge_from": edge_from,
#             "use_llm_probes": use_llm_probes,
#             "use_llm_gibbs": args.use_llm_gibbs,
#         },

#         # output directory
#         "out_root": out_root,
#     }

#     run_parallel_interleaving(
#         ds=ds,
#         llm_model=llm_model,
#         run_plan=run_plan,
#         verbose=True
#     )
#     print(f"[DONE] parallel runs finished -> {out_root}")

#     total_time = time.perf_counter() - TOTAL_START
#     # ===== call logger =====
#     log_runtime(
#         llm_model=llm_model,
#         run_plan=run_plan,
#         llm_time=llm_time,
#         total_time=total_time,
#         out_root=out_root,
#     )

# if __name__ == "__main__":
#     # Usage:
#     # python S04S05_parallel_globalKG.py --mode Status --llm-model none --iters 100 --edge-from base_set --k-min 10 --k-max 20 --p-node 10 --p-edge 20
#     # python S04S05_parallel_globalKG.py --mode Prediction --llm-model Qwen --iters 100 --edge-from base_set --use-llm-probes true --use-llm-gibbs true --k-min 10 --k-max 20 --p-node 10 --p-edge 20
#     main()

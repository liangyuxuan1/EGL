#!/usr/bin/env python3
"""
main.py

Run multiple experiments from a YAML config with:
- bases/base/override inheritance (deep merge)
- --list to list resolved experiment names
- --dry-run to print the fully materialized config for selected experiments

Expected YAML structure
-----------------------
bases:
  <base_name>:
    dataset: {...}
    task: {...}
    preprocess: {...}
    run: {...}

experiments:
  - name: <exp_name>
    base: <base_name>            # optional
    override: { ... }            # optional (deep-merged into base)
    # You can also put dataset/task/preprocess/run directly at top-level if no base.

Usage
-----
python main.py --config config.yaml --list
python main.py --config config.yaml --pattern copdgene_ --dry-run
python main.py --config config.yaml --only exp_name
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

# --------------------------------------------------------------------
# IMPORTS FROM PROJECT
# --------------------------------------------------------------------
# You should update these imports to match your repo structure.
from config_registry import EVENT_COLS_REGISTRY
from utils_dataset import (
    DatasetSpec,
    TaskSpec,
    PreprocessSpec,
    RunSpec,
    ExperimentSpec,
    build_dataset_ctx,
)
import utils_models
from main_loop import run_parallel_interleaving, log_runtime
# --------------------------------------------------------------------


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def _to_bool(x: Any) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if x is None:
        return False
    if isinstance(x, (int, float, np.integer, np.floating)):
        return bool(int(x))
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n", ""}:
        return False
    return bool(s)


def _norm_llm_model(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if s.lower() in {"none", "null", ""}:
        return None
    return s


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        obj = yaml.safe_load(f)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError("Top-level YAML must be a dict.")
    return obj


def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge dict b into dict a (b overrides a).
    - If both values are dicts -> recursive merge.
    - Otherwise -> overwrite.
    """
    out = dict(a or {})
    for k, v in (b or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve_base(
    cfg: Dict[str, Any],
    base_name: str,
    *,
    stack: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Resolve a base from cfg['bases'], supporting multi-level inheritance.

    A base may itself declare:
      base: <parent_base_name>
      override: {...}   # optional local override block on top of parent

    Resolution order for a base B:
      parent_base -> B.override -> B.inline_fields
    """
    bases = cfg.get("bases", {}) or {}
    if base_name not in bases:
        raise KeyError(f"Unknown base '{base_name}'. Available: {sorted(bases.keys())}")

    stack = list(stack or [])
    if base_name in stack:
        cycle = " -> ".join(stack + [base_name])
        raise ValueError(f"Detected cyclic base inheritance: {cycle}")

    base_obj = dict(bases[base_name] or {})
    parent_name = base_obj.get("base", None)
    override_block = base_obj.get("override", {}) or {}
    inline_obj = {k: v for k, v in base_obj.items() if k not in {"base", "override"}}

    if parent_name:
        parent_resolved = resolve_base(cfg, str(parent_name), stack=stack + [base_name])
        merged = deep_merge(parent_resolved, override_block)
        merged = deep_merge(merged, inline_obj)
    else:
        merged = deep_merge({}, override_block)
        merged = deep_merge(merged, inline_obj)

    return merged


def materialize_experiment(cfg: Dict[str, Any], exp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve an experiment definition into a fully materialized config:
      base -> override -> exp_inline

    Rules:
    - if exp has 'base', recursively resolve cfg['bases'][base], then deep-merge exp['override']
    - then deep-merge any inline keys from exp itself (dataset/task/preprocess/run)
      so users can override without using the 'override' block if desired
    """
    if "name" not in exp:
        raise ValueError("Each experiment must have a 'name'.")

    base_name = exp.get("base", None)
    override_block = exp.get("override", {}) or {}

    # Inline blocks: allow users to directly put dataset/task/preprocess/run at the same level
    # (useful for quick edits without nesting under override)
    exp_inline = {
        k: v for k, v in exp.items()
        if k not in {"base", "override"}  # keep name, plus inline dataset/task/preprocess/run
    }

    if base_name:
        base_obj = resolve_base(cfg, str(base_name))
        merged = deep_merge(base_obj, override_block)
        merged = deep_merge(merged, exp_inline)  # exp inline overrides base+override
    else:
        merged = dict(exp_inline)  # no base -> use inline directly

    # Ensure name stays
    merged["name"] = exp["name"]

    return merged


def select_experiments(
    exps: List[Dict[str, Any]],
    *,
    only: str = "",
    pattern: str = "",
) -> List[Dict[str, Any]]:
    out = []
    for exp in exps:
        name = str(exp.get("name", ""))
        if not name:
            continue
        if only and name != only:
            continue
        if pattern and pattern not in name:
            continue
        out.append(exp)
    return out


def safe_json(obj: Any) -> str:
    def _default(o):
        # Avoid dumping dataclass objects or Path objects if present
        try:
            if isinstance(o, Path):
                return str(o)
        except Exception:
            pass
        return str(o)

    return json.dumps(obj, indent=2, ensure_ascii=False, default=_default)


# --------------------------------------------------------------------
# Output directory naming
# --------------------------------------------------------------------
def build_out_root(
    *,
    base_out_root: Path,
    exp_name: str,
    llm_model: Optional[str],
    iters: int,
    edge_from: str,
    use_llm_probes: bool,
    use_llm_gibbs: bool,
) -> Path:
    out_root = base_out_root / f"{exp_name}"
    out_root = Path(f"{out_root}_iters{int(iters)}")
    out_root = Path(f"{out_root}{'_NodeT' if edge_from == 'node_targets' else '_BaseS'}")

    if llm_model:
        out_root = Path(f"{out_root}{'_useP' if use_llm_probes else '_notP'}")
        out_root = Path(f"{out_root}{'_useDB' if use_llm_gibbs else '_notDB'}")

    return out_root


# --------------------------------------------------------------------
# Convert config blocks -> Specs
# --------------------------------------------------------------------
def make_dataset_spec(d: Dict[str, Any]):
    return DatasetSpec(
        name=str(d["name"]),
        feather_path=Path(d["feather_path"]),
        id_col=str(d["id_col"]),
        visit_col=d.get("visit_col", None),
        dataset_col=d.get("dataset_col", None),
        dataset_value=d.get("dataset_value", None),
        colmap=d.get("colmap", None),
        require_unique_id_visit=bool(d.get("require_unique_id_visit", True)),
    )


def make_task_spec(d: Dict[str, Any]):
    """
    Build TaskSpec.

    Supports:
      - event_cols_ref: <string key>  (recommended)
      - event_cols: {col: cat/num, ...}
      - event_cols: [col1, col2, ...]  (treated as all 'cat')
    """
    # 1) Prefer reference-based event cols
    ref = d.get("event_cols_ref", None)
    if ref is not None:
        ref = str(ref)
        if ref not in EVENT_COLS_REGISTRY:
            raise KeyError(
                f"Unknown event_cols_ref: {ref}. "
                f"Available: {sorted(EVENT_COLS_REGISTRY.keys())}"
            )
        event_cols = dict(EVENT_COLS_REGISTRY[ref])  # copy
    else:
        # 2) Fallback to inline event_cols for quick experiments
        raw = d.get("event_cols", {})
        if isinstance(raw, list):
            event_cols = {str(c): "cat" for c in raw}
        elif isinstance(raw, dict):
            event_cols = {str(k): str(v) for k, v in raw.items()}
        else:
            raise ValueError("task.event_cols must be a list or a dict, or provide task.event_cols_ref")

    return TaskSpec(
        name=str(d["name"]),
        mode=str(d["mode"]),
        label_builder=str(d["label_builder"]),
        event_cols=event_cols,
        label_kwargs=dict(d.get("label_kwargs", {})),
    )


def make_preprocess_spec(d: Dict[str, Any]):
    return PreprocessSpec(
        meta_csv=Path(d["meta_csv"]),
        allow_maybe=bool(d.get("allow_maybe", True)),
        outlier_p_lo=float(d.get("outlier_p_lo", 0.01)),
        outlier_p_hi=float(d.get("outlier_p_hi", 0.99)),
        missing_rate_threshold=float(d.get("missing_rate_threshold", 0.7)),
    )


def make_run_spec(
    exp_name: str,
    run_cfg: Dict[str, Any],
):
    llm_model = _norm_llm_model(run_cfg.get("llm_model", None))
    global_seed = int(run_cfg.get("global_seed", 42))
    cv_model = str(run_cfg.get("cv_model", "Logistic"))
    warmup_seed_offset = int(run_cfg.get("warmup_seed_offset", 7))
    warmup_n_min = int(run_cfg.get("warmup_n_min", 30))
    warmup_cap_global = int(run_cfg.get("warmup_cap_global", 400))
    warmup_mad_k = float(run_cfg.get("warmup_mad_k", 2.0))
    warmup_floor_q = float(run_cfg.get("warmup_floor_q", 0.10))
    warmup_k_grid = tuple(int(x) for x in run_cfg.get("warmup_k_grid", [10, 15, 20, 25, 30]))
    warmup_sets_per_k = int(run_cfg.get("warmup_sets_per_k", 10))
    warmup_anchor_frac = float(run_cfg.get("warmup_anchor_frac", 0.30))
    warmup_missing_rate_max = float(run_cfg.get("warmup_missing_rate_max", 0.30))
    warmup_prefer_non_diag = _to_bool(run_cfg.get("warmup_prefer_non_diag", True))

    run_num = int(run_cfg.get("run_num", 5))
    iters = int(run_cfg.get("iters", 2))
    k_min = int(run_cfg.get("k_min", 10))
    k_max = int(run_cfg.get("k_max", 20))
    p_node = int(run_cfg.get("p_node", 10))
    p_edge = int(run_cfg.get("p_edge", 30))

    edge_from = str(run_cfg.get("edge_from", "base_set"))
    if edge_from not in {"base_set", "node_targets"}:
        raise ValueError(f"[{exp_name}] invalid edge_from: {edge_from}")

    use_llm_probes = _to_bool(run_cfg.get("use_llm_probes", True))
    use_llm_gibbs = _to_bool(run_cfg.get("use_llm_gibbs", True))

    if not llm_model:
        use_llm_probes = False
        use_llm_gibbs = False

    base_out_root = Path(run_cfg.get("out_root", "outputs"))
    out_root = build_out_root(
        base_out_root=base_out_root,
        exp_name=exp_name,
        llm_model=llm_model,
        iters=iters,
        edge_from=edge_from,
        use_llm_probes=use_llm_probes,
        use_llm_gibbs=use_llm_gibbs,
    )

    return RunSpec(
        llm_model=llm_model,
        global_seed=global_seed,
        cv_model=cv_model,
        warmup_seed_offset=warmup_seed_offset,
        warmup_n_min=warmup_n_min,
        warmup_cap_global=warmup_cap_global,
        warmup_mad_k=warmup_mad_k,
        warmup_floor_q=warmup_floor_q,
        warmup_k_grid=warmup_k_grid,
        warmup_sets_per_k=warmup_sets_per_k,
        warmup_anchor_frac=warmup_anchor_frac,
        warmup_missing_rate_max=warmup_missing_rate_max,
        warmup_prefer_non_diag=warmup_prefer_non_diag,
        run_num=run_num,
        iters=iters,
        k_min=k_min,
        k_max=k_max,
        p_node=p_node,
        p_edge=p_edge,
        edge_from=edge_from,
        use_llm_probes=use_llm_probes,
        use_llm_gibbs=use_llm_gibbs,
        out_root=out_root,
    )


def make_experiment_spec(exp_mat: Dict[str, Any]) -> ExperimentSpec:
    exp_name = str(exp_mat.get("name", "unnamed_exp"))
    dataset_spec = make_dataset_spec(exp_mat["dataset"])
    task_spec = make_task_spec(exp_mat["task"])
    preprocess_spec = make_preprocess_spec(exp_mat["preprocess"])
    run_spec = make_run_spec(
        exp_name,
        exp_mat.get("run", {}),
    )
    return ExperimentSpec(
        name=exp_name,
        dataset=dataset_spec,
        task=task_spec,
        preprocess=preprocess_spec,
        run=run_spec,
    )


# --------------------------------------------------------------------
# Main experiment loop
# --------------------------------------------------------------------
def run_one_experiment(exp_mat: Dict[str, Any]) -> None:
    """
    exp_mat is the fully materialized experiment dict:
    - contains dataset/task/preprocess/run and top-level defaults merged in
    """
    TOTAL_START = time.perf_counter()

    exp_name = str(exp_mat.get("name", "unnamed_exp"))

    # ----- validate required blocks -----
    for blk in ("dataset", "task", "preprocess", "run"):
        if blk not in exp_mat or not isinstance(exp_mat[blk], dict):
            raise ValueError(f"[{exp_name}] missing required block: '{blk}'")

    exp_spec = make_experiment_spec(exp_mat)

    run_spec = exp_spec.run
    llm_model = run_spec.llm_model

    # Optional: probe model server
    if llm_model:
        if not utils_models.probe_server(llm_model):
            raise RuntimeError(f"[{exp_name}] LLM probe failed for {llm_model}")

    # ----- output dir -----
    out_root = run_spec.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    # ----- build dataset ctx (shared by the run loop) -----
    datasetctx = build_dataset_ctx(
        dataset_spec=exp_spec.dataset,
        task_spec=exp_spec.task,
        preprocess_spec=exp_spec.preprocess,
    )

    # datasetctx.df_all.to_excel(out_root / "df_all.xlsx", index=False, engine="openpyxl")
    datasetctx.df_all.to_csv(out_root / "df_all.csv", index=False)
    datasetctx.df_all.to_feather(out_root / "df_all.feather")
    datasetctx.pool_meta.to_csv(out_root / "pool_meta.csv", index=False)
    ds_stats_log = Path(f"{out_root}/ds_stats.json")
    with ds_stats_log.open("w", encoding="utf-8") as f:
        json.dump(datasetctx.stats, f, ensure_ascii=False, indent=2)

    # yes + maybe feature的数量，75 in total
    print("-" * 78)
    print(f"[INFO] feature pool size={len(datasetctx.all_features)}")
    if "missing_rate" in datasetctx.pool_meta.columns:
        miss_df = (
            datasetctx.pool_meta[["var_name", "missing_rate"]]
            .dropna(subset=["var_name"])
            .assign(var_name=lambda d: d["var_name"].astype(str))
            .sort_values("missing_rate", ascending=False)
        )
        print("[INFO] feature missing rates (desc):")
        for _, r in miss_df.iterrows():
            print(f"  {r['var_name']}: {r['missing_rate']:.6f}")
        print("-" * 78)
    else:
        print("[INFO] missing_rate not found in pool_meta; skip missing-rate listing.")


    # ----- run main loop -----
    llm_time = run_parallel_interleaving(
        ds=datasetctx,
        exp_spec=exp_spec,
    )

    total_time = time.perf_counter() - TOTAL_START
    log_runtime(exp_spec, llm_time, total_time, out_root)
    print(f"[DONE] {exp_name} -> {out_root}")


def main():
    parser = argparse.ArgumentParser("Run multiple experiments from YAML config (bases/override supported)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")

    sel = parser.add_argument_group("Selection")
    sel.add_argument("--only", type=str, default="", help="Run only experiment with this exact name")
    sel.add_argument("--pattern", type=str, default="", help="Run only experiments whose name contains this substring")

    util = parser.add_argument_group("Utilities")
    util.add_argument("--list", action="store_true", help="List available experiments and exit")
    util.add_argument("--dry-run", action="store_true", help="Print materialized config for selected experiments and exit")

    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    exps = cfg.get("experiments", [])
    if not isinstance(exps, list) or not exps:
        raise ValueError("config.yaml must contain a non-empty 'experiments' list")

    selected = select_experiments(exps, only=args.only, pattern=args.pattern)

    print("[EXPERIMENTS]")
    for exp in exps:
        nm = exp.get("name", "")
        if nm:
            print(f" - {nm}")

    if args.list:
        return

    if not selected:
        print("[INFO] No experiments matched selection.")
        return

    # Materialize configs (apply base/override and defaults)
    mats: List[Dict[str, Any]] = [materialize_experiment(cfg, exp) for exp in selected]

    if args.dry_run:
        print("[DRY_RUN] Materialized experiment configs:\n")
        for m in mats:
            print("=" * 80)
            print(f"EXPERIMENT: {m.get('name')}")
            print("=" * 80)
            print(safe_json(m))
            print()
        return

    # Run experiments in order
    for m in mats:
        name = str(m.get("name", "unnamed_exp"))
        print("\n" + "=" * 80)
        print(f"[RUN] {name}")
        print("=" * 80)
        run_one_experiment(m)


if __name__ == "__main__":
    main()

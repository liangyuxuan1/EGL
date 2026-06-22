from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from plot_config_utils import (
    csv_set,
    enabled,
    figure_dir_name,
    figure_subdir,
    get_pair,
    iter_pairs,
    kg_path,
    load_plot_config,
    make_arg_parser,
    output_root,
)


def run_stable_kg(cfg: dict[str, Any], pair: dict[str, Any]) -> None:
    section = pair.get("stable_kg", {})
    if not enabled(section, default=True):
        return
    import plot_02_stable_kg as stable_kg

    out_dir = figure_subdir(cfg, pair, "kg_stable_figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    kg1 = kg_path(cfg, pair, "method_a")
    kg2 = kg_path(cfg, pair, "method_b")
    run_r = "yes" if section.get("run_r", False) else "no"
    print(f"[stable_kg] {pair['name']}: {kg1} vs {kg2} -> {out_dir} (run_r={run_r})")
    stable_kg.plot_node_edge_figures(
        str(kg1),
        str(kg2),
        str(out_dir),
        tag_a=pair.get("tag_a", "EGL"),
        tag_b=pair.get("tag_b", "EGL-LLM"),
        run_r=run_r,
        domain_map_csv=section.get("domain_map", cfg.get("stable_domain_map", "outputs/ct_clin_integration/COPD_ct_variable_summary.csv")),
    )
    if section.get("convergence", True):
        stable_kg.plot_convergence_figures(
            results_root=str(output_root(cfg, pair)),
            method_a=pair["method_a"],
            method_b=pair["method_b"],
            tag_a=pair.get("tag_a", "EGL"),
            tag_b=pair.get("tag_b", "EGL-LLM"),
            top_k_nodes=int(section.get("top_k_nodes", 15)),
            top_k_edges=int(section.get("top_k_edges", 15)),
            save_dir=str(out_dir),
        )


def run_iterations(cfg: dict[str, Any], pair: dict[str, Any]) -> None:
    section = pair.get("iterations", {})
    if not enabled(section, default=True):
        return
    import plot_01_iterations as plot_iter

    save_dir = figure_subdir(cfg, pair, "kg_evidence_plots")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[iterations] {pair['name']} -> {save_dir}")
    paths, plotted_cols = plot_iter.plot_iteration_log_reward_items(
        str(output_root(cfg, pair)),
        method_a=pair["method_a"],
        method_b=pair["method_b"],
        save_dir=str(save_dir),
    )
    print("  columns:", plotted_cols)
    for path in paths:
        print("  saved:", path)


def run_path_agenda(cfg: dict[str, Any], pair: dict[str, Any]) -> None:
    section = pair.get("path_agenda", {})
    if not enabled(section, default=True):
        return
    import plot_03_path_agenda as path_agenda

    if section.get("v1", True):
        save_dir = figure_subdir(cfg, pair, "kg_path_agenda_v1")
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"[path_agenda:v1] {pair['name']} -> {save_dir}")
        path_agenda.run_path_and_agenda_analysis(
            out_root=str(output_root(cfg, pair)),
            method_a=pair["method_a"],
            method_b=pair["method_b"],
            save_dir=str(save_dir),
            P_EDGE=section.get("p_edge", 30),
            topk=int(section.get("topk", 15)),
        )

    if section.get("v2", True):
        save_dir = figure_subdir(cfg, pair, "kg_path_agenda_v2")
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"[path_agenda:v2] {pair['name']} -> {save_dir}")
        path_agenda.main_paper_views(
            out_root=str(output_root(cfg, pair)),
            method_a=pair["method_a"],
            method_b=pair["method_b"],
            save_dir=str(save_dir),
            window_last=int(section.get("window_last", 50)),
            topk=int(section.get("topk", 15)),
        )


def run_edge_interaction(cfg: dict[str, Any], pair: dict[str, Any]) -> None:
    section = pair.get("edge_interaction", {})
    if not enabled(section, default=True):
        return
    import plot_04_edge_interaction as edge_int

    save_dir = figure_subdir(cfg, pair, "kg_edge_interaction_effects")
    save_dir.mkdir(parents=True, exist_ok=True)
    method_b = pair["method_b"]
    if section.get("recompute", False):
        print(f"[edge_interaction:compute] {pair['name']} -> {save_dir}")
        edge_int.run_edge_interaction_effect_analysis(
            out_root=str(output_root(cfg, pair)),
            method=method_b,
            top_k_edges=int(section.get("top_k_edges", 15)),
            metric_names=list(section.get("metrics_for_compute", ["auc", "auprc", "precision", "recall", "sensitivity", "specificity", "f1", "bal_acc"])),
            save_dir=str(save_dir),
        )
    else:
        print(f"[edge_interaction:plot-only] {pair['name']} -> {save_dir}")

    table_csv = save_dir / f"edge_effect_table_{method_b}.csv"
    if not table_csv.exists():
        raise FileNotFoundError(f"Missing edge interaction table: {table_csv}. Set recompute: true first.")
    edge_int.plot_edge_effect_table(
        str(table_csv),
        save_dir=str(save_dir),
        primary_metric=str(section.get("primary_metric", "auc")),
        scatter_x=str(section.get("scatter_x", "delta_mean")),
        metrics=list(section.get("metrics_for_plot", ["auc", "auprc", "specificity", "precision", "f1", "bal_acc", "sensitivity", "recall"])),
    )


def run_comparisons(cfg: dict[str, Any], only: set[str] | None = None) -> None:
    import plot_05_compare_figures as compare

    for comp in cfg.get("comparisons", []):
        if not enabled(comp, default=True):
            continue
        if only is not None and comp.get("name") not in only:
            continue
        left = get_pair(cfg, comp["left"])
        right = get_pair(cfg, comp["right"])
        output_dir = comp.get("output_dir")
        print(f"[compare] {comp.get('name', comp['left'] + '_vs_' + comp['right'])}")
        compare.compare_experiments(
            {"method_a": left["method_a"], "method_b": left["method_b"], "figure_dir": figure_dir_name(left)},
            {"method_a": right["method_a"], "method_b": right["method_b"], "figure_dir": figure_dir_name(right)},
            list(comp.get("figure_dirs", [])),
            dict(comp.get("image_names", {})),
            output_root=str(cfg.get("figure_root", cfg.get("output_root", "outputs"))),
            output_dir=output_dir,
        )


SECTION_RUNNERS = {
    "stable_kg": run_stable_kg,
    "iterations": run_iterations,
    "path_agenda": run_path_agenda,
    "edge_interaction": run_edge_interaction,
}


def run_config(config: str = "plot_config.yaml", only: set[str] | None = None, sections: set[str] | None = None, skip_comparisons: bool = False) -> None:
    cfg = load_plot_config(config)
    selected_sections = sections or set(SECTION_RUNNERS)
    for pair in iter_pairs(cfg, only=only):
        for name, runner in SECTION_RUNNERS.items():
            if name in selected_sections:
                runner(cfg, pair)
    if not skip_comparisons and (sections is None or "comparisons" in sections):
        run_comparisons(cfg)


def main(default_sections: list[str] | None = None) -> None:
    parser = make_arg_parser("Run EGL plotting from plot_config.yaml")
    args = parser.parse_args()
    only = csv_set(args.only)
    sections = csv_set(args.sections) or (set(default_sections) if default_sections else None)
    run_config(args.config, only=only, sections=sections, skip_comparisons=args.skip_comparisons)


if __name__ == "__main__":
    main()

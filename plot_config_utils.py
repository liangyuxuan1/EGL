from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import yaml

DEFAULT_CONFIG_PATH = Path("plot_config.yaml")


def load_plot_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = str(path)
    return cfg


def csv_set(value: str | None) -> set[str] | None:
    if value is None or value == "":
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def enabled(item: dict[str, Any], default: bool = True) -> bool:
    return bool(item.get("enabled", default))


def iter_pairs(cfg: dict[str, Any], only: Iterable[str] | None = None) -> list[dict[str, Any]]:
    only_set = set(only) if only else None
    pairs = []
    for pair in cfg.get("pairs", []):
        if only_set is not None and pair.get("name") not in only_set:
            continue
        if not enabled(pair):
            continue
        pairs.append(pair)
    return pairs


def get_pair(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    for pair in cfg.get("pairs", []):
        if pair.get("name") == name:
            return pair
    raise KeyError(f"Unknown pair in plot config: {name}")


def output_root(cfg: dict[str, Any], pair: dict[str, Any]) -> Path:
    return Path(pair.get("output_root", cfg.get("output_root", "outputs")))


def figure_root(cfg: dict[str, Any], pair: dict[str, Any]) -> Path:
    return Path(pair.get("figure_root", cfg.get("figure_root", cfg.get("output_root", "outputs"))))


def figure_dir_name(pair: dict[str, Any]) -> str:
    if pair.get("figure_dir"):
        return str(pair["figure_dir"])
    return f"Figures_{pair['method_a']}_vs_{pair['method_b']}"


def figure_subdir(cfg: dict[str, Any], pair: dict[str, Any], subdir: str) -> Path:
    return figure_root(cfg, pair) / figure_dir_name(pair) / subdir


def kg_path(cfg: dict[str, Any], pair: dict[str, Any], side: str) -> Path:
    method = pair[side]
    return output_root(cfg, pair) / method / "kg_snapshot.json"


def make_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Plot config YAML path.")
    parser.add_argument("--only", default=None, help="Comma-separated pair names to run, e.g. status,prediction.")
    parser.add_argument("--skip-comparisons", action="store_true", help="Skip configured comparison figures.")
    parser.add_argument("--sections", default=None, help="Comma-separated plot sections to run.")
    return parser

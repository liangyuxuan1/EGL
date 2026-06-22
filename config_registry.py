# config_registry.py
from __future__ import annotations

from typing import Dict

# 已有的定义：保持“权威来源”在这些 config_*.py 里
from config_cvd import CVD_EVENT_COLS

# 未来扩展时会加：
# from config_copd import COPD_EVENT_COLS
# from config_lung_cancer import LC_EVENT_COLS


# A simple registry: string key -> Dict[col_name, vtype]
EVENT_COLS_REGISTRY: Dict[str, Dict[str, str]] = {
    # COPDGene / NLST CVD events (baseline + since-last-visit)
    "CVD_EVENT_COLS": CVD_EVENT_COLS,

    # Examples for future:
    # "COPD_EVENT_COLS": COPD_EVENT_COLS,
    # "LC_EVENT_COLS": LC_EVENT_COLS,
}
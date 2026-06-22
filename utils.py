from typing import Any, Dict, List, Optional, Tuple, Set, Deque
import math
import numpy as np
import random

# -------------------------------------------------------------------
# Stochastic helpers
# -------------------------------------------------------------------
def set_global_seed(seed: int) -> random.Random:
    random.seed(seed)
    np.random.seed(seed)
    return random.Random(seed)

# -------------------------------------------------------------------
# helper functions
# -------------------------------------------------------------------

def sig4(x: float) -> float:
    return float(f"{x:.4g}")

def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))

def safe_float(x: Any, default: float = 0.0) -> float:
    """Convert to float safely; fallback to default for None/NaN/inf."""
    try:
        v = float(x)
        if not math.isfinite(v):
            return float(default)
        return v
    except Exception:
        return float(default)

def as_float(d: Dict[str, Any], key: str) -> float:
    # 不做 fallback：key 不存在就抛错，避免静默错误
    if key not in d:
        raise KeyError(f"{d} missing required key: {key}")
    return safe_float(d[key])


def safe_int(x: Any, default: int = 0) -> int:
    try:
        v = int(x)
        return int(v)
    except Exception:
        return int(default)


def safe_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x).strip()


def sigmoid(x: float) -> float:
    # stable sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def trust_weight(support: float, tau: float, *, min_w: float = 0.0) -> float:
    """
    support -> (min_w, 1] 的“可信度权重”
    - 小 support：权重接近 min_w（不为0，避免早期全为0导致无梯度）
    - 大 support：渐近 1

    tau 越大：增长越慢（更保守）
    """
    s = max(0.0, float(support))
    tau = max(1e-6, float(tau))
    w = 1.0 - math.exp(-s / tau)
    return float(max(min_w, min(1.0, w)))
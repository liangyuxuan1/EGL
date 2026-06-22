from __future__ import annotations

import sqlite3
import time
from typing import Any, Dict, Optional, List, Tuple, TYPE_CHECKING
import numpy as np
import hashlib
from pathlib import Path

import utils_models
from utils_graph import set_to_bitvec
if TYPE_CHECKING:
    from utils_graph import KnowledgeGraph

def bitvec_to_bytes(x: np.ndarray) -> bytes:
    """
    Pack 0/1 int array into compact bytes using np.packbits.
    F=75 -> 10 bytes.
    """
    x = np.asarray(x, dtype=np.uint8).reshape(-1)
    # enforce {0,1}
    x = (x > 0).astype(np.uint8)
    return np.packbits(x, bitorder="little").tobytes()

def bytes_key_hex(b: bytes) -> str:
    return b.hex()

def set_to_cache_key_hex(current_set: List[str], feat_index: Dict[str, int]) -> Tuple[str, bytes, int]:
    """
    Build a stable cache key for a feature set:
      key = hex(packed_bitmask)
    Returns:
      (key_hex, mask_bytes, k)
    """
    # canonicalize to avoid duplicates/noise
    clean = [f.strip() for f in current_set if isinstance(f, str) and f.strip()]
    clean = sorted(set(clean))  # important: stable identity
    x = set_to_bitvec(clean, feat_index)           # reuse your function
    b = bitvec_to_bytes(x)
    return bytes_key_hex(b), b, int(x.sum())


# 你 summary 里真正用到的 keys（mean/std）
SUMMARY_KEYS = [
    "auc", "auprc", "brier", "ece", "acc", "bal_acc",
    "precision", "recall", "f1", "sensitivity", "specificity",
]

class EvalCacheSQLite:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, timeout=30)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA temp_store=MEMORY;")
        self.conn.execute("PRAGMA cache_size=-200000;")  # ~200MB, tune if needed
        self._init_table()

    def _init_table(self):
        cols = []
        for m in SUMMARY_KEYS:
            cols.append(f"{m}_mean REAL")
            cols.append(f"{m}_std REAL")
        cols_sql = ",\n            ".join(cols)

        self.conn.execute(f"""
        CREATE TABLE IF NOT EXISTS eval_cache (
            key TEXT PRIMARY KEY,
            mask BLOB,
            k INTEGER,
            created_ts REAL,
            {cols_sql}
        );
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_cache_k ON eval_cache(k);")
        self.conn.commit()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.execute("SELECT * FROM eval_cache WHERE key = ? LIMIT 1;", (key,))
        row = cur.fetchone()
        if row is None:
            return None
        col_names = [d[0] for d in cur.description]
        rec = dict(zip(col_names, row))

        # rebuild your metrics_summary format (no per_fold)
        summary: Dict[str, Any] = {"model": "XGBoost", "cv_splits": 5}
        for m in SUMMARY_KEYS:
            summary[f"{m}_mean"] = rec.get(f"{m}_mean", float("nan"))
            summary[f"{m}_std"]  = rec.get(f"{m}_std", float("nan"))
        summary["n_features"] = int(rec.get("k", 0))
        return summary

    def put(self, key: str, mask_bytes: bytes, k: int, metrics_summary: Dict[str, Any]):
        created_ts = float(time.time())

        cols = ["key", "mask", "k", "created_ts"]
        vals = [key, sqlite3.Binary(mask_bytes), int(k), created_ts]

        for m in SUMMARY_KEYS:
            cols.append(f"{m}_mean")
            vals.append(float(metrics_summary.get(f"{m}_mean", float("nan"))))
            cols.append(f"{m}_std")
            vals.append(float(metrics_summary.get(f"{m}_std", float("nan"))))

        placeholders = ",".join(["?"] * len(cols))
        sql = f"INSERT OR REPLACE INTO eval_cache ({','.join(cols)}) VALUES ({placeholders});"
        self.conn.execute(sql, vals)
        self.conn.commit()

    def close(self):
        try:
            self.conn.commit()
        finally:
            self.conn.close()


def cached_evaluate_auc_and_more_cv(
    *,
    current_set: List[str],
    kg: KnowledgeGraph,
    cache: EvalCacheSQLite,
    df,
    label,
    selected_schema: Dict[str, Any],
    model_name: str = "Logistic",
    seed: int = 42,
) -> Tuple[Dict[str, Any], bool, str]:
    """
    Returns:
      metrics_summary, cache_hit, cache_key
    """
    key_mask, mask_bytes, k = set_to_cache_key_hex(current_set, kg.feat_index)
    key = f"{model_name}|seed={int(seed)}|{key_mask}"

    hit = cache.get(key)
    if hit is not None:
        return hit, True, key

    # cache miss -> real CV
    metrics_summary, cv_lines, dropped_msg, hard_index_pack, fold_df = utils_models.evaluate_auc_and_more_cv(
        df=df,
        label=label,
        feature_schema=selected_schema,
        model_name=model_name,
        seed=int(seed),
    )
    metrics_summary = metrics_summary or {}
    metrics_summary["n_features"] = int(k)

    cache.put(key, mask_bytes, k, metrics_summary)
    return metrics_summary, False, key


def feature_universe_signature(index_feat: List[str]) -> str:
    payload = "\n".join(index_feat).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def init_eval_cache_with_universe_guard(
    *,
    kg: KnowledgeGraph,
    eval_cache_path: str,
    universe_sig_path: str,
) -> tuple[EvalCacheSQLite, str]:
    """
    - Compute FEATURE_UNIVERSE_SHA256 from kg.index_feat
    - Persist to universe_sig_path
    - On each run, compare with existing signature:
        * if different -> delete eval_cache_path + universe_sig_path (reset)
        * if same      -> keep cache
    Returns:
        (eval_cache_obj, current_sig)
    """
    cache_fp = Path(eval_cache_path)
    sig_fp = Path(universe_sig_path)

    cur_sig = feature_universe_signature(getattr(kg, "index_feat", []))

    # compare with previous signature if exists
    prev_sig = None
    if sig_fp.exists():
        try:
            prev_sig = sig_fp.read_text(encoding="utf-8").strip()
        except Exception:
            prev_sig = None

    # if changed -> reset cache + sig log
    if prev_sig and prev_sig != cur_sig:
        # remove cache db
        try:
            if cache_fp.exists():
                cache_fp.unlink()
        except Exception as e:
            print(f"[WARN] failed to delete eval_cache: {cache_fp} ({e})")

        # remove signature file
        try:
            if sig_fp.exists():
                sig_fp.unlink()
        except Exception as e:
            print(f"[WARN] failed to delete universe sig file: {sig_fp} ({e})")

        # after reset, write the new signature
        try:
            sig_fp.write_text(cur_sig + "\n", encoding="utf-8")
        except Exception as e:
            print(f"[WARN] failed to write universe sig file: {sig_fp} ({e})")

        print("[CACHE_RESET] FEATURE_UNIVERSE_SHA256 changed -> reset eval_cache + sig log")
    else:
        # no previous sig or same
        if not prev_sig:
            # first run or unreadable -> write it
            try:
                sig_fp.parent.mkdir(parents=True, exist_ok=True)
                sig_fp.write_text(cur_sig + "\n", encoding="utf-8")
            except Exception as e:
                print(f"[WARN] failed to write universe sig file: {sig_fp} ({e})")

        print("[CACHE_OK] FEATURE_UNIVERSE_SHA256 unchanged")

    # open cache AFTER potential reset
    eval_cache = EvalCacheSQLite(str(cache_fp))
    return eval_cache, cur_sig

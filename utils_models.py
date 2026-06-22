from typing import Any, Dict, List, Optional, Tuple
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from math import exp
import numpy as np
import pandas as pd
import os
import json
import ast
from openai import OpenAI
from openai import APITimeoutError, APIConnectionError, RateLimitError
import httpx
import time

# -------------------------------------------------------------------
# LLM server config
# -------------------------------------------------------------------
# EGL-LLM uses an OpenAI-compatible Qwen endpoint.  Public code must not
# include private endpoint addresses or API keys, so all values are read from
# environment variables.
#
# Example:
#   export EGL_QWEN_API_BASE=http://localhost:8001/v1
#   export EGL_QWEN_API_KEY=your_api_key_or_placeholder
#   export EGL_QWEN_MODEL=Qwen3-14B-Instruct

llm_models = {
    "Qwen": {
        "model_name": os.getenv("EGL_QWEN_MODEL", "Qwen3-14B-Instruct"),
        "api_base": os.getenv("EGL_QWEN_API_BASE", "http://localhost:8001/v1"),
        "api_key": os.getenv("EGL_QWEN_API_KEY", "EMPTY"),
    },
}

_CLIENT_CACHE = {}

# -------------------------------------------------------------------
# LLM helpers
# -------------------------------------------------------------------
def get_llm_client(llm_model: str):
    model = llm_models.get(llm_model)
    if model is None:
        raise ValueError(f"Invalid LLM model: {llm_model}")

    model_name = model["model_name"]
    api_key = model["api_key"]
    api_base = model["api_base"]
    cache_key = (api_base, api_key)

    if cache_key not in _CLIENT_CACHE:
        timeout = httpx.Timeout(
            connect=30.0,
            read=900.0,
            write=60.0,
            pool=60.0,
        )
        _CLIENT_CACHE[cache_key] = OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout,
            max_retries=2,
        )

    return _CLIENT_CACHE[cache_key], model_name


def probe_server(llm_model: str) -> bool:
    """Quick ping to ensure the LLM endpoint is reachable."""
    try:
        client, model_name = get_llm_client(llm_model)
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": [{"type": "text", "text": "ping"}]}],
            max_tokens=8,
        )
        txt = resp.choices[0].message.content
        print(f"[INFO] LLM ping response: {txt}")
        return True
    except Exception as exc:
        print(f"[WARN] LLM ping failed: {exc}")
        return False




def ask_llm(llm_model: str, messages: List[Dict], enable_thinking: bool = False) -> Dict:
    """
    LLM call with strict-JSON expectation.
    If the model returns extra text, we try to extract the JSON block.
    """
    client, model_name = get_llm_client(llm_model)

    max_attempts = 3
    last_err = None

    for attempt in range(max_attempts):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.2,
                max_tokens=1200,
                response_format={"type": "json_object"},
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": enable_thinking},
                },
            )

            txt = completion.choices[0].message.content or ""

            try:
                return json.loads(txt)
            except Exception:
                try:
                    start = txt.index("{")
                    end = txt.rindex("}") + 1
                    return json.loads(txt[start:end])
                except Exception:
                    print(f"[ask_llm] return: {txt}\n")
                    print("[ask_llm] failed to parse JSON from model output")
                    return {}

        except (APITimeoutError, APIConnectionError, RateLimitError) as e:
            last_err = e
            wait_s = 2 * (attempt + 1)
            print(f"[ask_llm] attempt {attempt + 1}/{max_attempts} failed: {type(e).__name__}: {e}")
            time.sleep(wait_s)

        except Exception as e:
            print(f"[ask_llm] unexpected error: {type(e).__name__}: {e}")
            return {}

    print(f"[ask_llm] all retries failed: {type(last_err).__name__}: {last_err}")
    return {}

# -------------------------------------------------------------------
# Metrics: diversity / coverage / redundancy / stability
# -------------------------------------------------------------------
def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def diversity_score(selected: List[str], past_sets: List[List[str]]) -> float:
    """
    in [0,1], higher => more different from prior selections.
    Using 1 - max_jaccard_similarity vs recent history.
    """
    if not past_sets:
        return 1.0
    sims = [jaccard(selected, ps) for ps in past_sets if ps]
    return float(1.0 - max(sims)) if sims else 1.0


def coverage_score(selected: List[str], pool_meta: pd.DataFrame) -> float:
    """
    fraction of domains covered by current selection, relative to pool domains.
    """
    if "clinical_domain" not in pool_meta.columns:
        return 0.0
    pool_domains = set(safe_text(x) for x in pool_meta["clinical_domain"].tolist() if safe_text(x))
    if not pool_domains:
        return 0.0
    sel_meta = pool_meta[pool_meta["var_name"].isin(selected)]
    sel_domains = set(safe_text(x) for x in sel_meta["clinical_domain"].tolist() if safe_text(x))
    return float(len(sel_domains) / max(1, len(pool_domains)))


def parsimony_score(n_features: int, k_target: int = 20) -> float:
    """Simple parsimony: if <= k_target then 1 else k_target/n."""
    if n_features <= 0:
        return 0.0
    if n_features <= k_target:
        return 1.0
    return float(k_target / n_features)


def redundancy_penalty(df: pd.DataFrame, selected: List[str], max_pairs: int = 2000) -> float:
    """
    Simple redundancy proxy: mean abs corr among RAW numeric selected features.
    (Not perfect, but cheap & monotonic.)
    """
    nums = [c for c in selected if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(nums) < 2:
        return 0.0
    X = df[nums].apply(pd.to_numeric, errors="coerce")
    corr = X.corr().abs().values
    iu = np.triu_indices_from(corr, k=1)
    vals = corr[iu]
    if vals.size == 0:
        return 0.0
    if vals.size > max_pairs:
        vals = vals[:max_pairs]
    return float(np.nanmean(vals))


# -------------------------------------------------------------------
# >>> V5 NEW: extended CV evaluation (AUC + AUPRC + calibration + FN/FP metrics)
# -------------------------------------------------------------------
def _ece_binary(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE) for binary classification.
    - bin probabilities into n_bins
    - compute weighted |acc - conf| across bins
    """
    y_true = y_true.astype(int)
    y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        acc = float((y_true[mask] == (y_prob[mask] >= 0.5).astype(int)).mean())
        conf = float(y_prob[mask].mean())
        ece += (mask.sum() / max(1, n)) * abs(acc - conf)

    return float(ece)


def auc_penalty_factor(best_auc: float, auc_mean: float, tau: float, eps: float = 1e-4) -> float:
    """
    V4 penalty was:
      exp(-(best_auc - auc_mean)/tau) if auc_mean < best_auc else 1

    >>> V5 NEW (more engineering-friendly):
      - Use a tolerance band: if best_auc - auc_mean <= eps, no penalty.
      - best_auc should be "best_auc_so_far within this run" (already in your code).
    """
    if tau <= 1e-8:
        tau = 1e-3
    gap = float(best_auc) - float(auc_mean)
    if gap <= float(eps):
        return 1.0
    return float(exp(-max(0.0, gap) / tau))



def identify_hard_samples(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    delta: float = 0.15,
    soft_fp_max: float = 0.7,
    soft_fn_min: float = 0.3,
    conf_fp_min: float = 0.9,
    conf_fn_max: float = 0.1,
    exclusive: bool = True,   # <<< NEW: whether to enforce disjoint buckets
) -> dict:
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred_prob).astype(float)
    p = np.clip(p, 1e-6, 1 - 1e-6)

    # base masks
    uncertain = np.abs(p - 0.5) <= float(delta)

    fp = (p >= 0.5) & (y_true == 0)
    fn = (p < 0.5) & (y_true == 1)

    fp_soft = fp & (p <= float(soft_fp_max))
    fn_soft = fn & (p >= float(soft_fn_min))

    confident_error = (fp & (p >= float(conf_fp_min))) | (fn & (p <= float(conf_fn_max)))

    if exclusive:
        # prioritize error buckets; uncertain becomes "uncertain_only"
        uncertain_only = uncertain & (~fp_soft) & (~fn_soft)
        fp_soft_only = fp_soft
        fn_soft_only = fn_soft
    else:
        # multi-label
        uncertain_only = uncertain
        fp_soft_only = fp_soft
        fn_soft_only = fn_soft

    hard_any = uncertain_only | fp_soft_only | fn_soft_only

    return {
        "hard_uncertain_index": np.where(uncertain_only)[0],
        "hard_fp_soft_index": np.where(fp_soft_only)[0],
        "hard_fn_soft_index": np.where(fn_soft_only)[0],
        "hard_any_index": np.where(hard_any)[0],
        "confident_error_index": np.where(confident_error)[0],
        "meta": {
            "exclusive": bool(exclusive),
            "delta": float(delta),
            "soft_fp_max": float(soft_fp_max),
            "soft_fn_min": float(soft_fn_min),
            "conf_fp_min": float(conf_fp_min),
            "conf_fn_max": float(conf_fn_max),
        }
    }


def evaluate_auc_and_more_cv(
    df,
    label,
    feature_schema: Dict[str, str],
    model_name: str = "XGBoost",
    hard_cfg: Dict[str, float] = None,   # <<< NEW: thresholds for identify_hard_samples
    seed: int = 42,
) -> Tuple[Dict[str, Any], List[str], str, Dict[str, List[Any]]]:
    """
    V6+ (hard-case aware) CV evaluation:
      - Per fold: after prob computed and BEFORE thresholding:
          call identify_hard_samples
      - Convert fold-local indices -> global df.index values
      - Concatenate across folds, de-duplicate (stable order not required)

    Returns:
      summary, lines, dropped_msg, hard_index_pack
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, brier_score_loss,
        accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix,
    )

    # ---- defaults for hard-case thresholds ----
    if hard_cfg is None:
        hard_cfg = dict(
            delta=0.15,
            soft_fp_max=0.7,
            soft_fn_min=0.3,
            conf_fp_min=0.9,
            conf_fn_max=0.1,
        )

    # Accept either pandas Series or numpy-like label input.
    if isinstance(label, pd.Series):
        label_series = label.reset_index(drop=True)
    else:
        label_series = pd.Series(np.asarray(label), index=np.arange(len(df)))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    metric_keys = [
        "auc", "auprc", "brier", "ece",
        "acc", "bal_acc", "precision", "recall", "f1",
        "sensitivity", "specificity"
    ]
    fold_metrics: Dict[str, List[float]] = {k: [] for k in metric_keys}
    dropped_msgs: List[str] = []

    # >>> NEW: global index collectors (df.index values)
    hard_global = {
        "hard_uncertain_index": [],
        "hard_fp_soft_index": [],
        "hard_fn_soft_index": [],
        "hard_any_index": [],
        "confident_error_index": [],
    }

    def _append_nan_fold():
        for k in metric_keys:
            fold_metrics[k].append(float("nan"))

    for fold_i, (tr_idx, te_idx) in enumerate(skf.split(df, label_series), start=1):
        df_tr = df.iloc[tr_idx]
        y_tr = label_series.iloc[tr_idx].values.astype(int)
        df_te = df.iloc[te_idx]
        y_te = label_series.iloc[te_idx].values.astype(int)

        # --- freeze cols on fold-train ---
        Xtr, pre, msg_tr, kept_cols = prepare_data_v6(df_tr, feature_schema, kept_columns=None)
        dropped_msgs.append(f"[Fold {fold_i}] {msg_tr}")

        if (kept_cols is None) or (len(kept_cols) == 0) or (Xtr.shape[1] == 0):
            _append_nan_fold()
            continue

        # --- apply same kept_cols to test split ---
        Xte, _, msg_te, _ = prepare_data_v6(df_te, feature_schema, kept_columns=kept_cols)
        if Xte.shape[1] == 0:
            dropped_msgs.append(f"[Fold {fold_i}] [VAL] Xte has 0 cols; msg={msg_te}")
            _append_nan_fold()
            continue

        pos = int(y_tr.sum())
        neg = int(len(y_tr) - pos)

        models = build_models_v6(
            names=[model_name],
            preprocessor=pre,
            feature_schema=feature_schema,
            pos=pos,
            neg=neg,
        )
        pipe = models.get(model_name, None)
        if pipe is None:
            dropped_msgs.append(f"[Fold {fold_i}] Model '{model_name}' not available; skipped.")
            _append_nan_fold()
            continue

        # --- Train / Predict ---
        try:
            pipe.fit(Xtr, y_tr)
            prob = np.clip(pipe.predict_proba(Xte)[:, 1], 1e-6, 1 - 1e-6)

            # >>> NEW: identify hard samples BEFORE pred binarization
            hard_pack = identify_hard_samples(y_true=y_te, y_pred_prob=prob, **hard_cfg)

            # fold-local -> global df.index mapping
            te_index = df_te.index.to_numpy()
            for k in hard_global.keys():
                loc = np.asarray(hard_pack.get(k, []), dtype=int)
                if loc.size > 0:
                    hard_global[k].extend(te_index[loc].tolist())

            # now threshold
            pred = (prob >= 0.5).astype(int)

        except Exception as e:
            dropped_msgs.append(f"[Fold {fold_i}] Training/predict failed: {type(e).__name__}: {e}")
            _append_nan_fold()
            continue

        # --- Metrics ---
        auc = roc_auc_score(y_te, prob) if len(np.unique(y_te)) > 1 else np.nan
        auprc = average_precision_score(y_te, prob) if len(np.unique(y_te)) > 1 else np.nan
        brier = brier_score_loss(y_te, prob)
        ece = _ece_binary(y_te, prob, n_bins=10)

        acc = accuracy_score(y_te, pred)
        bal_acc = balanced_accuracy_score(y_te, pred)
        prec = precision_score(y_te, pred, zero_division=0)
        rec = recall_score(y_te, pred, zero_division=0)
        f1 = f1_score(y_te, pred, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
        sens = tp / max(1, (tp + fn))
        spec = tn / max(1, (tn + fp))

        fold_metrics["auc"].append(float(auc))
        fold_metrics["auprc"].append(float(auprc))
        fold_metrics["brier"].append(float(brier))
        fold_metrics["ece"].append(float(ece))
        fold_metrics["acc"].append(float(acc))
        fold_metrics["bal_acc"].append(float(bal_acc))
        fold_metrics["precision"].append(float(prec))
        fold_metrics["recall"].append(float(rec))
        fold_metrics["f1"].append(float(f1))
        fold_metrics["sensitivity"].append(float(sens))
        fold_metrics["specificity"].append(float(spec))

    # --- Summaries (nan-aware) ---
    summary: Dict[str, Any] = {"model": model_name, "cv_splits": 5}
    for k, vals in fold_metrics.items():
        arr = np.array(vals, dtype=float)
        summary[f"{k}_mean"] = float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan
        summary[f"{k}_std"] = float(np.nanstd(arr)) if np.isfinite(arr).any() else np.nan

    # >>> NEW: summarize hard counts (useful in logs)
    hard_summary_counts = {k: int(len(set(v))) for k, v in hard_global.items()}
    summary["hard_counts"] = hard_summary_counts
    summary["hard_cfg"] = dict(hard_cfg)

    lines = [
        f"{model_name} CV ROC-AUC: mean={summary['auc_mean']:.3f}, std={summary['auc_std']:.3f}",
        f"{model_name} CV AUPRC:  mean={summary['auprc_mean']:.3f}, std={summary['auprc_std']:.3f}",
        f"{model_name} CV Brier:  mean={summary['brier_mean']:.3f}, std={summary['brier_std']:.3f}",
        f"{model_name} CV ECE:    mean={summary['ece_mean']:.3f}, std={summary['ece_std']:.3f}",
        f"{model_name} CV BalAcc: mean={summary['bal_acc_mean']:.3f}, std={summary['bal_acc_std']:.3f}",
        f"{model_name} CV Sens/Spec: mean={summary['sensitivity_mean']:.3f}/{summary['specificity_mean']:.3f}",
        f"Hard counts (OOF, unique): {hard_summary_counts}",
    ]

    if dropped_msgs:
        dropped_msg = "Drop/skip notes (last 3 folds): " + " | ".join(dropped_msgs[-3:])
    else:
        dropped_msg = "No drop message."

    # >>> NEW: de-duplicate global indices (keep as list for JSON)
    hard_index_pack = {}
    for k, vals in hard_global.items():
        # use set for uniqueness; if你想保留稳定顺序，可改成 dict.fromkeys(vals)
        hard_index_pack[k] = list(dict.fromkeys(vals))

    # 返回每个fold的metrics，为S05 consolidateion做准备
    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_df.insert(0, "fold", np.arange(1, len(fold_metrics_df)+1))

    return summary, lines, dropped_msg, hard_index_pack, fold_metrics_df


def build_feature_reliability_summary(
    df: pd.DataFrame,
    pool_meta: pd.DataFrame,
    *,
    y_true: Optional[pd.Series] = None,                         # 0/1 labels aligned to df.index
    hard_index_pack: Optional[Dict[str, List[Any]]] = None,     # dict of df.index labels (GLOBAL)
    focus_features: Optional[List[str]] = None,
    outlier_prefix: str = "is_outlier__",
) -> Dict[str, Any]:
    """
    Reliability summary for LLM.

    Base:
      - missing_rate
      - outlier_rate_overall (prefer meta.outlier_rate; fallback to df flag mean)

    Conditional (if y_true):
      - outlier_rate_pos / outlier_rate_neg
      - gap_pos_minus_neg

    Hard buckets (if hard_index_pack):
      Uses EXACT keys from evaluate_auc_and_more_cv:
        - hard_uncertain_index
        - hard_fp_soft_index
        - hard_fn_soft_index
        - hard_any_index
        - confident_error_index
      For each bucket:
        - outlier_rate_<bucket>
        - gap_<bucket>_minus_neg (if neg available)

    NOTE:
      - hard_index_pack values are treated as df.index LABELS only (no positional fallback).
      - All rates use NON-missing entries of the variable as denominator.
      - Requires per-variable outlier flags in df: is_outlier__<var> (0/1).
    """
    out: Dict[str, Any] = {
        "available": True,
        "mode": "base",
        "items": [],
        "diagnostics": {},
    }

    # -----------------------------
    # lookups from meta
    # -----------------------------
    dom_lookup = dict(zip(pool_meta["var_name"], pool_meta["clinical_domain"])) if "clinical_domain" in pool_meta.columns else {}
    vtype_lookup = dict(zip(pool_meta["var_name"], pool_meta["vtype"])) if "vtype" in pool_meta.columns else {}
    overall_outlier_lookup = dict(zip(pool_meta["var_name"], pool_meta["outlier_rate"])) if "outlier_rate" in pool_meta.columns else {}

    # -----------------------------
    # conditioning masks (pos/neg)
    # -----------------------------
    pos_mask = neg_mask = None
    if y_true is not None:
        y = pd.to_numeric(y_true, errors="coerce").reindex(df.index)
        pos_mask = (y == 1)
        neg_mask = (y == 0)
        out["mode"] = "conditional"

    # -----------------------------
    # hard masks (index-label only)
    # -----------------------------
    expected_keys = [
        "hard_uncertain_index",
        "hard_fp_soft_index",
        "hard_fn_soft_index",
        "hard_any_index",
        "confident_error_index",
    ]

    hard_masks: Dict[str, pd.Series] = {}
    if isinstance(hard_index_pack, dict) and hard_index_pack:
        any_found = False
        for k in expected_keys:
            if k in hard_index_pack:
                idx_list = hard_index_pack.get(k, [])
                if idx_list is None:
                    idx_list = []
                # treat as df.index labels ONLY
                hard_masks[k] = pd.Series(df.index.isin(list(idx_list)), index=df.index)
                any_found = True
        if any_found:
            out["mode"] = "conditional+hard" if out["mode"] != "base" else "hard"

    # -----------------------------
    # variables to summarize
    # -----------------------------
    vars_all = [v for v in pool_meta["var_name"].tolist() if v in df.columns]
    if focus_features is not None:
        focus = {v for v in focus_features if isinstance(v, str)}
        vars_all = [v for v in vars_all if v in focus]

    # -----------------------------
    # helper: get outlier flag series
    # -----------------------------
    def _get_flag_series(var: str) -> Optional[pd.Series]:
        c = f"{outlier_prefix}{var}"
        if c not in df.columns:
            return None
        s = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        return (s > 0)

    # -----------------------------
    # per-variable loop
    # -----------------------------
    for v in vars_all:
        x = df[v]
        non = x.notna()
        miss_rate = float((~non).mean())

        flag = _get_flag_series(v)

        # overall outlier rate: prefer meta.outlier_rate, fallback to df flag mean on non-missing
        overall = overall_outlier_lookup.get(v, np.nan)
        if overall is None:
            overall = np.nan
        if (not np.isfinite(overall)) and (flag is not None):
            denom = int(non.sum())
            overall = float((flag & non).sum() / denom) if denom > 0 else np.nan

        item: Dict[str, Any] = {
            "var": v,
            "domain": (str(dom_lookup.get(v, "")).strip() or "unknown"),
            "vtype": (str(vtype_lookup.get(v, "")).strip()
                      or ("num" if pd.api.types.is_numeric_dtype(x) else "cat")),
            "missing_rate": miss_rate,
            "outlier_rate_overall": float(overall) if np.isfinite(overall) else np.nan,
            "has_outlier_flag": bool(flag is not None),
        }

        # --- conditional pos/neg ---
        if (flag is not None) and (pos_mask is not None) and (neg_mask is not None):
            pos_denom = int((pos_mask & non).sum())
            neg_denom = int((neg_mask & non).sum())

            item["outlier_rate_pos"] = float((flag & pos_mask & non).sum() / pos_denom) if pos_denom > 0 else np.nan
            item["outlier_rate_neg"] = float((flag & neg_mask & non).sum() / neg_denom) if neg_denom > 0 else np.nan

            if np.isfinite(item["outlier_rate_pos"]) and np.isfinite(item["outlier_rate_neg"]):
                item["gap_pos_minus_neg"] = float(item["outlier_rate_pos"] - item["outlier_rate_neg"])
            else:
                item["gap_pos_minus_neg"] = np.nan

        # --- hard buckets (exact keys) ---
        if (flag is not None) and hard_masks:
            for hk, hm in hard_masks.items():
                denom = int((hm & non).sum())
                rate = float((flag & hm & non).sum() / denom) if denom > 0 else np.nan
                item[f"outlier_rate_{hk}"] = rate

                if np.isfinite(rate) and np.isfinite(item.get("outlier_rate_neg", np.nan)):
                    item[f"gap_{hk}_minus_neg"] = float(rate - item["outlier_rate_neg"])
                else:
                    item[f"gap_{hk}_minus_neg"] = np.nan

        # --- tags (use exact gap names) ---
        tags: List[str] = []

        g_conf = item.get("gap_confident_error_index_minus_neg", np.nan)
        if np.isfinite(g_conf) and g_conf >= 0.05:
            tags.append("confident_error_outlier_prone")

        g_fn = item.get("gap_hard_fn_soft_index_minus_neg", np.nan)
        if np.isfinite(g_fn) and g_fn >= 0.05:
            tags.append("fn_soft_outlier_prone")

        g_fp = item.get("gap_hard_fp_soft_index_minus_neg", np.nan)
        if np.isfinite(g_fp) and g_fp >= 0.05:
            tags.append("fp_soft_outlier_prone")

        g_unc = item.get("gap_hard_uncertain_index_minus_neg", np.nan)
        if np.isfinite(g_unc) and g_unc >= 0.05:
            tags.append("uncertain_outlier_prone")

        gpn = item.get("gap_pos_minus_neg", np.nan)
        if np.isfinite(gpn):
            if gpn >= 0.05:
                tags.append("pos_outlier_prone")
            elif gpn <= -0.05:
                tags.append("neg_outlier_prone")

        item["risk_tag"] = tags[:4]
        out["items"].append(item)

    # -----------------------------
    # ranking
    # -----------------------------
    def _nz(x):
        return float(x) if np.isfinite(x) else 0.0

    def _rk_base(it: Dict[str, Any]):
        mr = float(it.get("missing_rate", 1.0))
        ol = _nz(it.get("outlier_rate_overall", np.nan))
        return (mr, ol)

    def _rk_cond(it: Dict[str, Any]):
        g_conf = _nz(it.get("gap_confident_error_index_minus_neg", np.nan))
        g_fn   = _nz(it.get("gap_hard_fn_soft_index_minus_neg", np.nan))
        g_fp   = _nz(it.get("gap_hard_fp_soft_index_minus_neg", np.nan))
        g_unc  = _nz(it.get("gap_hard_uncertain_index_minus_neg", np.nan))
        ol     = _nz(it.get("outlier_rate_overall", np.nan))
        mr = float(it.get("missing_rate", 1.0))
        return (-g_conf, -g_fn, -g_fp, -g_unc, -ol, mr)

    if out["mode"] == "base":
        out["items"] = sorted(out["items"], key=_rk_base)
    else:
        out["items"] = sorted(out["items"], key=_rk_cond)

    out["diagnostics"] = {
        # total variables considered (after optional focus_features filtering)
        "n_vars_total": int(len(vars_all)),
        "has_meta_outlier_rate": bool("outlier_rate" in pool_meta.columns),
        "has_y_true": bool(y_true is not None),
        "has_hard_index_pack": bool(bool(hard_masks)),
        "hard_buckets": [k for k in expected_keys if k in hard_masks],
        "focus_features_only": bool(focus_features is not None),
        "outlier_flag_prefix": outlier_prefix,
        "expected_hard_pack_keys": expected_keys,
        "hard_bucket_sizes": {k: int(hard_masks[k].sum()) for k in hard_masks} if hard_masks else {},
    }
    return out


def inspect_reliability_summary(
    reliability_summary: Dict[str, Any],
    info: str = "",
    *,
    top_k: int = 10,
    print_all: bool = False,
) -> None:
    """
    Print a human-readable diagnostic report for reliability_summary.

    For DEVELOPER sanity check, not for LLM.

    It answers:
      1) Is missing / outlier signal reasonable?
      2) Are there clearly risky variables?
      3) Are risk_tag assignments meaningful?
      4) If hard buckets exist, are bucket-conditional outlier rates/gaps meaningful?
    """
    print("\n" + "-" * 78)
    print(f"[Reliability Summary Inspection] {info}")
    # print("-" * 78)

    if not reliability_summary.get("available", False):
        print("[ERROR] reliability_summary.available == False")
        return

    items = reliability_summary.get("items", [])
    if not items:
        print("[WARN] reliability_summary.items is empty.")
        return

    df_items = pd.DataFrame(items)
    mode = reliability_summary.get("mode", "unknown")
    print(f"[INFO] mode = {mode}")

    diag = reliability_summary.get("diagnostics", {}) or {}

    # ------------------------------------------------
    # Counts: avoid the old "n_vars" ambiguity
    # ------------------------------------------------
    n_items_returned = int(len(df_items))
    n_vars_total = diag.get("n_vars_total", None)

    print(f"[INFO] n_items_returned = {n_items_returned}")
    if n_vars_total is not None:
        print(f"[INFO] n_vars_total     = {int(n_vars_total)}")
        if int(n_vars_total) != n_items_returned:
            print(
                "[NOTE] n_items_returned != n_vars_total: "
                "this is expected if you have truncation OR if focus_features filters vars "
                "before ranking, OR if some vars are missing in df."
            )

    # ------------------------------------------------
    # Diagnostics dump (print once)
    # ------------------------------------------------
    if diag:
        print("[INFO] diagnostics:")
        for k, v in diag.items():
            print(f"  - {k}: {v}")

    # ------------------------------------------------
    # Basic statistics sanity check
    # ------------------------------------------------
    print("\n[1] Basic statistics (sanity check)")
    cols = [c for c in ["missing_rate", "outlier_rate_overall"] if c in df_items.columns]
    if cols:
        print(df_items[cols].describe())
    else:
        print("[WARN] No missing_rate / outlier_rate_overall columns found.")

    # ------------------------------------------------
    # Top missing variables
    # ------------------------------------------------
    if "missing_rate" in df_items.columns:
        print(f"\n[2] Top-{top_k} variables by missing_rate")
        show_cols = [c for c in ["var", "missing_rate", "outlier_rate_overall"] if c in df_items.columns]
        print(
            df_items.sort_values("missing_rate", ascending=False)
                   .head(top_k)[show_cols]
                   .to_string(index=False)
        )

    # ------------------------------------------------
    # Top overall outlier-prone variables
    # ------------------------------------------------
    if "outlier_rate_overall" in df_items.columns:
        print(f"\n[3] Top-{top_k} variables by outlier_rate_overall")
        show_cols = [c for c in ["var", "outlier_rate_overall", "missing_rate"] if c in df_items.columns]
        print(
            df_items.sort_values("outlier_rate_overall", ascending=False)
                   .head(top_k)[show_cols]
                   .to_string(index=False)
        )

    # ------------------------------------------------
    # Risk-tag overview
    # ------------------------------------------------
    print("\n[4] Risk-tag overview")
    if "risk_tag" not in df_items.columns:
        print("[WARN] risk_tag column not found.")
    else:
        tag_counts: Dict[str, int] = {}
        for tags in df_items["risk_tag"].dropna():
            if isinstance(tags, list):
                for t in tags:
                    tag_counts[t] = tag_counts.get(t, 0) + 1

        if not tag_counts:
            print("[INFO] No risk_tag assigned to any variable.")
        else:
            print("Tag counts:")
            for k, v in sorted(tag_counts.items(), key=lambda x: -x[1]):
                print(f"  - {k}: {v}")

            print(f"\nExamples of tagged variables (up to {top_k} per tag):")
            base_cols = [c for c in ["var", "missing_rate", "outlier_rate_overall"] if c in df_items.columns]
            for tag in sorted(tag_counts.keys()):
                sel = df_items[df_items["risk_tag"].apply(lambda x: isinstance(x, list) and (tag in x))]
                if not sel.empty:
                    print(f"\n  [{tag}]")
                    print(sel.head(top_k)[base_cols].to_string(index=False))

    # ------------------------------------------------
    # Hard-bucket related signals (new naming)
    #   build_feature_reliability_summary now emits:
    #     - outlier_rate_<bucket_key>
    #     - gap_<bucket_key>_minus_neg
    # ------------------------------------------------
    hard_rate_cols = [c for c in df_items.columns if c.startswith("outlier_rate_") and c != "outlier_rate_overall"]
    hard_gap_cols = [c for c in df_items.columns if c.startswith("gap_") and c.endswith("_minus_neg")]

    if hard_rate_cols or hard_gap_cols:
        print("\n[5] Hard-bucket related signals (summary)")

        # summarize bucket rate columns
        if hard_rate_cols:
            print("  [Rates]")
            for c in sorted(hard_rate_cols):
                s = pd.to_numeric(df_items[c], errors="coerce")
                if s.notna().any():
                    print(f"    - {c}: mean={s.mean():.4f}, max={s.max():.4f}")

        # summarize bucket gap columns
        if hard_gap_cols:
            print("  [Gaps vs neg]")
            for c in sorted(hard_gap_cols):
                s = pd.to_numeric(df_items[c], errors="coerce")
                if s.notna().any():
                    print(f"    - {c}: mean={s.mean():.4f}, max={s.max():.4f}")

    # ------------------------------------------------
    # Optional full dump
    # ------------------------------------------------
    if print_all:
        print("\n[6] Full table dump")
        print(df_items.to_string(index=False))

    print(f"\n[End of Reliability Summary Inspection] {info}")
    print("-" * 78)


def stability_proxy(selected_feats, iter_rows):
    # 简单代理：用最近若干轮 AUC 波动反映稳定性（越稳定越高）
    if not iter_rows:
        return 0.5
    aucs = []
    for r in iter_rows[-6:]:
        sc = r.get("scorecard", {})
        auc = sc.get("auc_mean", None)
        if auc is not None:
            try:
                aucs.append(float(auc))
            except Exception:
                pass
    if len(aucs) < 3:
        return 0.5
    sd = float(np.nanstd(aucs))
    # sd=0 -> 1.0; sd>=0.05 -> ~0
    return float(np.clip(1.0 - sd / 0.05, 0.0, 1.0))

def parsimony_score(n_features: int, top_k: int):
    # 越接近 top_k 的一个合理比例越好（这里鼓励更少一点，避免过拟合）
    if top_k <= 0:
        return 0.0
    ratio = n_features / float(top_k)
    # 0.5~1.0 区间较好
    return float(np.clip(1.0 - abs(ratio - 0.7), 0.0, 1.0))


def parse_scorecard(sc_raw: Any) -> Dict[str, Any]:
    """Robustly parse scorecard that may be dict, JSON string, or Python-literal string."""
    if isinstance(sc_raw, dict):
        return sc_raw
    if isinstance(sc_raw, str):
        # Try strict JSON
        try:
            obj = json.loads(sc_raw)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            pass
        # Fall back to Python literal (single quotes)
        try:
            obj = ast.literal_eval(sc_raw)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


if __name__ == "__main__":
    llm_model = "Qwen"

    if probe_server(llm_model):
        print("Test LLM server ok.")
    else:
        print("Test LLM server failed.")

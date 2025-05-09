"""
metrics.py  ·  unified metrics for fuzzy & classical ECG classifiers
--------------------------------------------------------------------------
• beat_level_metrics()   – per-beat performance, incl. F1, MCC, PR-AUC, ROC-AUC
• patient_level_metrics()– patient-centric sensitivity & false-alarms/hour

Changes v2 2025-05-06
────────────────────
1.  Consistent class ordering → always DEFAULT_LABELS first.
2.  Optional exclusion of the "unknown" class from macro metrics.
3.  Safer handling of edge-cases (division-by-zero, no positive samples…).
4.  False-alarms/hour now uses *total* time instead of simple mean.
"""
from __future__ import annotations
import pandas as pd

from typing import Iterable, Sequence, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    auc,
    balanced_accuracy_score,
    matthews_corrcoef,
)

# ---------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------
DEFAULT_LABELS: tuple[str, ...] = ("normalna", "mierna", "zavazna")
POS_LABEL: str = "zavazna"   # clinically most important


# =====================================================================
#  BEAT-LEVEL METRICS
# =====================================================================

def _ordered_labels(y_true: Iterable[str], y_pred: Iterable[str], *, keep_unknown: bool) -> list[str]:
    """Return a stable label order: DEFAULT_LABELS first, then extras."""
    present = set(y_true) | set(y_pred)
    ordered = [lbl for lbl in DEFAULT_LABELS if lbl in present]
    extras  = [lbl for lbl in sorted(present) if lbl not in DEFAULT_LABELS]
    if not keep_unknown:
        extras = [e for e in extras if e.lower() != "unknown"]
    return ordered + extras


def beat_level_metrics(
    y_true,
    y_pred,
    y_score: Sequence[float] | None = None,
    *,
    pos_label: str = POS_LABEL,
    keep_unknown: bool = False,
):
    """Compute per-beat metrics.

    Parameters
    ----------
    y_true, y_pred : array-like of str
        Ground-truth / predicted labels.
    y_score : optional, array-like of float
        Continuous severity score (e.g., centroid from fuzzy model). Needed for PR-AUC & ROC-AUC.
    pos_label : str, default "zavazna"
        Which class is treated as positive for binary AUC metrics.
    keep_unknown : bool, default False
        If *True*, the label "unknown" (any case) is kept in the evaluation. Otherwise it is ignored.
    """

    # -------- 1) sanitise missing values ---------------------------------
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)

    if y_score is not None:
        y_score = np.asarray(y_score, dtype=float)
        mask = (~pd.isna(y_true)) & (~pd.isna(y_pred)) & (~np.isnan(y_score))
        y_score = y_score[mask]
    else:
        mask = (~pd.isna(y_true)) & (~pd.isna(y_pred))

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    # -------- 1b) optionally drop "unknown" -------------------------------
    if not keep_unknown:
        str_true = np.char.lower(y_true.astype(str))
        str_pred = np.char.lower(y_pred.astype(str))
        unk_mask = (str_true != "unknown") & (str_pred != "unknown")

        y_true = y_true[unk_mask]
        y_pred = y_pred[unk_mask]
        if y_score is not None:
            y_score = y_score[unk_mask]

    # -------- 2) core classification metrics -----------------------------
    labels_eval = _ordered_labels(y_true, y_pred, keep_unknown=keep_unknown)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels_eval,
        digits=3,
        zero_division=0,
        output_dict=True,
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels_eval)

    f1_per_class = {
        f"f1_{lbl}": report.get(lbl, {}).get("f1-score", np.nan) for lbl in DEFAULT_LABELS
    }

    metrics: dict[str, float | np.ndarray | Mapping] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        **f1_per_class,
        "mcc": matthews_corrcoef(y_true, y_pred),
        "confusion_matrix": cm,
        "classification_report": report,
    }

    # -------- 3) PR-AUC & ROC-AUC  (if score is supplied) ---------------
    if y_score is not None and (y_true == pos_label).any():
        y_true_bin = (y_true == pos_label).astype(int)
        precision, recall, _ = precision_recall_curve(y_true_bin, y_score)
        metrics["pr_auc"] = auc(recall, precision)
        try:
            metrics["roc_auc"] = roc_auc_score(y_true_bin, y_score)
        except ValueError:
            metrics["roc_auc"] = np.nan

    return metrics


# =====================================================================
#  PATIENT-LEVEL METRICS
# =====================================================================


def patient_metrics(y_true, y_pred, patient_ids, timestamps, abnormal_labels):
    """
    y_true / y_pred : 1-D array s beat labelmi
    patient_ids     : 1-D array rovnakej dĺžky
    timestamps      : 1-D array epoch-time v sekundách
    abnormal_labels : set labelov považovaných za 'abnormal'
    """
    df = pd.DataFrame(
        {"y_true": y_true, "y_pred": y_pred,
         "id": patient_ids, "t": timestamps}
    )

    # Sensitivita na pacienta
    sens_num, sens_den = 0, 0
    for pid, grp in df.groupby("id"):
        has_abn_true = (grp.y_true.isin(abnormal_labels)).any()
        if not has_abn_true:
            continue                      # zdravý pacient → nepočíta sa
        sens_den += 1
        if (grp.y_pred.isin(abnormal_labels)).any():
            sens_num += 1
    patient_sensitivity = sens_num / sens_den if sens_den else float("nan")

    # False alarms / hod
    fa_rates = []
    for pid, grp in df.groupby("id"):
        false_alarms = ((~grp.y_true.isin(abnormal_labels)) & (grp.y_pred.isin(abnormal_labels))).sum()
        hours = (grp.t.max() - grp.t.min()) / 3600
        if hours:  # hours == 0 pre prázdny záznam
            fa_rates.append(false_alarms / hours)

    false_alarms_per_hour = np.mean(fa_rates) if fa_rates else float("nan")

    return patient_sensitivity, false_alarms_per_hour


# =====================================================================
#  Quick self-test
# =====================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    yT = rng.choice(DEFAULT_LABELS, size=1_000, p=[0.85, 0.14, 0.01])
    yP = rng.choice(DEFAULT_LABELS, size=1_000, p=[0.90, 0.09, 0.01])
    centroid = rng.random(1_000)

    print("Beat-level test:\n", beat_level_metrics(yT, yP, y_score=centroid))
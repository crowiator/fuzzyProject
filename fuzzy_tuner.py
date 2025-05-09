"""
fuzzy_tuner.py – end‑to‑end utility that learns optimal fuzzy membership
functions from labelled beat‑level CSV files and exports the parameters.

Workflow
========
1. Put your *training* CSVs (one file per record/patient)
   in a folder, e.g.  data/train_csv/ (must contain columns
   HR_bpm, QRS_ms, T_amp, TrueLabel, FuzzyLabel, Centroid, R_index).
2. From a Python session / notebook run:

        from fuzzy_tuner import tune_memberships, load_dataset

        train_df = load_dataset("data/train_csv")
        best_params = tune_memberships(train_df,
                                       n_trials=300,
                                       target_sens=0.9)

   This will run Optuna Bayesian optimisation and return a dict
   with the best MF break‑points.
3. Write the parameters to JSON:

        import json, pathlib
        pathlib.Path("best_mf.json").write_text(json.dumps(best_params, indent=2))

4. Evaluate on *test* CSV folder:

        from fuzzy_tuner import evaluate_dataset, FuzzyClassifier, set_mf

        test_df = load_dataset("data/test_csv")
        clf = FuzzyClassifier()
        set_mf(clf, best_params)   # install learned MFs
        sens, fa = evaluate_dataset(test_df, clf)
        print("patient‑sens =", sens, "FA/h =", fa)

You can re‑run `tune_memberships` as many times as you wish; the study is
saved to `optuna_study.db` (SQLite) so it resumes automatically.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
from optuna.storages import RDBStorage

from classifiers.fuzzy_classifier import FuzzyClassifier

FS = 360  # sampling rate (Hz) – adjust if needed
ABN = {"zavazna"}

# ---------------------------------------------------------------------------
# Helper – load all CSVs in directory
# ---------------------------------------------------------------------------

def load_dataset(folder: str | Path) -> pd.DataFrame:
    """Load all *.csv from *folder* (recursively) and return concatenated DF."""
    folder = Path(folder)
    rows = []
    for p in folder.rglob("*.csv"):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if "Record" in df.columns:
            pid = str(df["Record"].iloc[0])
        else:
            pid = p.stem.split("_")[1]  # mitdb_104_...
        df["PatientID"] = pid
        rows.append(df[
            [
                "PatientID",
                "R_index",
                "HR_bpm",
                "QRS_ms",
                "T_amp",
                "TrueLabel",
            ]
        ])
    full = pd.concat(rows, ignore_index=True)
    full["t_sec"] = full["R_index"] / FS
    return full

# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def patient_metrics(df: pd.DataFrame, clf: FuzzyClassifier) -> Tuple[float, float]:
    """Return (patient_sensitivity, FA/h) for whole DF.
    Robust voči tomu, že `clf.predict()` môže v okrajových situáciách
    vrátiť tuple ``(centroid, label, μ)`` namiesto dict.
    """
    preds: List[str] = []
    for hr, qrs, twa in df[["HR_bpm", "QRS_ms", "T_amp"]].itertuples(index=False, name=None):
        res = clf.predict(hr, qrs, twa)
        if isinstance(res, dict):
            preds.append(res.get("label", "invalid"))
        else:  # fallback – tuple (centroid, label, …)
            try:
                preds.append(res[1])  # label
            except Exception:
                preds.append("invalid")

    df = df.copy()
    df["Pred"] = preds

    sens_list, fa_list = [], []
    for pid, g in df.groupby("PatientID"):
        has_abn = g["TrueLabel"].isin(ABN).any()
        if has_abn:
            sens_list.append(1.0 if g["Pred"].isin(ABN).any() else 0.0)
        fa_mask = (~g["TrueLabel"].isin(ABN)) & (g["Pred"].isin(ABN))
        hours = (g["t_sec"].max() - g["t_sec"].min()) / 3600
        fa_list.append(fa_mask.sum() / hours if hours else 0.0)

    sens = float(np.mean(sens_list)) if sens_list else math.nan
    fa = float(np.mean(fa_list)) if fa_list else math.nan
    return sens, fa

# ---------------------------------------------------------------------------
# Parametrisation of membership functions
# ---------------------------------------------------------------------------

def set_mf(clf: FuzzyClassifier, p: Dict[str, float]) -> None:
    """Overwrite MF nodes with *triangular* arrays matching universe length.
    Each node triple (a, b, c) is passed to ``fuzz.trimf`` so that the shape
    of ``.mf`` equals len(variable.universe) – avoiding the NumPy interp
    error you hit."""
    import skfuzzy as fuzz

    # HR -----------------------------------------------------------
    clf.hr["nizka"].mf     = fuzz.trimf(clf.hr.universe,     [p["HR_L1"], p["HR_L2"], p["HR_L3"]])
    clf.hr["normalna"].mf  = fuzz.trimf(clf.hr.universe,     [p["HR_N1"], p["HR_N2"], p["HR_N3"]])
    clf.hr["vysoka"].mf    = fuzz.trimf(clf.hr.universe,     [p["HR_H1"], p["HR_H2"], p["HR_H3"]])

    # QRS ----------------------------------------------------------
    clf.qrs["kratky"].mf    = fuzz.trimf(clf.qrs.universe,    [p["QRS_K1"], p["QRS_K2"], p["QRS_K3"]])
    clf.qrs["normalny"].mf  = fuzz.trimf(clf.qrs.universe,    [p["QRS_N1"], p["QRS_N2"], p["QRS_N3"]])
    clf.qrs["dlhy"].mf      = fuzz.trimf(clf.qrs.universe,    [p["QRS_D1"], p["QRS_D2"], p["QRS_D3"]])

    # TWA – nízka a normálna ostávajú, ladíme len "vysoka" ---------
    high = p["TWA_high"]
    clf.twa["vysoka"].mf   = fuzz.trimf(clf.twa.universe,    [0.8*high, 0.9*high, high])

    # Výstup Arrhythmia -------------------------------------------
    clf.arr["normalna"].mf = fuzz.trimf(clf.arr.universe,    [0.0, p["ARR_N2"], p["ARR_N3"]])
    clf.arr["mierna"].mf   = fuzz.trimf(clf.arr.universe,    [p["ARR_M1"], p["ARR_M2"], p["ARR_M3"]])
    clf.arr["zavazna"].mf  = fuzz.trimf(clf.arr.universe,    [p["ARR_S1"], p["ARR_S2"], 1.0])

# ---------------------------------------------------------------------------
# Optimisation wrapper
# ---------------------------------------------------------------------------

def tune_memberships(df: pd.DataFrame, *, n_trials: int = 300, target_sens: float = 0.9) -> Dict[str, float]:
    """Run Optuna to tune MF nodes on *df* and return best param dict."""

    storage = RDBStorage(url="sqlite:///optuna_study.db")
    study = optuna.create_study(study_name="fuzzy_mf", direction="minimize", storage=storage, load_if_exists=True)

    def suggest_nodes(trial: optuna.Trial) -> Dict[str, float]:
        p = {}
        # HR nizka  (40–60‑80)
        p["HR_L1"] = trial.suggest_float("HR_L1", 30, 50)
        p["HR_L2"] = trial.suggest_float("HR_L2", p["HR_L1"] + 5, 60)
        p["HR_L3"] = trial.suggest_float("HR_L3", p["HR_L2"] + 5, 70)
        # HR normalna
        p["HR_N1"] = trial.suggest_float("HR_N1", 55, 70)
        p["HR_N2"] = trial.suggest_float("HR_N2", p["HR_N1"], 85)
        p["HR_N3"] = trial.suggest_float("HR_N3", p["HR_N2"], 100)
        # HR vysoka
        p["HR_H1"] = trial.suggest_float("HR_H1", 90, 110)
        p["HR_H2"] = trial.suggest_float("HR_H2", p["HR_H1"], 140)
        p["HR_H3"] = trial.suggest_float("HR_H3", p["HR_H2"], 220)
        # QRS kratky/normalny/dlhy
        p["QRS_K1"] = trial.suggest_float("QRS_K1", 40, 60)
        p["QRS_K2"] = trial.suggest_float("QRS_K2", p["QRS_K1"], 75)
        p["QRS_K3"] = trial.suggest_float("QRS_K3", p["QRS_K2"], 90)
        p["QRS_N1"] = trial.suggest_float("QRS_N1", 70, 90)
        p["QRS_N2"] = trial.suggest_float("QRS_N2", p["QRS_N1"], 120)
        p["QRS_N3"] = trial.suggest_float("QRS_N3", p["QRS_N2"], 160)
        p["QRS_D1"] = trial.suggest_float("QRS_D1", 110, 160)
        p["QRS_D2"] = trial.suggest_float("QRS_D2", p["QRS_D1"], 250)
        p["QRS_D3"] = trial.suggest_float("QRS_D3", p["QRS_D2"], 400)
        # TWA high
        p["TWA_high"] = trial.suggest_float("TWA_high", 0.6, 1.2)
        # Output arr nodes
        p["ARR_N2"] = trial.suggest_float("ARR_N2", 0.15, 0.30)
        p["ARR_N3"] = trial.suggest_float("ARR_N3", p["ARR_N2"], 0.45)
        p["ARR_M1"] = trial.suggest_float("ARR_M1", 0.45, 0.60)
        p["ARR_M2"] = trial.suggest_float("ARR_M2", p["ARR_M1"], 0.75)
        p["ARR_M3"] = trial.suggest_float("ARR_M3", p["ARR_M2"], 0.85)
        p["ARR_S1"] = trial.suggest_float("ARR_S1", 0.70, 0.80)
        p["ARR_S2"] = trial.suggest_float("ARR_S2", p["ARR_S1"], 0.95)
        return p

    def objective(trial: optuna.Trial) -> float:
        params = suggest_nodes(trial)
        clf = FuzzyClassifier()
        set_mf(clf, params)
        sens, fa = patient_metrics(df, clf)
        # store for inspection
        trial.set_user_attr("sens", sens)
        trial.set_user_attr("fa", fa)
        # loss: FA/h + big penalty if sens < target
        penalty = 100 * max(0, target_sens - (sens if not math.isnan(sens) else 0))
        return fa + penalty

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = study.best_trial
    best_params = study.best_params
    print(
        f"Best trial: FA/h={best.user_attrs['fa']:.2f}, "
        f"patient-sens={best.user_attrs['sens']:.3f}")
    return best_params

# ---------------------------------------------------------------------------
# Simple evaluation on a new dataset
# ---------------------------------------------------------------------------

def evaluate_dataset(df: pd.DataFrame, clf: FuzzyClassifier) -> Tuple[float, float]:
    """Return (patient-sensitivity, FA/h) for *df* using current *clf*."""
    return patient_metrics(df, clf)


train_df = load_dataset("mitdb_fuzzy_results")
best = tune_memberships(train_df, n_trials=300, target_sens=0.9)
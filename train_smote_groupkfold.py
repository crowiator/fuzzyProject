
"""
train_smote_groupkfold.py
=========================
Pipeline that combines:

– traditional_models.py     (KNN, Decision‑Tree, Random‑Forest, SVM helpers)
– SMOTE                     (imbalanced‑learn oversampling)
– StratifiedGroupKFold      (patient‑wise CV with preserved class ratio)

Usage
-----
$ python train_smote_groupkfold.py --csv_dir data/feat --models rf knn     --n_splits 5 --random_state 42

Requires:
    pip install scikit-learn imbalanced-learn tabulate
"""

import argparse
import glob
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, f1_score, matthews_corrcoef
from sklearn.model_selection import GroupKFold

# Try stratified groups if available (sklearn >= 1.3)
try:
    from sklearn.model_selection import StratifiedGroupKFold
    SGK = StratifiedGroupKFold
except ImportError:  # fallback
    print("⚠️  StratifiedGroupKFold not available, using plain GroupKFold.")
    SGK = None

from imblearn.over_sampling import SMOTE

# --- local project modules ---------------------------------------------------
from classifiers.traditional_models import (
    train_knn,
    train_decision_tree,
    train_random_forest,
    train_svm,
    evaluate_model,
)
from preprocessing.annotation_mapping import ANNOTATION_TO_FUZZY, EXCLUDED_ANN

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def physiologic_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Optionally remove extreme values."""
    mask = (
        df["HR_bpm"].between(25, 200, inclusive="both")
        & df["QRS_ms"].between(40, 300, inclusive="both")
        & (df["T_amp"].abs() < 2.5)
    )
    return df[mask]


def load_feature_frames(csv_dir: Path) -> pd.DataFrame:
    """Load all csv files produced by extract_features()."""
    frames = []
    for csv_path in sorted(Path(csv_dir).glob("ecg_*_features_full.csv")):
        frames.append(pd.read_csv(csv_path))
    if not frames:
        raise RuntimeError(f"No feature files found in {csv_dir}")
    df = pd.concat(frames, ignore_index=True)
    return df


def build_model(name: str, X_train, y_train):
    """Dispatch to training helpers inside traditional_models.py."""
    if name == "knn":
        return train_knn(X_train, y_train, metric="minkowski", n_neighbors=5, weights="uniform")
    if name == "dt":
        return train_decision_tree(
            X_train, y_train,
            criterion="gini",
            max_depth=None,
            max_features="sqrt",
            min_samples_leaf=1,
            min_samples_split=2,
        )
    if name == "rf":
        return train_random_forest(
            X_train, y_train,
            criterion="gini",
            max_depth=None,
            max_features="sqrt",
            min_samples_leaf=1,
            min_samples_split=2,
            n_estimators=300,
        )
    if name == "svm":
        return train_svm(X_train, y_train, kernel="rbf", C=1.0, gamma="scale")
    raise ValueError(f"Unknown model name: {name}")

# --------------------------------------------------------------------------- #

def main(cfg):
    df = load_feature_frames(cfg.csv_dir)
    print(f"🔢 Loaded {len(df):,} beats from {cfg.csv_dir}")

    # map annotation -> fuzzy label
    df = df[~df["Annotation"].isin(EXCLUDED_ANN)]
    df["FuzzyLabel"] = df["Annotation"].map(ANNOTATION_TO_FUZZY)
    df = df.dropna(subset=["FuzzyLabel"])  # drop unknown labels

    df = physiologic_filter(df)
    print(f"✅ After physio filter: {len(df):,} beats")

    X = df[["HR_bpm", "QRS_ms", "T_amp"]].values.astype(float)
    y = df["FuzzyLabel"].values
    groups = df["Record"].values

    cv_class = SGK if (SGK is not None and cfg.stratify) else GroupKFold
    cv = cv_class(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)

    overall_scores = defaultdict(list)

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        print(f"📂 Fold {fold+1}/{cfg.n_splits}")
        X_train, y_train = X[train_idx], y[train_idx]
        X_test,  y_test  = X[test_idx],  y[test_idx]

        # oversample training set only
        smote = SMOTE(k_neighbors=3, random_state=cfg.random_state)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f"   ↳ After SMOTE: {len(y_train)} → {len(y_train_res)} samples")

        for name in cfg.models:
            model = build_model(name, X_train_res, y_train_res)
            evaluate_model(model, X_test, y_test,
                           save_path=None, model_name=name)

            # collect scores
            y_pred = model.predict(X_test)
            overall_scores[f"{name}_f1"].append(
                f1_score(y_test, y_pred, average="macro")
            )
            overall_scores[f"{name}_mcc"].append(
                matthews_corrcoef(y_test, y_pred)
            )

    # summary
    print("\n=== Mean CV scores ===")
    for key, vals in overall_scores.items():
        print(f"{key:12s}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    # save raw scores
    out_json = Path(cfg.output).with_suffix(".json")
    out_json.write_text(json.dumps(overall_scores, indent=2))
    print(f"📄 Saved fold scores to {out_json}")

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train MIT-BIH beat classifier with SMOTE + GroupKFold.")
    p.add_argument("--csv_dir", type=Path, required=True, help="Directory with ecg_*_features_full.csv")
    p.add_argument("--models", nargs="+", default=["rf", "knn"], choices=["rf", "knn", "dt", "svm"],
                   help="Models to train (by nickname).")
    p.add_argument("--n_splits", type=int, default=5, help="Number of CV folds.")
    p.add_argument("--random_state", type=int, default=42, help="Random seed.")
    p.add_argument("--stratify", action="store_true", help="Use StratifiedGroupKFold if available.")
    p.add_argument("--output", default="cv_scores", help="Basename for JSON score output.")
    args = p.parse_args()
    main(args)

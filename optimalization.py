import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import os
import csv
from classifiers.fuzzy_classifier import init_fuzzy_system

from classifiers.fuzzy_classifier import predict_arrhythmia_centroid
from preprocessing.fuzzy_feature_loader import load_or_extract_fuzzy_features
from classifiers.traditional_models import (
    evaluate_model,
)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


def save_best_params(model_name, params, hybrid=False):
    os.makedirs("results", exist_ok=True)
    file_path = "results/best_params.csv"
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write header if file is new
        if not file_exists:
            writer.writerow(["Model", "Hybrid", "Best Parameters"])

        writer.writerow([model_name, hybrid, params])


def is_valid_for_fuzzy(hr, qrs, twa):
    return (40 <= hr <= 120) and (50 <= qrs <= 120) and (0.0 <= twa <= 0.6)


def train_decision_tree(X_train, y_train, hybrid=False):
    param_grid = {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': [None, 10, 15, 20, 25, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None, 'sqrt', 'log2']
    }

    dt_grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )

    dt_grid.fit(X_train, y_train)
    print(" Decision Tree best params:", dt_grid.best_params_)
    save_best_params("Decision Tree", dt_grid.best_params_, hybrid)
    return dt_grid.best_estimator_


def train_random_forest(X_train, y_train, hybrid=False):
    param_grid = {
        'n_estimators': [100, 150, 200],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )

    rf_grid.fit(X_train, y_train)
    print(" Random Forest best params:", rf_grid.best_params_)
    save_best_params("Random Forest", rf_grid.best_params_, hybrid)
    return rf_grid.best_estimator_


def train_svm(X_train, y_train, hybrid=False):
    param_grid = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.01, 0.1],
        'degree': [2, 3]
    }

    svm_grid = GridSearchCV(
        SVC(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )

    svm_grid.fit(X_train, y_train)
    print(" SVM best params:", svm_grid.best_params_)
    save_best_params("SVM", svm_grid.best_params_, hybrid)
    return svm_grid.best_estimator_


def train_knn(X_train, y_train, hybrid=False):
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }

    knn_grid = GridSearchCV(
        KNeighborsClassifier(),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )

    knn_grid.fit(X_train, y_train)
    print(" kNN best params:", knn_grid.best_params_)
    save_best_params("kNN", knn_grid.best_params_, hybrid)
    return knn_grid.best_estimator_


def run_classical_algorithm(df):
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CV_FOLDS = 5

    X = df[["HR", "QRS_interval", "T_wave"]].values
    y = df["Label"].values

    X_resampled, y_resampled = SMOTE(random_state=RANDOM_STATE).fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=TEST_SIZE, stratify=y_resampled, random_state=RANDOM_STATE
    )

    print("\n Decision Tree:")
    dt_model = train_decision_tree(X_train, y_train, hybrid=False)
    evaluate_model(dt_model, X_test, y_test, save_path="results/reports/dt_predictions.csv")
    dt_scores = cross_val_score(dt_model, X_resampled, y_resampled, cv=CV_FOLDS)
    print(f" Decision Tree cross-validation accuracy: {np.mean(dt_scores):.4f}")

    print("\n Random Forest:")
    rf_model = train_random_forest(X_train, y_train, hybrid=False)
    evaluate_model(rf_model, X_test, y_test, save_path="results/reports/rf_predictions.csv")
    rf_scores = cross_val_score(rf_model, X_resampled, y_resampled, cv=CV_FOLDS)
    print(f" Random Forest cross-validation accuracy: {np.mean(rf_scores):.4f}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)
    X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(
        X_scaled, y_resampled, test_size=TEST_SIZE, stratify=y_resampled, random_state=RANDOM_STATE
    )

    print("\n Support Vector Machine (SVM):")
    svm_model = train_svm(X_train_svm, y_train_svm, hybrid=False)
    evaluate_model(svm_model, X_test_svm, y_test_svm, save_path="results/reports/svm_predictions.csv")
    svm_scores = cross_val_score(svm_model, X_scaled, y_resampled, cv=CV_FOLDS)
    print(f" SVM cross-validation accuracy: {np.mean(svm_scores):.4f}")

    print("\n k-Nearest Neighbors (kNN):")
    knn_model = train_knn(X_train, y_train, hybrid=False)
    evaluate_model(knn_model, X_test, y_test, save_path="results/reports/knn_predictions.csv")
    knn_scores = cross_val_score(knn_model, X_resampled, y_resampled, cv=CV_FOLDS)
    print(f" kNN cross-validation accuracy: {np.mean(knn_scores):.4f}")


def run_hybrid_models(df):
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CV_FOLDS = 5

    def extract_fuzzy_features(row):
        score, _, membership = predict_arrhythmia_centroid(
            row["HR"], row["QRS_interval"], row["T_wave"], return_membership=True
        )
        return pd.Series({
            "Fuzzy_Score": score,
            "Fuzzy_\u03bc_Normal": membership.get("normal", 0),
            "Fuzzy_\u03bc_Moderate": membership.get("moderate", 0),
            "Fuzzy_\u03bc_Severe": membership.get("severe", 0),
        })

    df_fuzzy = df.apply(extract_fuzzy_features, axis=1)
    df = pd.concat([df, df_fuzzy], axis=1)

    X = df[[
        "HR", "QRS_interval", "T_wave",
        "Fuzzy_Score", "Fuzzy_\u03bc_Normal", "Fuzzy_\u03bc_Moderate", "Fuzzy_\u03bc_Severe"
    ]].values
    y = df["Label"].values

    X_resampled, y_resampled = SMOTE(random_state=RANDOM_STATE).fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=TEST_SIZE, stratify=y_resampled, random_state=RANDOM_STATE
    )

    print("\n Hybrid Decision Tree:")
    dt_model = train_decision_tree(X_train, y_train, hybrid=True)
    evaluate_model(dt_model, X_test, y_test, save_path="results/reports/hybrid_dt_predictions.csv")
    dt_scores = cross_val_score(dt_model, X_resampled, y_resampled, cv=CV_FOLDS)
    print(f" Hybrid Decision Tree cross-validation accuracy: {np.mean(dt_scores):.4f}")

    print("\n Hybrid Random Forest:")
    rf_model = train_random_forest(X_train, y_train, hybrid=True)
    evaluate_model(rf_model, X_test, y_test, save_path="results/reports/hybrid_rf_predictions.csv")
    rf_scores = cross_val_score(rf_model, X_resampled, y_resampled, cv=CV_FOLDS)
    print(f" Hybrid Random Forest cross-validation accuracy: {np.mean(rf_scores):.4f}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)
    X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(
        X_scaled, y_resampled, test_size=TEST_SIZE, stratify=y_resampled, random_state=RANDOM_STATE
    )

    print("\n Hybrid Support Vector Machine (SVM):")
    svm_model = train_svm(X_train_svm, y_train_svm, hybrid=True)
    evaluate_model(svm_model, X_test_svm, y_test_svm, save_path="results/reports/hybrid_svm_predictions.csv")
    svm_scores = cross_val_score(svm_model, X_scaled, y_resampled, cv=CV_FOLDS)
    print(f"Hybrid SVM cross-validation accuracy: {np.mean(svm_scores):.4f}")

    print("\n🧐 Hybrid k-Nearest Neighbors (kNN):")
    knn_model = train_knn(X_train, y_train, hybrid=True)
    evaluate_model(knn_model, X_test, y_test, save_path="results/reports/hybrid_knn_predictions.csv")
    knn_scores = cross_val_score(knn_model, X_resampled, y_resampled, cv=CV_FOLDS)
    print(f"🔄 Hybrid kNN cross-validation accuracy: {np.mean(knn_scores):.4f}")


if __name__ == "__main__":
    init_fuzzy_system()
    features, labels, _ = load_or_extract_fuzzy_features()
    df = pd.DataFrame(features)
    df["Label"] = labels
    df["is_valid"] = df.apply(lambda row: is_valid_for_fuzzy(row["HR"], row["QRS_interval"], row["T_wave"]), axis=1)
    df_valid = df[df["is_valid"]].drop(columns=["is_valid"])

    print(f"✅ Počet validných vzoriek pre fuzzy: {len(df_valid)}")

    run_classical_algorithm(df_valid)
    run_hybrid_models(df_valid)

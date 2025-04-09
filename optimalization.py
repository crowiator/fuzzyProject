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
from preprocessing.fuzzy_feature_loader import load_or_extract_fuzzy_features
from classifiers.traditional_models import evaluate_model
from classifiers.fuzzy_classifier import FuzzyClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

def save_best_params(model_name, params, hybrid=False):
    os.makedirs("results", exist_ok=True)
    file_path = "results/best_params.csv"
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Model", "Hybrid", "Best Parameters"])
        writer.writerow([model_name, hybrid, params])

def is_valid_for_fuzzy(hr, qrs, twa):
    return (40 <= hr <= 120) and (50 <= qrs <= 120) and (0.0 <= twa <= 0.6)

def train_with_grid_search(estimator, param_grid, X, y, model_name, hybrid):
    grid = GridSearchCV(estimator, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
    grid.fit(X, y)
    print(f" {model_name} best params: {grid.best_params_}")
    save_best_params(model_name, grid.best_params_, hybrid)
    return grid.best_estimator_

def run_classical_algorithm(df):
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CV_FOLDS = 5

    X = df[["HR", "QRS_interval", "T_wave"]].values
    y = df["Label"].values

    X_resampled, y_resampled = SMOTE(random_state=RANDOM_STATE).fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=TEST_SIZE, stratify=y_resampled, random_state=RANDOM_STATE)

    print("\n Decision Tree:")
    dt_model = train_with_grid_search(DecisionTreeClassifier(random_state=42), {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': [None, 10, 15, 20, 25, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None, 'sqrt', 'log2']
    }, X_train, y_train, "Decision Tree", False)
    evaluate_model(dt_model, X_test, y_test, save_path="results/reports/dt_predictions.csv")
    print(f" Decision Tree cross-validation accuracy: {np.mean(cross_val_score(dt_model, X_resampled, y_resampled, cv=CV_FOLDS)):.4f}")

    print("\n Random Forest:")
    rf_model = train_with_grid_search(RandomForestClassifier(random_state=42), {
        'n_estimators': [100, 150, 200],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }, X_train, y_train, "Random Forest", False)
    evaluate_model(rf_model, X_test, y_test, save_path="results/reports/rf_predictions.csv")
    print(f" Random Forest cross-validation accuracy: {np.mean(cross_val_score(rf_model, X_resampled, y_resampled, cv=CV_FOLDS)):.4f}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)
    X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X_scaled, y_resampled, test_size=TEST_SIZE, stratify=y_resampled, random_state=RANDOM_STATE)

    print("\n Support Vector Machine (SVM):")
    svm_model = train_with_grid_search(SVC(random_state=42), {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.01, 0.1],
        'degree': [2, 3]
    }, X_train_svm, y_train_svm, "SVM", False)
    evaluate_model(svm_model, X_test_svm, y_test_svm, save_path="results/reports/svm_predictions.csv")
    print(f" SVM cross-validation accuracy: {np.mean(cross_val_score(svm_model, X_scaled, y_resampled, cv=CV_FOLDS)):.4f}")

    print("\n k-Nearest Neighbors (kNN):")
    knn_model = train_with_grid_search(KNeighborsClassifier(), {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }, X_train, y_train, "kNN", False)
    evaluate_model(knn_model, X_test, y_test, save_path="results/reports/knn_predictions.csv")
    print(f" kNN cross-validation accuracy: {np.mean(cross_val_score(knn_model, X_resampled, y_resampled, cv=CV_FOLDS)):.4f}")

def run_hybrid_models(df):
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CV_FOLDS = 5
    fuzzy = FuzzyClassifier()

    def extract_fuzzy_features(row):
        score, _, membership = fuzzy.predict(row["HR"], row["QRS_interval"], row["T_wave"])
        return pd.Series({
            "Fuzzy_Score": score,
            "Fuzzy_μ_Normal": membership.get("normal", 0),
            "Fuzzy_μ_Moderate": membership.get("moderate", 0),
            "Fuzzy_μ_Severe": membership.get("severe", 0),
        })

    df_fuzzy = df.apply(extract_fuzzy_features, axis=1)
    df = pd.concat([df, df_fuzzy], axis=1)
    df = df.dropna()
    X = df[["HR", "QRS_interval", "T_wave", "Fuzzy_Score", "Fuzzy_μ_Normal", "Fuzzy_μ_Moderate", "Fuzzy_μ_Severe"]].values
    y = df["Label"].values

    X_resampled, y_resampled = SMOTE(random_state=RANDOM_STATE).fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=TEST_SIZE, stratify=y_resampled, random_state=RANDOM_STATE)

    print("\n Hybrid Decision Tree:")
    dt_model = train_with_grid_search(DecisionTreeClassifier(random_state=42), {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': [None, 10, 15, 20, 25, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None, 'sqrt', 'log2']
    }, X_train, y_train, "Decision Tree", True)
    evaluate_model(dt_model, X_test, y_test, save_path="results/reports/hybrid_dt_predictions.csv")
    print(f" Hybrid Decision Tree cross-validation accuracy: {np.mean(cross_val_score(dt_model, X_resampled, y_resampled, cv=CV_FOLDS)):.4f}")

    print("\n Hybrid Random Forest:")
    rf_model = train_with_grid_search(RandomForestClassifier(random_state=42), {
        'n_estimators': [100, 150, 200],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }, X_train, y_train, "Random Forest", True)
    evaluate_model(rf_model, X_test, y_test, save_path="results/reports/hybrid_rf_predictions.csv")
    print(f" Hybrid Random Forest cross-validation accuracy: {np.mean(cross_val_score(rf_model, X_resampled, y_resampled, cv=CV_FOLDS)):.4f}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)
    X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X_scaled, y_resampled, test_size=TEST_SIZE, stratify=y_resampled, random_state=RANDOM_STATE)

    print("\n Hybrid Support Vector Machine (SVM):")
    svm_model = train_with_grid_search(SVC(random_state=42), {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.01, 0.1],
        'degree': [2, 3]
    }, X_train_svm, y_train_svm, "SVM", True)
    evaluate_model(svm_model, X_test_svm, y_test_svm, save_path="results/reports/hybrid_svm_predictions.csv")
    print(f"Hybrid SVM cross-validation accuracy: {np.mean(cross_val_score(svm_model, X_scaled, y_resampled, cv=CV_FOLDS)):.4f}")

    print("\n Hybrid k-Nearest Neighbors (kNN):")
    knn_model = train_with_grid_search(KNeighborsClassifier(), {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }, X_train, y_train, "kNN", True)
    evaluate_model(knn_model, X_test, y_test, save_path="results/reports/hybrid_knn_predictions.csv")
    print(f" Hybrid kNN cross-validation accuracy: {np.mean(cross_val_score(knn_model, X_resampled, y_resampled, cv=CV_FOLDS)):.4f}")

if __name__ == "__main__":
    fuzzy = FuzzyClassifier()
    features, labels, _ = load_or_extract_fuzzy_features()
    df = pd.DataFrame(features)
    df["Label"] = labels
    df["is_valid"] = df.apply(lambda row: fuzzy.is_valid(row["HR"], row["QRS_interval"], row["T_wave"]), axis=1)
    df_valid = df[df["is_valid"]].drop(columns=["is_valid"])

    print(f"Number of valid : {len(df_valid)}")
    #run_classical_algorithm(df_valid)
    run_hybrid_models(df_valid)

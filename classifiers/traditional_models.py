# classifiers/traditional_models.py
# Importy potrebných knižníc pre klasifikátory a vyhodnotenie modelov
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import pandas as pd

"""
Súbor traditional_models.py obsahuje funkcie, 
ktoré umožňujú trénovanie a vyhodnotenie klasických metód strojového učenia
(K-Nearest Neighbors, Decision Tree, Random Forest, Support Vector Machine)
na klasifikačné úlohy. 
Každá funkcia pripraví a vráti natrénovaný klasifikačný model,
pričom sú parametre trénovania flexibilné a možno ich meniť podľa potreby. 
Súčasťou súboru je aj funkcia na vyhodnotenie výsledkov klasifikácie.
"""


def train_knn(X_train, y_train, metric, n_neighbors, weights):
    # Inicializácia modelu KNN s definovanými parametrami (metrika vzdialenosti, počet susedov, váhovanie)
    # model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model = KNeighborsClassifier(metric=metric, n_neighbors=n_neighbors, weights=weights)
    # Trénovanie modelu na trénovacích dátach
    model.fit(X_train, y_train)
    # Vrátenie natrénovaného modelu
    return model


# Funkcia na trénovanie klasifikátora typu Decision Tree (rozhodovací strom)
def train_decision_tree(X_train, y_train, criterion, max_depth, max_features, min_samples_leaf, min_samples_split):
    # Inicializácia modelu Decision Tree s parametrami (kritérium delenia, maximálna hĺbka stromu, atď.)
    # model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, max_features=max_features,
                                   min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                                   random_state=42)
    # Trénovanie modelu na trénovacích dátach
    model.fit(X_train, y_train)
    # Vrátenie natrénovaného modelu
    return model


# Funkcia na trénovanie klasifikátora typu Random Forest
def train_random_forest(X_train, y_train, criterion, max_depth, max_features, min_samples_leaf, min_samples_split,
                        n_estimators):
    # Inicializácia modelu Random Forest s definovanými parametrami
    # (počet stromov, maximálna hĺbka, kritérium delenia, atď.)
    # model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model = RandomForestClassifier(criterion=criterion, max_depth=max_depth, max_features=max_features,
                                   min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                                   n_estimators=n_estimators, random_state=42)
    # Trénovanie modelu na trénovacích dátach
    model.fit(X_train, y_train)

    return model


# Funkcia na trénovanie klasifikátora Support Vector Machine (SVM)
def train_svm(X_train, y_train, C, degree, gamma, kernel):
    # Inicializácia modelu SVM s definovanými parametrami
    # (regularizácia C, stupeň pre polynómiálny kernel, gamma a typ kernelu)
    # model = SVC(kernel=kernel, random_state=random_state)
    model = SVC(C=C, degree=degree, gamma=gamma, kernel=kernel, random_state=42)
    model.fit(X_train, y_train)
    return model


# Funkcia na vyhodnotenie natrénovaného modelu na testovacích dátach
def evaluate_model(model, X_test, y_test, save_path=None, model_name="model", cv_scores=None):


    y_pred = model.predict(X_test)

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("\nClassification Report:")
    report = classification_report(y_test, y_pred)
    print(report)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    if save_path:
        base_path = os.path.splitext(save_path)[0]

        # Save predictions
        df_results = pd.DataFrame(X_test, columns=[f"Feature_{i + 1}" for i in range(X_test.shape[1])])
        df_results["True_Label"] = y_test
        df_results["Predicted_Label"] = y_pred
        csv_path = f"{base_path}_{model_name}.csv"
        df_results.to_csv(csv_path, index=False)
        print(f"Predictions saved to: {csv_path}")

        # Save confusion matrix
        class_names = ["Normal", "Moderate", "Severe"]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap="Blues")
        plt.title(f"Confusion Matrix – {model_name}")
        cm_path = f"{base_path}_{model_name}_confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        print(f"Confusion Matrix saved to: {cm_path}")

        # Save classification report
        report_path = f"{base_path}_{model_name}_report.txt"
        with open(report_path, "w") as f:
            f.write(f"Classification Report – {model_name}\n")
            f.write("=" * 40 + "\n\n")
            f.write(report)
            f.write(f"\nAccuracy: {accuracy:.4f}\n")

            if cv_scores is not None:
                mean_cv = np.mean(cv_scores)
                std_cv = np.std(cv_scores)
                f.write("\nCross-validation results:\n")
                f.write(f"- Mean CV Accuracy: {mean_cv:.4f}\n")
                f.write(f"- Std CV Accuracy: {std_cv:.4f}\n")
                f.write(f"- Raw Scores: {cv_scores.tolist()}\n")

        print(f"Classification report saved to: {report_path}")

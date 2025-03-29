import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd



def train_knn(X_train, y_train, n_neighbors=5):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train, max_depth=None):
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None):
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    return model


def train_svm(X_train, y_train, kernel='rbf'):
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)
    return model

def train_logistic_regression(X_train, y_train, max_iterations=1000):
    model = LogisticRegression(max_iter=max_iterations)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, save_path=None):
    y_pred = model.predict(X_test)
    print("📊 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    if save_path:
        df_results = pd.DataFrame(X_test, columns=[f"Feature_{i+1}" for i in range(X_test.shape[1])])
        df_results["True_Label"] = y_test
        df_results["Predicted_Label"] = y_pred
        df_results.to_csv(save_path, index=False)
        print(f"💾 Predikcie uložené do: {save_path}")

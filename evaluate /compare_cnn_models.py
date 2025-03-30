import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from preprocessing.prepare_shared_cnn_dataset import prepare_shared_cnn_dataset

# Načítanie dát
X_segments, X_fuzzy, y_encoded, encoder = prepare_shared_cnn_dataset()
X_train, X_test, X_fuzzy_train, X_fuzzy_test, y_train, y_test = train_test_split(
    X_segments, X_fuzzy, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# Načítanie modelov
cnn_model = load_model("results/models/cnn_ecg_model.h5")
hybrid_model = load_model("results/models/cnn_hybrid_model.h5")

# Predikcie
y_pred_cnn = np.argmax(cnn_model.predict(X_test), axis=1)
y_pred_hybrid = np.argmax(hybrid_model.predict([X_test, X_fuzzy_test]), axis=1)

# Reporty
report_cnn = classification_report(y_test, y_pred_cnn, target_names=encoder.classes_, output_dict=True)
report_hybrid = classification_report(y_test, y_pred_hybrid, target_names=encoder.classes_, output_dict=True)

# Výstupná tabuľka
rows = []
for cls in encoder.classes_:
    rows.append({
        "Class": cls,
        "CNN_F1": report_cnn[cls]["f1-score"],
        "CNN_Precision": report_cnn[cls]["precision"],
        "CNN_Recall": report_cnn[cls]["recall"],
        "Hybrid_F1": report_hybrid[cls]["f1-score"],
        "Hybrid_Precision": report_hybrid[cls]["precision"],
        "Hybrid_Recall": report_hybrid[cls]["recall"]
    })

df = pd.DataFrame(rows)
print(df.to_string(index=False))

# 💾 Uloženie do CSV
df.to_csv("results/reports/cnn_vs_hybrid_cnn_comparison.csv", index=False)
print("💾 Výsledková tabuľka uložená do results/reports/cnn_vs_hybrid_cnn_comparison.csv")
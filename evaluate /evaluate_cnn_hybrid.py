import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

from preprocessing.load import load_mitbih_record
from preprocessing.segmentation import segment_heartbeats
from preprocessing.annotation_mapping import map_annotations_to_peaks
from config import RECORD_NAMES

# Parametre segmentácie
PRE_R = 0.2
POST_R = 0.4

all_segments = []
y_labels = []

for record in RECORD_NAMES:
    try:
        signal, fs, r_peaks, _, _, annotation = load_mitbih_record(record, path="../data/mit/")
        segments = segment_heartbeats(signal, r_peaks, fs, pre_R=PRE_R, post_R=POST_R)
        adjusted_peaks = r_peaks[1:len(segments)+1]
        labels = map_annotations_to_peaks(adjusted_peaks, annotation.sample, annotation.symbol)

        for seg, label in zip(segments, labels):
            if label != "Unknown":
                all_segments.append(seg)
                y_labels.append(label)

        print(f"✅ Záznam {record} pripravený: {len(segments)} segmentov")

    except Exception as e:
        print(f"⚠️ Chyba pri zázname {record}: {e}")

X = np.expand_dims(np.array(all_segments), axis=-1)
y = np.array(y_labels)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Načítanie modelu
model = load_model("results/models/cnn_ecg_model.h5")

# Predikcie a vyhodnotenie
y_pred_probs = model.predict(X)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_encoded, y_pred))

print("\n📋 Classification Report:")
print(classification_report(y_encoded, y_pred, target_names=encoder.classes_))

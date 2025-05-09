# smote_analysis.py
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from config import FEAT_DIR, ALL_RECORDS
from pathlib import Path

# --- Načítanie a spojenie všetkých datasetov do jedného DataFrame ---
dfs = []

for record in ALL_RECORDS:
    file_path = FEAT_DIR / f"ecg_{record}_features_full.csv"
    if Path(file_path).exists():
        temp_df = pd.read_csv(file_path)
        temp_df['record_id'] = record  # pridanie stĺpca record_id
        dfs.append(temp_df)
    else:
        print(f"Súbor {file_path} neexistuje!")

df = pd.concat(dfs, ignore_index=True)

# --- Rozdelenie pacientov (train/test) ---

patients = df['record_id'].unique()
train_patients, test_patients = train_test_split(
    patients, test_size=0.3, random_state=42)

train_df = df[df['record_id'].isin(train_patients)]
# pred SMOTE pridaj tento riadok:
train_df = train_df.dropna(subset=['HR_bpm', 'QRS_ms', 'T_amp', 'Annotation'])

# --- SMOTE (s príznakmi, ktoré si zvolil) ---
X_train = train_df[['HR_bpm', 'QRS_ms', 'T_amp']]
y_train = train_df['Fuzzy_label']  # <- správne meno stĺpca!
print(y_train.value_counts())
X_resampled, y_resampled = SMOTE(random_state=42, k_neighbors=3).fit_resample(X_train, y_train)

smote_df = pd.DataFrame(X_resampled, columns=['HR_bpm', 'QRS_ms', 'T_amp'])
smote_df['Fuzzy_label'] = y_resampled

# --- Vizualizácia pre fuzzy úpravy ---
for feature in ['HR_bpm', 'QRS_ms', 'T_amp']:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='Annotation', y=feature, data=smote_df, palette='Set2')  # <- správny stĺpec!
    plt.title(f'Rozdelenie {feature} podľa tried (po SMOTE)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FEAT_DIR / f'{feature}_smote_boxplot.png')
    plt.show()
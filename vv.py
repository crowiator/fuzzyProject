import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use("TkAgg")
# Načítaj CSV
df = pd.read_csv("results/reports/qrs_comparison.csv")

# 1. 🧮 Percento výberu zdroja TWA
source_counts = df["TWA_source"].value_counts(normalize=True) * 100
print("📊 Percento použitia TWA zdroja:")
print(source_counts.round(2))

# 2. 📦 Rozdelenie rozdielov medzi orig a wavelet
plt.figure(figsize=(10, 5))
sns.histplot(df["abs_diff"], bins=30, kde=True, color="teal")
plt.title("Rozdiel medzi TWA_orig a TWA_wavelet")
plt.xlabel("Absolútny rozdiel (mV)")
plt.ylabel("Počet úderov")
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. 📊 Boxplot rozdelenia podľa zdroja
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="TWA_source", y="TWA_used", palette="pastel")
plt.title("Rozdelenie TWA použitých hodnôt podľa zdroja")
plt.ylabel("TWA (mV)")
plt.xlabel("Zdroj")
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. 🧹 Vyexportuj záznamy s veľkým rozdielom
threshold = 0.3  # môžeš si nastaviť
dirty = df[df["abs_diff"] > threshold]
dirty.to_csv("results/reports/twa_conflict_cases.csv", index=False)
print(f"🧼 Uložené {len(dirty)} 'špinavých' záznamov s rozdielom > {threshold} do twa_conflict_cases.csv")
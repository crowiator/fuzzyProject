import pandas as pd
import matplotlib
import pandas as pd
from collections import Counter
import sns as sns
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import os
matplotlib.use("TkAgg")
df = pd.read_csv("results/reports/twa_wavelet_vs_original.csv")

# Popisné štatistiky
print(df[["TWA_orig", "TWA_wavelet", "abs_diff"]].describe())

plt.figure(figsize=(8, 4))
plt.hist(df["abs_diff"], bins=50, color='skyblue', edgecolor='black')
plt.title("Rozdiely medzi TWA_orig a TWA_wavelet")
plt.xlabel("Absolútny rozdiel (mV)")
plt.ylabel("Počet úderov")
plt.grid(True)
plt.tight_layout()
plt.show()

count_extreme = (df["abs_diff"] > 0.4).sum()
print(f"🔍 Počet úderov s rozdielom > 0.4 mV: {count_extreme} z {len(df)} ({100 * count_extreme / len(df):.2f}%)")
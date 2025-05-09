# scripts/explore_centroids.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from config import OUT_DIR, ALL_RECORDS
import matplotlib.pyplot as plt
import matplotlib
from config import ALL_RECORDS, LEAD, OUT_DIR, FEAT_DIR

matplotlib.use('TkAgg')
# 1) načítaj všetky uložené fuzzy CSV
dfs = []
for rec in ALL_RECORDS:
    p = OUT_DIR / f"mitdb_{rec}_fuzzy_results.csv"
    if p.is_file():
        dfs.append(pd.read_csv(p).assign(Record=rec))
df = pd.concat(dfs, ignore_index=True)

# 2) základné štatistiky centier podľa *skutočnej* triedy
stats = (
    df.groupby("TrueLabel")["Centroid"]
      .agg(["count", "min", "median", "quantile"])
)
print(stats)

# 3) histogramy – uvidíš pretečenie do 'závažná'
for lbl, sub in df.groupby("TrueLabel"):
    plt.hist(sub["Centroid"], bins=50, alpha=.4, label=lbl, density=True)
plt.axvline(0.25, c="k", ls="--", label="hr. mierna")
plt.axvline(0.60, c="r", ls="--", label="hr. závažná")
plt.legend(); plt.xlabel("Centroid"); plt.show()

# --- Detailné zobrazenie priebehov signálu bolo odstránené.
# Ak chceš kresliť surový a filtrovaný signál, nahraj sem premenné
# `time_axis`, `signal`, `signal_lowpass`, `signal_dwt`, `signal_filtered`
# a definuj `samples`, potom zakóduj grafy podobne ako predtým.
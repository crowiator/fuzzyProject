"""
compute_mf_params.py
--------------------

Skript na **vygenerovanie parametrov fuzzy členovacích funkcií (MF)** z
tréningovej tabuľky príznakov. Navrhnutý je tak, aby sa dal spustiť priamo
v PyCharme – stačí upraviť pár premenných v sekcii *NASTAVENIA* a stlačiť
▶ Run.

### Čo robí
1. Načíta tabuľku s príznakmi (CSV / Parquet / Feather).
2. Pre každý zvolený číselný stĺpec vypočíta percentily **10 | 50 | 90**.
3. Z týchto bodov vybuduje tri MF (LOW / MED / HIGH) s miernym rozšírením
   okrajov o `EXPAND` %.
4. Výsledok uloží do JSON, ktorý sa dá priamo načítať vo
   `fuzzy_classifier.py`.

---
"""
from __future__ import annotations
from config import ALL_RECORDS, FEAT_DIR

import json
from pathlib import Path
from typing import Sequence, Dict, Any

import pandas as pd
import numpy as np

# --------------------------------------------------------------------- #
# 🛠  NASTAVENIA – uprav podľa potreby
# --------------------------------------------------------------------- #
FEATURES_DIR = Path(".")                # priečinok, kde ležia súbory ecg_<ID>_features_full.csv
OUTPUT_PATH  = Path("mf_params_all.json")
FEATURES     = ["HR_bpm", "QRS_ms", "T_amp"]
# None = spracujú sa všetky číselné stĺpce
#a  # alebo napr. ["HR_bpm", "QRS_ms", "T_amp"]


# Helper to load and concatenate feature tables for all records
def load_feature_tables(record_ids: Sequence[str]) -> pd.DataFrame:
    """Načíta a spojí CSV tabulky ecg_<ID>_features_full.csv pre všetky ID."""
    dfs = []
    for rec in record_ids:
        csv_path = FEAT_DIR / f"ecg_{rec}_features_full.csv"
        if not csv_path.exists():
            print(f"⚠️  Súbor {csv_path} neexistuje – preskakujem.")
            continue
        dfs.append(pd.read_csv(csv_path))
    if not dfs:
        raise FileNotFoundError("Neboli nájdené žiadne súbory s príznakmi.")
    return pd.concat(dfs, ignore_index=True)


PERCENTILES = (10, 50, 90)   # (low, med, high)
EXPAND      = 0.05           # rozšírenie okrajov (5 %)
# --------------------------------------------------------------------- #


def percentiles_to_mf(values: np.ndarray,
                      p_low: int, p_med: int, p_high: int,
                      expand: float) -> Dict[str, Any]:
    """Z percentilov vytvorí parametre troch MF pre jeden príznak."""
    p10, p50, p90 = np.percentile(values, [p_low, p_med, p_high])
    delta = (p90 - p10) * expand

    x_min, x_max = float(values.min()), float(values.max())

    # body lichobežníkov/ trojuholníka
    a = max(p10 - delta, x_min)
    b = p10
    c = p50
    d = p90
    e = min(p90 + delta, x_max)

    return {
        "universe": [x_min, x_max],
        "low":  [x_min, x_min, a, b],
        "med":  [a, b, d],
        "high": [c, d, x_max, x_max],
    }


def load_table(path: Path) -> pd.DataFrame:
    """Načíta tabuľku podľa prípony súboru."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".feather", ".ft"}:
        return pd.read_feather(path)
    raise ValueError(f"Nepodporovaný formát súboru: {path.suffix}")


def main() -> None:
    # načítame a spojíme všetky tabuľky podľa ALL_RECORDS
    df = load_feature_tables(ALL_RECORDS)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    sel_cols: Sequence[str] = FEATURES or numeric_cols
    sel_cols = [c for c in sel_cols if c in numeric_cols]

    if not sel_cols:
        raise ValueError("V tabuľke nie sú žiadne číselné stĺpce na spracovanie.")

    result: Dict[str, Any] = {}
    for col in sel_cols:
        values = df[col].dropna().to_numpy()
        result[col] = percentiles_to_mf(
            values,
            p_low=PERCENTILES[0],
            p_med=PERCENTILES[1],
            p_high=PERCENTILES[2],
            expand=EXPAND,
        )

    OUTPUT_PATH.write_text(json.dumps(result, indent=2))
    print(f"✅ MF parametre z {len(df)} úderov uložené do {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
# toto robim
import wfdb, neurokit2 as nk
import pandas as pd, numpy as np, math
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
from preprocessing.filtering import lowpass_filter, dwt_filtering
from preprocessing.annotation_mapping import ANNOTATION_TO_FUZZY, EXCLUDED_ANN
matplotlib.use('TkAgg')
from config import FEAT_DIR, MIT_LOCAL_DIR


PREFERRED_LEADS = ["MLII", "II", "V1", "V2", "V5", "V6"]
# --------------------------------------------------------------------- #
# 1 · Vizualizácia surového a filtrovaného signálu
# --------------------------------------------------------------------- #
def plot_raw_vs_clean(raw_signal: np.ndarray,
                      clean_signal: np.ndarray,
                      fs: int,
                      start_sec: float = 0.0,
                      duration_sec: float = 5.0):
    """Vykreslí prekrytie surový × vyčistený EKG pre zvolené časové okno."""
    start = int(start_sec * fs)
    stop = int((start_sec + duration_sec) * fs)
    t = np.arange(start, stop) / fs  # časová os v sekundách

    plt.figure(figsize=(9, 3))
    plt.plot(t, raw_signal[start:stop], label="Surový signál", alpha=0.6)
    plt.plot(t, clean_signal[start:stop], label="Vyčistený signál", linewidth=1.2)
    plt.xlabel("Čas (s)")
    plt.ylabel("Amplitúda (mV)")
    plt.title(f"EKG – surový vs. vyčistený ({duration_sec}-sek úsek)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def choose_lead(rec, preferred=PREFERRED_LEADS) -> str:
    """
    Vráti prvý preferovaný zvod, ktorý sa v danom zázname naozaj nachádza.
    Ak žiadny z preferovaných nie je, vráti rec.sig_name[0].
    """
    for ld in preferred:
        if ld in rec.sig_name:
            return ld
    # fallback – prvý dostupný kanál
    return rec.sig_name[0]

# --------------------------------------------------------------------- #
# 2 · Načítanie záznamu
# --------------------------------------------------------------------- #
def load_record(record_id: str = "100", lead: str = "MLII",
                base_dir: Path = MIT_LOCAL_DIR):
    """
        Načíta MIT-BIH záznam z lokálneho priečinka data/mit
        a vráti (raw_signal, fs, ann, used_lead).
        Ak `lead` je None alebo v zázname chýba, vyberie sa automaticky.
        """
    rec_path = base_dir / record_id  # napr. data/mit/100
    # načítanie signálu a anotácií z lokálnej cesty
    rec = wfdb.rdrecord(str(rec_path))  # .dat + .hea
    ann = wfdb.rdann(str(rec_path), "atr")  # .atr

    # -- výber zvodu ------------------------------------------------------
    if lead is None or lead not in rec.sig_name:
        lead = choose_lead(rec)  # ⬅️ funkcia z bodu 1
        print(f"⚠️  Lead MLII nie je k dispozícii, používam {lead}")

    fs = rec.fs
    raw = rec.p_signal[:, rec.sig_name.index(lead)]
    return raw, fs, ann

# --------------------------------------------------------------------- #
# 3 · Čistenie signálu
# --------------------------------------------------------------------- #
def custom_filtering(raw, fs,
                    lp_cutoff=30, lp_order=4,
                    wavelet="db4", thr_factor=0.2):
    """
    Všetko čistenie v jednom kroku:
        1. low-pass (EMG, RF šum)           – lowpass_filter
        2. DWT baseline-wander removal      – dwt_filtering
    Výstup:
        signals – numpy.ndarray (N,)  alebo DataFrame,
        info    – slovník s meta-údajmi (využijete neskôr)
    """
    sig_lp   = lowpass_filter(raw, fs, cutoff=lp_cutoff, order=lp_order)
    sig_filt = dwt_filtering(sig_lp, wavelet=wavelet,
                             threshold_factor=thr_factor)

    # Ak zvyšok pipeline čaká DataFrame s kolónkou „ECG_Clean“:
    signals = pd.DataFrame({"ECG_Clean": sig_filt})
    info    = {"fs": fs,
               "lp_cutoff": lp_cutoff,
               "wavelet": wavelet,
               "thr_factor": thr_factor}
    return signals, info


def clean_ecg(raw_signal: np.ndarray, fs: int, method: str = "neurokit"):
    """
    Spustí nk.ecg_process() a vráti:
        • signals – DataFrame so všetkými stĺpcami,
        • info    – slovník s metadátami.
    """
    if method == "neurokit":
        signals, info = nk.ecg_process(raw_signal, sampling_rate=fs, method=method)
        return signals, info
    if method == "custom":
        filt_sig, info = custom_filtering(raw_signal, fs, lp_cutoff=30, lp_order=4,
                    wavelet="db4", thr_factor=0.2)
        return filt_sig, info  # filt_sig môže byť ndarray alebo DataFrame

        # fallback – žiadne spracovanie
    return raw_signal, {"fs": fs, "note": "raw"}

# --------------------------------------------------------------------- #
# 4 · Výpočet príznakov okolo MIT-BIH R-peakov
# --------------------------------------------------------------------- #
def extract_features(clean_signal: np.ndarray,
                      fs: int,
                      ann,
                      record_id: str,
                     out_csv: Path | None = None) -> pd.DataFrame:
    """
    Z vyčisteného signálu vyťaží:
        • HR_bpm      – srdcovú frekvenciu,
        • QRS_ms      – trvanie QRS komplexu,
        • T_amp       – amplitúdu T-vlny,
    pričom R-peaky sa berú priamo z *.atr* (zlatý štandard).
    """
    # Delineácia bodov okolo známych R-peakov
    # 	NeuroKit dostane presné pozície vrcholov a okolo každého z nich
    # deteguje tzv. fiduciálne body:
    # 	•	ECG_R_Onsets – index, kde QRS začína (približne nástup Q-vlny),
    # 	•	ECG_R_Offsets – index konca QRS (koniec S-vlny),
    # 	•	ECG_T_Peaks – vrchol T-vlny.
    # info - slovnik obsahuje tieto veci
    _, info = nk.ecg_delineate(clean_signal,
                               rpeaks=ann.sample,
                               sampling_rate=fs,
                               method="dwt",
                               show=False)

    r_on = np.asarray(info["ECG_R_Onsets"])
    r_off = np.asarray(info["ECG_R_Offsets"])
    t_pk = np.asarray(info["ECG_T_Peaks"])


    rows, prev_r = [], math.nan
    # 	•	k            – poradové číslo úderu,
    # 	•	r_idx        – index R-peaku v signále.
    for k, r_idx in enumerate(ann.sample):

        # QRS dĺžka (ms) – ak chýba onset/offset, zostane NaN
        if np.isnan(r_on[k]) or np.isnan(r_off[k]):
            qrs_ms = np.nan
        else:
            qrs_ms = (r_off[k] - r_on[k]) / fs * 1000
            # 	•	filter < 40 ms (pravdepodobne šum) alebo > 300 ms (zlý onset/offset).
            if not 40 <= qrs_ms <= 400:  # fyziologicky akceptovateľné hranice
                qrs_ms = np.nan

        # Amplitúda T-vlny
        t_amp = clean_signal[int(t_pk[k])] if not np.isnan(t_pk[k]) else np.nan

        # Heart-rate z predchádzajúceho RR intervalu
        # 	•	RR-interval = rozdiel dvoch po sebe idúcich R-peakov.
        # 	•	Pre prvý beat (prev_r je NaN) HR nepočítame.
        # 	•	Orez 30-180 bpm, aby sa extrasystola s RR ≈ 0.16 s (366 bpm) ignorovala.
        hr = 60 / ((r_idx - prev_r) / fs) if not math.isnan(prev_r) else np.nan
        if hr < 20 or hr > 220:
            hr = np.nan
        prev_r = r_idx

        rows.append({
            "Record": record_id,
            "R_index": int(r_idx),
            "HR_bpm": hr,
            "QRS_ms": qrs_ms,
            "T_amp": t_amp,
            "Annotation": ann.symbol[k],
        })

    df = pd.DataFrame(rows)
    # --- ukladanie --------------------------------------------------------
    if out_csv is None:
        out_csv = FEAT_DIR / f"ecg_{record_id}_features_full.csv"
    else:
        out_csv = Path(out_csv)

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if out_csv.exists():
        print(f"ℹ️  Prepisujem existujúci súbor {out_csv}")

    df.to_csv(out_csv, index=False)
    print(f"✅ Súbor uložený do {out_csv}")
    return df


def ecg_plot(signals, info, fs, seconds=5):
    """
    Zobrazí oficiálny NeuroKit graf pre prvých *seconds* sekúnd.
    """
    nk.ecg_plot(signals.iloc[: seconds * fs], info)
    plt.suptitle(f"Prvých {seconds} sekúnd – MLII", fontweight="bold")
    fig = plt.gcf()
    fig.set_size_inches(11, 7, forward=True)

    # panel s priemerným úderom je tretí (index 2)
    ax_avg = fig.axes[2]

    # ak už legenda existuje (nemali sme ju remove())
    if ax_avg.get_legend() is not None:
        ax_avg.legend(loc="upper right", framealpha=0.9, fontsize=8)

    # voliteľne: skráť text nad grafom, aby nezasahoval
    ax_avg.set_title("Priemerný tvar QRS + P, T", fontsize=10)

    plt.tight_layout()
    fig.savefig("ecg_overview.png", dpi=300, bbox_inches="tight")
    plt.show()


# --------------------------------------------------------------------- #
# Demo – spustí sa iba ak skript voláš priamo
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    # 1) načítaj záznam a anotácie
    print("1")
    raw_ecg, fs, ann = load_record("100", lead="MLII")
    print("1")
    # 2) vyčisti signál
    clean_ecg_sig_df, info = clean_ecg(raw_ecg, fs)
    clean_ecg_sig = clean_ecg_sig_df["ECG_Clean"].to_numpy()
    # ecg_plot(signals=clean_ecg_sig_df, info=info, fs=fs, seconds=5)
    # 3) vyťaž príznaky do DataFrame
    df = extract_features(clean_ecg_sig, fs, ann, record_id="100",
                          out_csv="ecg_100_features_full.csv")
    print("1")
    print("Počet riadkov:", len(df))
    print(df["Annotation"].value_counts(), "\n")

    print("Rozsahy:")
    print(df[["HR_bpm", "QRS_ms", "T_amp"]].describe().loc[["min", "max"]])

    print("\nNaN podiel:")
    print((df[["HR_bpm", "QRS_ms", "T_amp"]].isna().mean() * 100).round(1), "%")
    # 4) zobraz 5-sekundový úsek na vizuálnu kontrolu
    # plot_raw_vs_clean(raw_ecg, clean_ecg_sig, fs, start_sec=0, duration_sec=5)

    # 5) rýchle štatistiky
    print(df.describe()[["HR_bpm", "QRS_ms", "T_amp"]])
    print("\nPočty úderov podľa anotácie:")
    print(df["Annotation"].value_counts())

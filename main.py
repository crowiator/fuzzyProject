import matplotlib
import numpy as np
import wfdb
import pandas as pd
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import pywt
import neurokit2 as nk
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from biosppy.signals import ecg
from scipy.integrate import trapezoid


import pandas as pd

# 1️⃣ Funkcia na načítanie EKG signálu a anotácií

def load_mitbih_record(record_name, path="./mit/"):
    """Načíta EKG signál a anotácie z MIT-BIH databázy."""
    try:
        record = wfdb.rdrecord(f"{path}{record_name}")
        annotation = wfdb.rdann(f"{path}{record_name}", 'atr')
        fs = record.fs
        signal = record.p_signal[:, 0]

        # Extrahujeme R-peaky a kategórie tepov
        r_peak_positions = annotation.sample

        # aby neobshahoval nejake speicalne symboly
        beat_types = annotation.symbol
        valid_beats = ["N", "L", "R", "V", "A"]
        filtered_r_peaks, filtered_beat_types = zip(
            *[(r, bt) for r, bt in zip(r_peak_positions, beat_types) if bt in valid_beats]
        )
        print(beat_types)

        return signal, fs, r_peak_positions, filtered_beat_types
    except Exception as e:
        print(f"❌ Chyba pri načítaní záznamu {record_name}: {e}")
        return None, None, None, None


# 2️⃣ Funkcia na odstránenie baseline driftu (Butterworth High-Pass Filter)
def butter_highpass(signal, cutoff=0.5, fs=360, order=4):
    """Aplikuje Butterworth High-Pass Filter na EKG signál."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


def wavelet_denoising(signal, wavelet='db4', level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    threshold = np.median(np.abs(coeffs[-level])) / 0.6745
    coeffs_thresholded = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    denoised_signal = pywt.waverec(coeffs_thresholded, wavelet)

    # Kontrola dĺžky
    if len(denoised_signal) > len(signal):
        denoised_signal = denoised_signal[:len(signal)]
    elif len(denoised_signal) < len(signal):
        denoised_signal = np.pad(denoised_signal, (0, len(signal) - len(denoised_signal)), mode='reflect')

    return denoised_signal


# 3️⃣ Funkcia na vykreslenie porovnania surového a filtrovaného signálu
def compare_signals(filtered_signal, denoised_signal, r_peaks, record_name, num_samples=1000):
    plt.figure(figsize=(12, 5))

    # 🔹 1. Graf - Butterworth filter (baseline drift odstránený)
    plt.subplot(1, 2, 1)
    plt.plot(filtered_signal[:num_samples], label="Po Butterworth filtre", color="b")
    plt.scatter(r_peaks[r_peaks < num_samples], filtered_signal[r_peaks[r_peaks < num_samples]], color="r", marker='o',
                label="R-peaky")
    plt.xlabel("Vzorky")
    plt.ylabel("Amplitúda")
    plt.title(f"EKG po Butterworth filtre - {record_name}")
    plt.legend()

    # 🔹 2. Graf - Po Waveletovej filtrácii (DWT)
    plt.subplot(1, 2, 2)
    plt.plot(denoised_signal[:num_samples], label="Po DWT filtrácii", color="g")
    plt.scatter(r_peaks[r_peaks < num_samples], denoised_signal[r_peaks[r_peaks < num_samples]], color="r", marker='o',
                label="R-peaky")
    plt.xlabel("Vzorky")
    plt.ylabel("Amplitúda")
    plt.title(f"EKG po Wavelet filtrácii (DWT) - {record_name}")
    plt.legend()

    plt.tight_layout()
    plt.show()


# Funkcia na detekciu R-vĺn pomocou Pan-Tompkins algoritmu (NeuroKit2)
def detect_peaks(signal, fs):
    """
    Vykoná detekciu R-vĺn v EKG signáli pomocou Pan-Tompkins algoritmu.

    Parametre:
    signal (array): filtrovaný a odšumený EKG signál
    fs (int): vzorkovacia frekvencia signálu

    Výstup:
    r_peaks (array): indexy detegovaných R-vĺn
    processed_signals (DataFrame): signál so spracovaním (QRS komplexy, T/P vlny)
    """
    processed_signals, info = nk.ecg_process(signal, sampling_rate=fs)
    r_peaks = info["ECG_R_Peaks"]
    return r_peaks, processed_signals


# Funkcia na vizualizáciu výsledkov detekcie R-vĺn
def plot_detected_r_peaks(signal, r_peaks, fs, record_name, num_samples=2000):
    """
    Vykreslí EKG signál s detegovanými R-vlnami.

    Parametre:
    signal (array): filtrovaný EKG signál
    r_peaks (array): indexy detegovaných R-vĺn
    fs (int): vzorkovacia frekvencia
    record_name (str): názov EKG záznamu
    num_samples (int): počet vzoriek na zobrazenie v grafe
    """
    plt.figure(figsize=(12, 4))

    # Vykreslenie signálu
    plt.plot(signal[:num_samples], label="Filtrovaný EKG signál", color="navy")

    # Vykreslenie detegovaných R-vĺn
    plt.scatter(r_peaks[r_peaks < num_samples],
                signal[r_peaks[r_peaks < num_samples]],
                color='red', marker='o', label="Detegované R-vlny")

    plt.xlabel("Vzorky")
    plt.ylabel("Amplitúda")
    plt.title(f"EKG signál a detegované R-vlny (Pan-Tompkins) - {record_name}")
    plt.legend()
    plt.show()


# Funkcia na vyhodnotenie presnosti detekcie voči MIT-BIH anotáciám
def evaluate_detection(r_peaks_detected, r_peaks_true):
    """
    Porovná detegované R-vlny s anotáciami z MIT-BIH databázy a vypočíta TP, FP, FN.

    Parametre:
    r_peaks_detected (array): detegované R-vlny
    r_peaks_true (array): skutočné (anotované) R-vlny z databázy
    tolerance (int): povolená odchýlka detekcie v počte vzoriek

    Výstup:
    Vypíše počet True Positives, False Positives a False Negatives
    """
    tolerance = int(0.03 * fs)
    true_positives = sum([1 for peak in r_peaks_detected if any(abs(r_peaks_true - peak) <= tolerance)])
    false_positives = len(r_peaks_detected) - true_positives
    false_negatives = len(r_peaks_true) - true_positives

    print(f"✔️ True Positives (TP): {true_positives}")
    print(f"❌ False Positives (FP): {false_positives}")
    print(f"❌ False Negatives (FN): {false_negatives}")

    # Funkcia na segmentáciu EKG signálu





def plot_single_segment(segment, fs, record_name, segment_index=0):
    """
    Vykreslí jeden segment EKG signálu (jedno srdcové tepo).

    Parametre:
    segment (array): vybraný segment EKG signálu
    fs (int): vzorkovacia frekvencia signálu
    record_name (str): názov záznamu
    segment_index (int): index segmentu v zozname (iba pre označenie)
    """
    plt.figure(figsize=(8, 4))

    # Generovanie časovej osi
    time_axis = np.linspace(0, len(segment) / fs, len(segment))

    plt.plot(time_axis, segment, label=f"Segment {segment_index}", color="blue")
    plt.axvline(x=time_axis[len(segment) // 2], color='red', linestyle="--", label="R-peak (odhad)")

    plt.xlabel("Čas (s)")
    plt.ylabel("Amplitúda")
    plt.title(f"EKG Segment {segment_index} - {record_name}")
    plt.legend()
    plt.show()


# Funkcia na vylepšenú detekciu S-píkov
def detect_s_peaks(signal, r_peaks, t_peaks, fs):
    s_peaks = []
    for i in range(len(r_peaks)):
        r_idx = r_peaks[i]
        t_idx = t_peaks[i] if i < len(t_peaks) else r_idx + int(0.2 * fs)  # Ak nie je T peak, nastavíme aproximáciu
        s_idx = np.argmin(signal[r_idx: min(r_idx + int(0.15 * fs), len(signal))]) + r_idx  # Najnižší bod medzi R a T
        s_peaks.append(s_idx)
    return np.array(s_peaks)


# Funkcia na vykreslenie P, Q, R, S, T píkov

def plot_peaks(signal, p_peaks, q_peaks, r_peaks, s_peaks, t_peaks, fs, record_name, num_samples=1000):
    plt.figure(figsize=(12, 6))
    plt.plot(signal[:num_samples], label="EKG Signal", color="blue")

    plt.scatter(p_peaks[p_peaks < num_samples], signal[p_peaks[p_peaks < num_samples]], color="green", label="P Peak",
                marker="o")
    plt.scatter(q_peaks[q_peaks < num_samples], signal[q_peaks[q_peaks < num_samples]], color="purple", label="Q Peak",
                marker="s")
    plt.scatter(r_peaks[r_peaks < num_samples], signal[r_peaks[r_peaks < num_samples]], color="red", label="R Peak",
                marker="^")
    plt.scatter(s_peaks[s_peaks < num_samples], signal[s_peaks[s_peaks < num_samples]], color="orange", label="S Peak",
                marker="v")
    plt.scatter(t_peaks[t_peaks < num_samples], signal[t_peaks[t_peaks < num_samples]], color="black", label="T Peak",
                marker="x")

    plt.xlabel("Vzorky")
    plt.ylabel("Amplitúda")
    plt.title(f"Detekcia P, Q, R, S, T píkov v EKG signále - {record_name}")
    plt.legend()
    plt.grid()
    plt.show()


def extract_features(signal, r_peaks, q_peaks, s_peaks, t_peaks, p_peaks, fs):
    """
        Extrahuje základné charakteristiky EKG signálu.

        Parametre:
        signal (array): EKG signál
        r_peaks (array): Indexy R-vĺn
        q_peaks (array): Indexy Q-vĺn
        s_peaks (array): Indexy S-vĺn
        t_peaks (array): Indexy T-vĺn
        p_peaks (array): Indexy P-vĺn
        fs (int): Vzorkovacia frekvencia

        Výstup:
        features (dict): Extrahované charakteristiky
        """
    features = {}

    # 1️⃣ RR interval (s)
    rr_intervals = np.diff(r_peaks) / fs  # prepočet na sekundy
    features["RR_interval"] = np.mean(rr_intervals) if len(rr_intervals) > 0 else np.nan

    # 2️⃣ Amplitúda R-vlny (mV)
    features["R_amplitude"] = np.mean(signal[r_peaks]) if len(r_peaks) > 0 else np.nan

    # 3️⃣ Šírka QRS komplexu (s)
    if len(q_peaks) > 0 and len(s_peaks) > 0:
        valid_qs = [(q, s) for q, s in zip(q_peaks, s_peaks) if q < s]
        if valid_qs:
            q_peaks_filtered, s_peaks_filtered = zip(*valid_qs)
            qrs_width = (np.array(s_peaks_filtered) - np.array(q_peaks_filtered)) / fs  # prepočet na sekundy
            features["QRS_width"] = np.mean(qrs_width)
        else:
            features["QRS_width"] = np.nan
    else:
        features["QRS_width"] = np.nan

    # 4️⃣ Amplitúda T-vlny (mV)
    features["T_wave_amplitude"] = np.mean(signal[t_peaks]) if len(t_peaks) > 0 else np.nan

    # 5️⃣ Amplitúda P-vlny (mV)
    features["P_wave_amplitude"] = np.mean(signal[p_peaks]) if len(p_peaks) > 0 else np.nan

    # 6️⃣ RR variabilita (ms)
    features["RR_variability"] = np.std(rr_intervals) * 1000 if len(rr_intervals) > 1 else np.nan  # v milisekundách

    return features

def fuzzy_logic_classification(features):
    # 1️⃣ Define Fuzzy Input Variables
    rr_prev = ctrl.Antecedent(np.arange(0.3, 1.5, 0.01), 'rr_prev')  # Previous RR interval (s)
    rr_next = ctrl.Antecedent(np.arange(0.3, 1.5, 0.01), 'rr_next')  # Next RR interval (s)
    p_wave = ctrl.Antecedent(np.arange(0, 1.5, 0.01), 'p_wave')  # P wave amplitude (mV)
    qrs_duration = ctrl.Antecedent(np.arange(0, 0.2, 0.01), 'qrs_duration')  # QRS duration (s)
    r_wave = ctrl.Antecedent(np.arange(0, 2.5, 0.01), 'r_wave')  # R wave amplitude (mV)
    t_wave = ctrl.Antecedent(np.arange(-0.5, 0.5, 0.01), 't_wave')  # T wave amplitude (mV)

    # 2️⃣ Define Fuzzy Output Variable (ECG Beat Type)
    beat_type = ctrl.Consequent(np.arange(0, 10, 1), 'beat_type')

    # 3️⃣ Define Membership Functions for Input Variables
    # RR Interval (Short, Normal, Long)
    rr_prev['short'] = fuzz.trimf(rr_prev.universe, [0.3, 0.4, 0.6])
    rr_prev['normal'] = fuzz.trimf(rr_prev.universe, [0.5, 0.8, 1.1])
    rr_prev['long'] = fuzz.trimf(rr_prev.universe, [0.9, 1.3, 1.5])

    rr_next['short'] = fuzz.trimf(rr_next.universe, [0.3, 0.4, 0.6])
    rr_next['normal'] = fuzz.trimf(rr_next.universe, [0.5, 0.8, 1.1])
    rr_next['long'] = fuzz.trimf(rr_next.universe, [0.9, 1.3, 1.5])

    # P wave amplitude (Absent, Low, High)
    p_wave['absent'] = fuzz.trapmf(p_wave.universe, [0, 0, 0.1, 0.2])
    p_wave['low'] = fuzz.trimf(p_wave.universe, [0.1, 0.3, 0.6])
    p_wave['high'] = fuzz.trapmf(p_wave.universe, [0.5, 1.0, 1.5, 1.5])

    # QRS duration (Narrow, Wide)
    qrs_duration['narrow'] = fuzz.trapmf(qrs_duration.universe, [0, 0, 0.08, 0.12])
    qrs_duration['wide'] = fuzz.trapmf(qrs_duration.universe, [0.1, 0.14, 0.2, 0.2])

    # R wave amplitude (Low, Normal, High)
    r_wave['low'] = fuzz.trapmf(r_wave.universe, [0, 0, 0.3, 0.6])
    r_wave['normal'] = fuzz.trimf(r_wave.universe, [0.3, 0.7, 1.1])
    r_wave['high'] = fuzz.trapmf(r_wave.universe, [0.8, 1.5, 2.5, 2.5])

    # T wave amplitude (Inverted, Normal)
    t_wave['inverted'] = fuzz.trapmf(t_wave.universe, [-0.5, -0.3, -0.1, 0.0])
    t_wave['normal'] = fuzz.trapmf(t_wave.universe, [0.0, 0.1, 0.5, 0.5])

    # 4️⃣ Define Membership Functions for Output (ECG Beat Types)
    beat_type['N'] = fuzz.trimf(beat_type.universe, [0, 0, 2])  # Normal
    beat_type['S'] = fuzz.trimf(beat_type.universe, [2, 3, 4])  # Supraventricular
    beat_type['V'] = fuzz.trimf(beat_type.universe, [4, 5, 6])  # Ventricular
    beat_type['F'] = fuzz.trimf(beat_type.universe, [6, 7, 8])  # Fusion
    beat_type['Q'] = fuzz.trimf(beat_type.universe, [8, 9, 10])  # Unknown

    # 5️⃣ Define Fuzzy Rules
    rule1 = ctrl.Rule((p_wave['low'] | p_wave['high']) & rr_prev['normal'] & qrs_duration['narrow'], beat_type['N'])
    rule2 = ctrl.Rule(p_wave['absent'] & rr_next['short'] & qrs_duration['narrow'], beat_type['S'])
    rule3 = ctrl.Rule(qrs_duration['wide'] & r_wave['high'] & t_wave['inverted'], beat_type['V'])
    rule4 = ctrl.Rule(qrs_duration['wide'] & r_wave['normal'] & rr_prev['normal'], beat_type['F'])
    rule5 = ctrl.Rule(rr_prev['short'] & rr_next['long'] & qrs_duration['narrow'], beat_type['Q'])

    # 6️⃣ Create and Simulate Fuzzy Inference System
    ecg_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
    ecg_sim = ctrl.ControlSystemSimulation(ecg_ctrl)

    # 7️⃣ Test the System with Sample Input
    ecg_sim.input['rr_prev'] = features["RR_interval"]  # Normal RR interval
    ecg_sim.input['rr_next'] = features["RR_variability"]  # Normal RR interval
    ecg_sim.input['p_wave'] = features["P_wave_amplitude"]  # Low P wave
    ecg_sim.input['qrs_duration'] = features["QRS_width"]  # Narrow QRS
    ecg_sim.input['r_wave'] = features["R_amplitude"]  # Normal R wave amplitude
    ecg_sim.input['t_wave'] = features["T_wave_amplitude"] # Normal T wave amplitude

    ecg_sim.compute()  # Compute fuzzy inference

    # Print Output Classification
    print(f"Predicted Beat Type: {ecg_sim.output['beat_type']}")
def save_features_to_file(features, filename):
    # Konverzia do DataFrame
    df = pd.DataFrame([features])

    # Uloženie do CSV súboru
    df.to_csv(filename, index=False)

    print(f"Súbor {filename} bol úspešne uložený.")
# 4️⃣ Fuzzy logika pre klasifikáciu


# 4️⃣ Hlavná časť programu
if __name__ == "__main__":
    record_name = "100"
    raw_signal, fs, r_peaks_true, beat_types = load_mitbih_record(record_name)

    if raw_signal is not None:
        # Odstránenie baseline driftu
        # 1. Odstránenie baseline driftu
        filtered_signal = butter_highpass(raw_signal, fs=fs)

        # 2. Odstránenie vysokofrekvenčného šumu pomocou DWT
        denoised_signal = wavelet_denoising(signal=filtered_signal, wavelet='db4', level=4)


        # Detekcia R-vĺn pomocou Pan-Tompkins algoritmu
        r_peaks_detected, processed_signals = detect_peaks(denoised_signal, fs)

        # Vizualizácia výsledkov
        #plot_detected_r_peaks(denoised_signal, r_peaks_detected, fs, record_name)
        # 4. Extrakcia píkov
        p_peaks = np.where(processed_signals["ECG_P_Peaks"] == 1)[0]
        q_peaks = np.where(processed_signals["ECG_Q_Peaks"] == 1)[0]
        r_peaks = np.where(processed_signals["ECG_R_Peaks"] == 1)[0]
        t_peaks = np.where(processed_signals["ECG_T_Peaks"] == 1)[0]
        print(f"P vlna {p_peaks}")
        # 5. Vylepšená detekcia S-píkov
        s_peaks = detect_s_peaks(denoised_signal, r_peaks, t_peaks, fs)

        # 6. Vizualizácia P, Q, R, S, T píkov

        plot_peaks(denoised_signal, p_peaks, q_peaks, r_peaks, s_peaks, t_peaks, fs, record_name)
        features = extract_features(denoised_signal, r_peaks, q_peaks, s_peaks, t_peaks, p_peaks, fs)
        print(features)
        fuzzy_logic_classification(features)



import matplotlib
import numpy as np
import wfdb
import pywt
import matplotlib.pyplot as plt
from collections import Counter
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks
from scipy.signal import welch
matplotlib.use("TkAgg")


# 1️⃣ Function to Load ECG Signal
def load_mitbih_record(record_name, path="./mit/"):
    """Load ECG signal and annotations from MIT-BIH database."""
    try:
        record = wfdb.rdrecord(f"{path}{record_name}")
        annotation = wfdb.rdann(f"{path}{record_name}", 'atr')
        fs = record.fs
        signal = record.p_signal[:, 0]

        r_peak_positions = annotation.sample
        beat_types = annotation.symbol
        beat_counts = Counter(beat_types)

        return signal, fs, r_peak_positions, beat_counts
    except Exception as e:
        print(f"❌ Error loading record {record_name}: {e}")
        return None, None, None, None


# 2️⃣ DWT-Based Baseline Drift Removal
import numpy as np
import pywt


def dwt_filtering(ecg_signal, wavelet='db4', level=9, threshold_factor=0.2):
    """
    Aplikuje DWT-based filtering s miernym oslabením detailných koeficientov.
    """
    coeffs = pywt.wavedec(ecg_signal, wavelet, level=level)

    # Výpočet univerzálneho prahu (Donoho)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = threshold_factor * sigma

    # Soft thresholding pre aproximáciu
    coeffs[0] = pywt.threshold(coeffs[0], threshold, mode='soft')

    # Namiesto úplného odstránenia detailov ich oslabíme len o 30% (miernejšia filtrácia)
    for i in range(1, 4):
        coeffs[i] *= 0.7

    # Rekonštrukcia signálu pomocou IDWT
    filtered_signal = pywt.waverec(coeffs, wavelet)

    return filtered_signal[:len(ecg_signal)]


def lowpass_filter(signal, fs, cutoff=30, order=4):
    """
    Apply a Butterworth low-pass filter to remove high-frequency noise.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)


# 3️⃣ Visualization Function
def show_plot_ecg_signal(raw_signal, filtered_signal, fs, duration):
    """Plots raw and filtered ECG signals."""
    num_samples = min(len(raw_signal), len(filtered_signal), int(fs * duration))
    time_axis = np.arange(num_samples) / fs

    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, raw_signal[:num_samples], label="Original Signal", alpha=0.7, color='red')
    plt.plot(time_axis, filtered_signal[:num_samples], label="Filtered Signal (DWT)", linewidth=2, color='blue')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Baseline Drift and Noise Removal in ECG Signal")
    plt.legend()
    plt.grid()
    plt.show()

    print(f"📊 Mean (Original): {np.mean(raw_signal):.5f}")
    print(f"📊 Mean (Filtered): {np.mean(filtered_signal):.5f}")
    print(f"🔺 Min/Max (Original): {np.min(raw_signal):.5f}, {np.max(raw_signal):.5f}")
    print(f"🔺 Min/Max (Filtered): {np.min(filtered_signal):.5f}, {np.max(filtered_signal):.5f}")
    print(f"🔹 Variance (Original): {np.var(raw_signal):.5f}")
    print(f"🔹 Variance (Filtered): {np.var(filtered_signal):.5f}")

    # Ensure equal lengths before computing correlation
    min_len = min(len(raw_signal), len(filtered_signal))
    correlation = np.corrcoef(raw_signal[:min_len], filtered_signal[:min_len])[0, 1]
    print(f"🔗 Correlation between Original and Filtered Signal: {correlation:.5f}")


def normalize_signal(filtered_signal, original_signal):
    """
    Normalize the amplitude of the filtered signal to match the original signal.
    """
    return (filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal) * np.std(original_signal) + np.mean(
        original_signal)


def normalize_zscore(signal):
    """Normalize ECG signal using Z-score normalization."""
    return (signal - np.mean(signal)) / np.std(signal)


def normalize_minmax(signal, range_min=-1, range_max=1):
    """Normalize ECG signal to a specified range (default: -1 to 1)."""
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) * (range_max - range_min) + range_min


def validate_ecg_signal(raw_signal, filtered_signal, fs, duration=2, normalize_method="zscore"):
    """Validates ECG signal after filtering by plotting and analyzing key features."""
    num_samples = min(len(raw_signal), len(filtered_signal), int(fs * duration))
    time_axis = np.arange(num_samples) / fs

    # Normalization
    if normalize_method == "zscore":
        raw_signal = normalize_zscore(raw_signal)
        filtered_signal = normalize_zscore(filtered_signal)
    elif normalize_method == "minmax":
        raw_signal = normalize_minmax(raw_signal)
        filtered_signal = normalize_minmax(filtered_signal)

    # Plot original and filtered signals
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, raw_signal[:num_samples], label="Original Signal", alpha=0.7, color='red')
    plt.plot(time_axis, filtered_signal[:num_samples], label="Filtered & Normalized Signal", linewidth=2, color='blue')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"ECG Signal Validation ({normalize_method} Normalization)")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.hist(filtered_signal, bins=50, alpha=0.7, label="Filtered Signal")
    plt.axvline(0, color='red', linestyle='dashed', linewidth=2, label="Baseline (0)")
    plt.title("Histogram Filtrovaného Signálu")
    plt.xlabel("Amplitúda")
    plt.ylabel("Počet vzoriek")
    plt.legend()
    plt.grid()
    plt.show()
    # Checking baseline drift
    baseline_before = np.mean(raw_signal)
    baseline_after = np.mean(filtered_signal)

    print(f"📊 Baseline Before Filtering: {baseline_before:.5f}")
    print(f"📊 Baseline After Filtering: {baseline_after:.5f}")

    # Ensure equal lengths before computing correlation
    min_len = min(len(raw_signal), len(filtered_signal))
    correlation = np.corrcoef(raw_signal[:min_len], filtered_signal[:min_len])[0, 1]
    print(f"🔗 Correlation between Original and Filtered Signal: {correlation:.5f}")

    # Checking if P, QRS, and T waves remain intact
    print("✅ Checking that P, QRS, and T waves are present...")
    print("   - Ensure that the main peak (R wave) is preserved.")
    print("   - The baseline should be around zero.")


def filter_and_validate_signal():
    raw_signal, fs, _, _ = load_mitbih_record("100")

    # 1. Najskôr odstránime vysokofrekvenčný šum
    filtered_signal = lowpass_filter(raw_signal, fs, cutoff=45)

    # 2. Potom odstránime baseline drift pomocou DWT
    filtered_signal = dwt_filtering(filtered_signal, threshold_factor=0.15)

    # 3. Validácia a kontrola zachovania vĺn P, QRS, T
    validate_ecg_signal(raw_signal, filtered_signal, fs, duration=5, normalize_method="zscore")


# ----------------------------------------------------------------
# Detekcia r peakov
def bandpass_filter_keep_p_t(signal, fs, lowcut=0.5, highcut=40, order=2):
    """Jemná filtrácia na zachovanie P a T vĺn (0.5 – 40 Hz)."""
    nyquist = 0.5 * fs
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype='band')
    return filtfilt(b, a, signal)


def differentiate(signal):
    """Aplikuje diferenciáciu na zdôraznenie rýchlych zmien v signáli."""
    diff_signal = np.diff(signal)
    return np.append(diff_signal, 0)


def squaring(signal):
    """Umocní signál na zvýraznenie veľkých hodnôt a potlačenie malých."""
    return np.power(signal, 2)


def moving_window_integration(signal, window_size=20):
    """Aplikuje klznú integráciu na vyhladenie signálu (menšie okno pre lepšiu detekciu R-vĺn)."""
    window = np.ones(window_size) / window_size
    return np.convolve(signal, window, mode='same')


def detect_r_peaks(integrated_signal, fs, threshold_factor=0.4):
    """Deteguje R-peaky pomocou adaptívneho prahu."""
    threshold = threshold_factor * np.max(integrated_signal)

    # Použijeme find_peaks() pre lepšiu detekciu lokálnych maxím
    peaks, _ = find_peaks(integrated_signal, height=threshold, distance=int(0.2 * fs), prominence=0.01)

    return peaks




def plot_ecg(time_axis, raw_signal, filtered_signal, detected_peaks, fs, title):

    plt.figure(figsize=(12, 6))

    # Ensure detected peaks are within valid range
    valid_peaks = detected_peaks[detected_peaks < len(filtered_signal)]

    plt.plot(time_axis, raw_signal, label="Original Signal", alpha=0.7, color='red')
    plt.plot(time_axis, filtered_signal, label="Filtered Signal", linewidth=1.5, color='blue')

    # Plot only valid detected peaks
    plt.scatter(valid_peaks / fs, filtered_signal[valid_peaks], color='green', label="Detected R-peaks", zorder=3)

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


def process_ecg(record_name="100", path="./mit/", duration=5):
    raw_signal, fs, _, _ = load_mitbih_record(record_name, path)
    if raw_signal is None:
        return

    # 1. Najskôr odstránime vysokofrekvenčný šum
    filtered_signal = lowpass_filter(raw_signal, fs, cutoff=40)

    # 2. Potom odstránime baseline drift pomocou DWT
    filtered_signal = dwt_filtering(filtered_signal, threshold_factor=0.15)

    # 3. Normalizácia Min-Max (aby R-peaky zostali správne)
    raw_signal = normalize_zscore(raw_signal)
    filtered_signal = normalize_zscore(filtered_signal)

    # 4. Pan-Tompkins metóda na zvýraznenie QRS komplexov
    diff_signal = differentiate(filtered_signal)
    squared_signal = squaring(diff_signal)
    integrated_signal = moving_window_integration(squared_signal, window_size=10)  # Väčšie okno

    # 5. Detekcia R-peakov s upravenými parametrami
    detected_r_peaks = detect_r_peaks(integrated_signal, fs, threshold_factor=0.4)

    # 6. Vizualizácia výsledkov (len prvých 5 sekúnd)
    time_axis = np.arange(min(len(raw_signal), len(filtered_signal), int(fs * duration))) / fs
    detected_peaks_within_range = detected_r_peaks[detected_r_peaks < len(time_axis)]

    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, raw_signal[:len(time_axis)], label="Original Signal", alpha=0.7, color='red')
    plt.plot(time_axis, filtered_signal[:len(time_axis)], label="Filtered Signal", linewidth=1.5, color='blue')
    plt.scatter(detected_peaks_within_range / fs, filtered_signal[detected_peaks_within_range],
                color='green', label="Detected R-peaks", zorder=3)

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("R-peak Detection with P & T Wave Preservation")
    plt.legend()
    plt.grid()
    plt.show()


def compare_with_annotations(detected_peaks, annotated_peaks, fs, tolerance=50):
    """
    Porovná detegované R-peaky s anotáciami z MIT-BIH a vypočíta presnosť detekcie.
    Tolerancia určuje maximálny rozdiel (v ms) medzi detegovaným a skutočným R-peakom.
    """

    tolerance_samples = int((tolerance / 1000) * fs)  # Konvertujeme toleranciu na počet vzoriek

    tp, fn, fp = 0, 0, 0  # True Positives, False Negatives, False Positives
    matched_annotations = np.zeros(len(annotated_peaks))

    for peak in detected_peaks:
        # Zistíme, či existuje anotovaný R-peak v tolerančnom intervale
        if np.any(np.abs(annotated_peaks - peak) <= tolerance_samples):
            tp += 1
            matched_annotations[np.argmin(np.abs(annotated_peaks - peak))] = 1
        else:
            fp += 1  # Falošné pozitíva (detegovaný R-peak, ktorý nemá reálny záznam)

    fn = np.sum(matched_annotations == 0)  # Počet R-peakov, ktoré neboli detegované

    accuracy = tp / (tp + fn + fp)

    print(f"✅ R-peak detection accuracy: {accuracy:.4f} (TP: {tp}, FN: {fn}, FP: {fp})")

    return accuracy, tp, fn, fp

def find_matching_peaks(annotated_peaks, detected_peaks, fs, tolerance=50):
    """Nájde počet správne detegovaných R-peakov s toleranciou (50 ms)."""
    # /1000 lebo prevadzam na milisekundy
    tolerance_samples = int((tolerance / 1000) * fs)  # Konverzia na počet vzoriek

    matched = []
    for peak in detected_peaks:
        if np.any(np.abs(annotated_peaks - peak) <= tolerance_samples):
            matched.append(peak)

    return np.array(matched)
def process_ecg_with_comparison(record_name="100", path="./mit/", duration=5, tolerance=50):
    raw_signal, fs, annotated_r_peaks, _ = load_mitbih_record(record_name, path)
    print(f"Anotovane R peaky: {annotated_r_peaks}")
    print(f"Pocet Anotovane R peaky: {len(annotated_r_peaks)}")
    if raw_signal is None:
        return

    # 1. Odstránime vysokofrekvenčný šum
    filtered_signal = lowpass_filter(raw_signal, fs, cutoff=40)

    # 2. Odstránime baseline drift pomocou DWT
    filtered_signal = dwt_filtering(filtered_signal, threshold_factor=0.15)

    # 3. Z-score normalizácia
    raw_signal = normalize_zscore(raw_signal)
    filtered_signal = normalize_zscore(filtered_signal)

    # 4. Pan-Tompkins metóda na zvýraznenie QRS komplexov
    diff_signal = differentiate(filtered_signal)
    squared_signal = squaring(diff_signal)
    integrated_signal = moving_window_integration(squared_signal, window_size=10)

    # 5. Detekcia R-peakov
    detected_r_peaks = detect_r_peaks(integrated_signal, fs, threshold_factor=0.35)
    print(f"Detekovane r-peakov {detected_r_peaks}")
    print(f" Pocet Detekovane r-peakov {len(detected_r_peaks)}")

    common_peaks = find_matching_peaks(annotated_r_peaks, detected_r_peaks, fs, tolerance=50)
    print(f"✅ Počet správne detegovaných R-peakov s toleranciou: {len(common_peaks)}")
    print(f"🚨 Počet chýbajúcich R-peakov: {len(annotated_r_peaks) - len(common_peaks)}")
    print(f"❌ Počet falošných detekcií: {len(detected_r_peaks) - len(common_peaks)}")
    # 6. Porovnanie s anotáciami MIT-BIH
    accuracy, tp, fn, fp = compare_with_annotations(detected_r_peaks, annotated_r_peaks, fs, tolerance)
    # 7. Vizualizácia detegovaných a anotovaných R-vĺn
    time_axis = np.arange(min(len(raw_signal), len(filtered_signal), int(fs * duration))) / fs
    detected_peaks_within_range = detected_r_peaks[detected_r_peaks < len(time_axis)]
    annotated_peaks_within_range = annotated_r_peaks[annotated_r_peaks < len(time_axis)]

    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, raw_signal[:len(time_axis)], label="Original Signal", alpha=0.7, color='red')
    plt.plot(time_axis, filtered_signal[:len(time_axis)], label="Filtered Signal", linewidth=1.5, color='blue')

    # Detegované R-peaky
    plt.scatter(detected_peaks_within_range / fs, filtered_signal[detected_peaks_within_range],
                color='green', label="Detected R-peaks", zorder=3)

    # Anotované R-peaky z MIT-BIH
    plt.scatter(annotated_peaks_within_range / fs, filtered_signal[annotated_peaks_within_range],
                color='orange', marker='x', label="Annotated R-peaks", zorder=3)

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"R-peak Detection with P & T Wave Preservation - Accuracy: {accuracy:.4f}")
    plt.legend()
    plt.grid()
    plt.show()

################################################################
# Segmentacia
#➡ Pre každú R-vlnu vyberieme segment signálu od R-0.2s do R+0.4s.
#Vytvoríme pole segmentov, kde každý segment má rovnakú dĺžku.
def segment_heartbeats(signal, r_peaks, fs, pre_R=0.2, post_R=0.4):
    """
    Segmentuje EKG signál na jednotlivé údery srdca okolo detegovaných R-vĺn.

    Parametre:
    - signal: EKG signál
    - r_peaks: indexy R-peakov
    - fs: vzorkovacia frekvencia
    - pre_R: čas pred R-vlnou (v sekundách)
    - post_R: čas po R-vlne (v sekundách)

    Výstup:
    - segments: pole segmentovaných úderov
    """
    pre_samples = int(pre_R * fs)  # Počet vzoriek pred R
    post_samples = int(post_R * fs)  # Počet vzoriek po R

    segments = []
    for r in r_peaks:
        if r - pre_samples >= 0 and r + post_samples < len(signal):
            segment = signal[r - pre_samples:r + post_samples]
            segments.append(segment)

    return np.array(segments)

def process_ecg_with_segmentation(record_name="100", path="./mit/", duration=5):
    raw_signal, fs, _, _ = load_mitbih_record(record_name, path)
    if raw_signal is None:
        return

    # 1. Odstránenie vysokofrekvenčného šumu
    filtered_signal = lowpass_filter(raw_signal, fs, cutoff=40)

    # 2. Odstránenie baseline driftu pomocou DWT
    filtered_signal = dwt_filtering(filtered_signal, threshold_factor=0.15)

    # 3. Z-score normalizácia
    raw_signal = normalize_zscore(raw_signal)
    filtered_signal = normalize_zscore(filtered_signal)

    # 4. Pan-Tompkins metóda na zvýraznenie QRS komplexov
    diff_signal = differentiate(filtered_signal)
    squared_signal = squaring(diff_signal)
    integrated_signal = moving_window_integration(squared_signal, window_size=10)

    # 5. Detekcia R-peakov
    detected_r_peaks = detect_r_peaks(integrated_signal, fs, threshold_factor=0.35)

    # 6. Segmentácia EKG signálu na údery
    segments = segment_heartbeats(filtered_signal, detected_r_peaks, fs)

    # 7. Vizualizácia niekoľkých segmentov
    plt.figure(figsize=(12, 6))
    for i in range(min(5, len(segments))):  # Zobrazíme prvých 5 segmentov
        plt.plot(segments[i], label=f"Segment {i+1}")

    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.title("Segmentácia srdcových úderov (zarovnané na R-vlnu)")
    plt.legend()
    plt.grid()
    plt.show()

    print(f"📊 Počet segmentov: {len(segments)}")




def extract_rr_features(r_peaks, fs):
    """
    Vytvorí pole záznamov s RR intervalmi (v milisekundách) a priemerným RR intervalom (v milisekundách).

    Parametre:
    - r_peaks: indexy R-peakov
    - fs: vzorkovacia frekvencia
    Extrahuje RR intervaly a základné HRV príznaky."""
    rr_intervals = np.diff(r_peaks) / fs  # Konverzia na sekundy

    rr_previous = np.insert(rr_intervals, 0, rr_intervals[0])  # RR₀ (predchádzajúca srdcová frekvencia)
    rr_next = np.append(rr_intervals, rr_intervals[-1])  # RR₁ (následná srdcová frekvencia)

    # HRV metriky
    sdnn = np.std(rr_intervals)  # Štandardná odchýlka RR intervalov
    rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))  # Root Mean Square of Successive Differences (RMSSD)

    # Spektrálna analýza HRV (LF/HF pomer)
    f, psd = welch(rr_intervals, fs=1/np.mean(rr_intervals), nperseg=len(rr_intervals))
    lf_power = np.sum(psd[(f >= 0.04) & (f < 0.15)])
    hf_power = np.sum(psd[(f >= 0.15) & (f < 0.4)])
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0

    return rr_previous, rr_next, sdnn, rmssd, lf_hf_ratio
def extract_wave_amplitudes(segments):
    """Extrahuje amplitúdy R, P a T vĺn z EKG segmentov."""
    r_amplitudes = np.max(segments, axis=1)  # Amplitúda R-vlny

    p_amplitudes = np.min(segments[:, :50], axis=1)  # P vlna je zvyčajne 50 vzoriek pred R
    t_amplitudes = np.max(segments[:, 100:], axis=1)  # T vlna je približne 100 vzoriek po R

    return r_amplitudes, p_amplitudes, t_amplitudes

def extract_wave_durations(r_peaks, segments, fs):
    """Presné meranie PQ, QRS, QT a ST intervalov pomocou lokálnych extrémov."""
    pq_intervals, qrs_durations, qt_intervals, st_intervals = [], [], [], []

    for i, r in enumerate(r_peaks):
        segment = segments[i]

        # Q bod je najnižší bod pred R
        q_index = np.argmin(segment[:50])
        q_time = (r - 50 + q_index) / fs

        # S bod je najnižší bod po R
        s_index = np.argmin(segment[50:100])
        s_time = (r + s_index) / fs

        # T vlna je maximum v posledných 100 ms segmentu
        t_index = np.argmax(segment[100:])
        t_time = (r + 100 + t_index) / fs

        pq_intervals.append(q_time - (r - 50) / fs)
        qrs_durations.append(s_time - q_time)
        qt_intervals.append(t_time - q_time)
        st_intervals.append(t_time - s_time)

    return np.array(pq_intervals), np.array(qrs_durations), np.array(qt_intervals), np.array(st_intervals)

def extract_derivative_features(segments):
    """Extrahuje maximálnu strmosť QRS komplexu."""
    diff_segments = np.diff(segments, axis=1)
    max_slope = np.max(diff_segments, axis=1)  # Maximálna derivácia

    return max_slope

    """Extrahuje všetky relevantné príznaky z EKG segmentov pre fuzzy klasifikáciu."""
    # Segmentácia srdcových úderov
    segments = segment_heartbeats(filtered_signal, detected_r_peaks, fs)

    # RR intervaly a HRV metriky
    rr_previous, rr_next, sdnn, rmssd, lf_hf_ratio = extract_rr_features(detected_r_peaks, fs)

    # Amplitúdy vĺn
    r_amplitudes, p_amplitudes, t_amplitudes = extract_wave_amplitudes(segments)

    # Časové intervaly medzi vlnami
    pq_intervals, qrs_durations, qt_intervals, st_intervals = extract_wave_durations(detected_r_peaks, fs)

    # Derivačné príznaky
    max_slope = extract_derivative_features(segments)

    # Uloženie príznakov do slovníka
    features = {
        "RR₀": rr_previous,
        "RR₁": rr_next,
        "SDNN": sdnn,
        "RMSSD": rmssd,
        "LF/HF Ratio": lf_hf_ratio,
        "R Amplitude": r_amplitudes,
        "P Amplitude": p_amplitudes,
        "T Amplitude": t_amplitudes,
        "PQ Interval": pq_intervals,
        "QRS Duration": qrs_durations,
        "QT Interval": qt_intervals,
        "ST Interval": st_intervals,
        "Max Slope": max_slope
    }

    return features
def extract_ecg_features(filtered_signal, detected_r_peaks, fs, segments):
    """Extrahuje všetky relevantné príznaky z EKG segmentov pre fuzzy klasifikáciu."""
    # Segmentácia srdcových úderov


    # RR intervaly a HRV metriky
    rr_previous, rr_next, sdnn, rmssd, lf_hf_ratio = extract_rr_features(detected_r_peaks, fs)

    # Amplitúdy vĺn
    r_amplitudes, p_amplitudes, t_amplitudes = extract_wave_amplitudes(segments)
    # Oprava veľkosti detegovaných R-peakov
    if len(detected_r_peaks) > len(segments):
        detected_r_peaks = detected_r_peaks[:len(segments)]
    # Časové intervaly medzi vlnami (opravené)
    pq_intervals, qrs_durations, qt_intervals, st_intervals = extract_wave_durations(detected_r_peaks, segments, fs)

    # Derivačné príznaky
    max_slope = extract_derivative_features(segments)

    # Vytvorenie slovníka príznakov
    features = {
        "RR₀": np.array(rr_previous),
        "RR₁": np.array(rr_next),
        "SDNN": np.array([sdnn]),  # Konvertované na pole
        "RMSSD": np.array([rmssd]),  # Konvertované na pole
        "LF/HF Ratio": np.array([lf_hf_ratio]),  # Konvertované na pole
        "R Amplitude": np.array(r_amplitudes),
        "P Amplitude": np.array(p_amplitudes),
        "T Amplitude": np.array(t_amplitudes),
        "PQ Interval": np.array(pq_intervals),
        "QRS Duration": np.array(qrs_durations),
        "QT Interval": np.array(qt_intervals),
        "ST Interval": np.array(st_intervals),
        "Max Slope": np.array(max_slope)
    }

    return features

def plot_segment_with_all_waves(segment, fs, r_amplitude, pq_interval, qrs_duration, p_amplitude, t_amplitude):
    """
    Vizualizuje jeden segment EKG so zaznačenými vlnami P, Q, R, S, T na základe extrahovaných časových príznakov.

    Parametre:
    - segment: Pole so vzorkami segmentu
    - fs: Vzorkovacia frekvencia EKG signálu
    - r_amplitude: Amplitúda R-vlny zo slovníka extrahovaných príznakov
    - pq_interval: Čas PQ intervalu v sekundách (na výpočet polohy Q vlny)
    - qrs_duration: Trvanie QRS komplexu v sekundách (na výpočet polohy S vlny)
    - p_amplitude: Amplitúda P-vlny zo slovníka extrahovaných príznakov
    - t_amplitude: Amplitúda T-vlny zo slovníka extrahovaných príznakov
    """

    time_axis = np.arange(len(segment)) / fs

    # Nájdeme R-peak ako najvyšší bod v segmente
    r_peak_index = np.argmax(segment)

    # Výpočet indexov Q a S na základe extrahovaných časových intervalov
    q_index = int(r_peak_index - (pq_interval * fs))  # Q vlna pred R-vlnou
    s_index = int(r_peak_index + (qrs_duration * fs))  # S vlna po R-vlne

    # P-vlna: Hľadáme lokálne maximum v PQ intervale
    p_search_range = max(0, q_index - int(0.1 * fs))
    p_index = np.argmax(segment[p_search_range:q_index]) + p_search_range

    # T-vlna: Hľadáme lokálne maximum v ST intervale
    t_search_range = min(len(segment), s_index + int(0.1 * fs))
    t_index = np.argmax(segment[s_index:t_search_range]) + s_index

    # Kontrola hraníc indexov
    q_index = max(0, q_index)
    s_index = min(len(segment) - 1, s_index)
    p_index = max(0, p_index)
    t_index = min(len(segment) - 1, t_index)

    # Vizualizácia segmentu
    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, segment, label="ECG Segment", linewidth=2)

    # Označenie vĺn
    plt.scatter(time_axis[r_peak_index], r_amplitude, color='red', label="R-wave", zorder=3)
    plt.scatter(time_axis[q_index], segment[q_index], color='purple', label="Q-wave (z PQ Interval)", zorder=3)
    plt.scatter(time_axis[s_index], segment[s_index], color='blue', label="S-wave (z QRS Duration)", zorder=3)
    plt.scatter(time_axis[p_index], p_amplitude, color='orange', label="P-wave (presná poloha)", zorder=3)
    plt.scatter(time_axis[t_index], t_amplitude, color='green', label="T-wave (presná poloha)", zorder=3)

    # Nastavenia grafu
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Segment EKG so zaznačenými P, Q, R, S, T vlnami (presné použitie príznakov)")
    plt.legend()
    plt.grid()
    plt.show()
# 4️⃣ Main Execution
if __name__ == "__main__":
    #filter_and_validate_signal()
    #detect_r_peaks_on_filtered_signal()
    #process_ecg(record_name="100", path="./mit/", duration=5)
    #process_ecg_with_comparison(record_name="100", path="./mit/", duration=5, tolerance=50)
    #process_ecg_with_segmentation(record_name="100", path="./mit/", duration=5)
    record_name = "100"
    path = "./mit/"
    raw_signal, fs, _, _ = load_mitbih_record(record_name, path)

    # 1. Odstránenie vysokofrekvenčného šumu
    filtered_signal = lowpass_filter(raw_signal, fs, cutoff=40)

    # 2. Odstránenie baseline driftu pomocou DWT
    filtered_signal = dwt_filtering(filtered_signal, threshold_factor=0.15)

    # 3. Z-score normalizácia
    raw_signal = normalize_zscore(raw_signal)
    filtered_signal = normalize_zscore(filtered_signal)

    # 4. Pan-Tompkins metóda na zvýraznenie QRS komplexov
    diff_signal = differentiate(filtered_signal)
    squared_signal = squaring(diff_signal)
    integrated_signal = moving_window_integration(squared_signal, window_size=10)

    # 5. Detekcia R-peakov
    detected_r_peaks = detect_r_peaks(integrated_signal, fs, threshold_factor=0.35)
    print(f"detected_r_peaks: {detected_r_peaks}")
    segments = segment_heartbeats(filtered_signal, detected_r_peaks, fs)
    # 6. Segmentácia EKG signálu na údery
    features = extract_ecg_features(filtered_signal, detected_r_peaks, fs, segments)
    print(f"features: {features}")
    # 7. Výpis príznakov
    for key, value in features.items():
        if isinstance(value, (np.ndarray, list)):  # Ak je pole, vypíš prvých 5 hodnôt
            print(f"{key}: {value[:5]} ...")
        else:  # Ak je to skalár, vypíš hodnotu priamo
            print(f"{key}: {value}") # Vypíšeme len prvých 5 hodnôt pre prehľadnosť

    if len(segments) > 0:
        first_segment = segments[0]  # Prvý segment

        # Načítanie príznakov z extrakcie
        r_amplitude = features["R Amplitude"][0]
        pq_interval = features["PQ Interval"][0]
        qrs_duration = features["QRS Duration"][0]
        p_amplitude = features["P Amplitude"][0]
        t_amplitude = features["T Amplitude"][0]

        # Vizualizácia segmentu s P, Q, R, S, T vlnami
        plot_segment_with_all_waves(first_segment, fs, r_amplitude, pq_interval, qrs_duration, p_amplitude, t_amplitude)
    else:
        print("❌ Neboli nájdené žiadne segmenty!")
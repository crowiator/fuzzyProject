import numpy as np
from preprocessing.load import load_mitbih_record
from preprocessing.filtering import lowpass_filter, dwt_filtering
from config import MIT_DATA_PATH
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
# Načítaj záznam
record_name = '100'  # príklad záznamu, môžeš použiť aj iné
signal, fs, _, _, _, _ = load_mitbih_record(record_name)

# Filtrácia
signal_lowpass = lowpass_filter(signal, fs)
signal_dwt = dwt_filtering(signal)
signal_filtered = dwt_filtering(signal_lowpass)

# Časová os v sekundách
duration = 5
samples = int(duration *fs)
time_axis = np.arange(samples)/fs
# Vykreslenie grafov
plt.figure(figsize=(14, 10))

# Originálny signál
plt.subplot(4, 1, 1)
plt.plot(time_axis, signal [:samples], color='black')
plt.title('Originálny EKG signál')
plt.xlabel('Čas (s)')
plt.ylabel('Amplitúda (mV)')

# Po low-pass filtrácii
plt.subplot(4, 1, 2)
plt.plot(time_axis, signal_lowpass[:samples], color='blue')
plt.title('EKG po low-pass filtrácii (Butterworth, 30 Hz)')
plt.xlabel('Čas (s)')
plt.ylabel('Amplitúda (mV)')

# Po DWT filtrácii (baseline wander)
plt.subplot(4, 1, 3)
plt.plot(time_axis, signal_dwt[:samples], color='green')
plt.title('EKG po DWT filtrácii (baseline wander removal)')
plt.xlabel('Čas (s)')
plt.ylabel('Amplitúda (mV)')

# Kompletná filtrácia (low-pass + DWT)
plt.subplot(4, 1, 4)
plt.plot(time_axis, signal_filtered[:samples], color='red')
plt.title('EKG po kompletnej filtrácii (Low-pass + DWT)')
plt.xlabel('Čas (s)')
plt.ylabel('Amplitúda (mV)')

plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def plot_fft_comparison(signal, filtered_signal, fs, record_id="unknown"):
    """
    Vykreslí porovnanie pôvodného a filtrovaného signálu:
    - v časovej doméne (výrez)
    - vo frekvenčnej doméne s frekvenciou na osi Y

    Uloží aj CSV s frekvenčnými údajmi.
    """


    # Výpočet FFT
    n = len(signal)
    frequencies = np.fft.rfftfreq(n, d=1/fs)
    fft_orig = np.abs(np.fft.rfft(signal)) / n
    fft_filtered = np.abs(np.fft.rfft(filtered_signal)) / n


    # Časová os
    t = np.arange(len(signal)) / fs

    # Vizualizácia
    plt.figure(figsize=(12, 5))

    # Časová doména
    plt.subplot(1, 2, 1)
    plt.plot(t[:1000], signal[:1000], label='Pôvodný signál')
    plt.plot(t[:1000], filtered_signal[:1000], label='Filtrovaný signál', alpha=0.75)
    plt.title("Signál v časovej doméne (výrez)")
    plt.xlabel("Čas [s]")
    plt.ylabel("Amplitúda")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_signal_comparison(raw_signal, lowpass_signal, dwt_signal, fs, record_id="unknown"):
    """
    Vykreslí porovnanie 3 signálov:
    - pôvodný signál
    - signál po lowpass filtrovaní
    - signál po dwt filtrovaní
    """


    # Časová os
    t = np.arange(len(raw_signal)) / fs

    # Výrez pre vizualizáciu (napr. prvých 1000 vzoriek)
    window = min(1000, len(raw_signal))

    # Vykreslenie
    plt.figure(figsize=(10, 5))
    plt.plot(t[:window], raw_signal[:window], label="Pôvodný signál")
    plt.plot(t[:window], lowpass_signal[:window], label="Lowpass filtrovaný", alpha=0.8)
    plt.plot(t[:window], dwt_signal[:window], label="DWT filtrovaný", alpha=0.8)
    plt.title(f"Porovnanie signálov – záznam {record_id}")
    plt.xlabel("Čas [s]")
    plt.ylabel("Amplitúda")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Uloženie obrázku
    os.makedirs("results/plots", exist_ok=True)
    plot_path = f"results/plots/signal_comparison_{record_id}.png"
    plt.savefig(plot_path)
    print(f"🖼️ Graf uložený do: {plot_path}")

    plt.show()

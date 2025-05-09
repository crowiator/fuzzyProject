# classifiers/fuzzy_classifier.py
from skfuzzy import control as ctrl, interp_membership
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
import json
import numpy as np
import skfuzzy as fuzz
from pathlib import Path
from config import MF_CFG_PATH

MF_CFG_PATH = Path(MF_CFG_PATH)  # ← cesta k tvojmu súboru
with MF_CFG_PATH.open() as f:
    MF = json.load(f)
COLORS = dict(
    normalna="#1f77b4",  # blue
    mierna="#ff7f0e",  # orange
    zavazna="#2ca02c",  # green
)
DEFAULT_LINEWIDTH = 1.8
DEFAULT_ALPHA = 0.6

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MFConfig:
    universe: tuple[float, float]
    low: list[float]
    med: list[float]
    high: list[float]


# ----------------------------------------------------------------------
# Tento skript definuje triedu FuzzyClassifier, ktorá pomocou trojvstupového
# fuzzy systému (HR, QRS, TWA) odhaduje závažnosť arytmie jedného EKG úderu.
# Vráti nielen výsledok, ale aj členstvá a zoznam najviac aktivovaných pravidiel
# pre jednoduché vysvetlenie výstupu lekárovi.
# ----------------------------------------------------------------------

# ==============================================================
# Hlavná trieda
# ==============================================================
class FuzzyClassifier:
    def __init__(self):
        self.diagnosis_ctrl, self.arr = self._init_fuzzy_system()
        self._sim: ctrl.ControlSystemSimulation | None = None

    def _init_fuzzy_system(self):
        # ------------------------------------------------------------------ #
        # 1) Vstupné a výstupné premenné –  univ. berieme z mf_params_all.json
        # ------------------------------------------------------------------ #
        # HR (bpm)
        self.hr = ctrl.Antecedent(
            np.linspace(*MF["HR_bpm"]["universe"], 200), 'HR')

        # QRS (ms)
        self.qrs = ctrl.Antecedent(
            np.linspace(*MF["QRS_ms"]["universe"], 200), 'QRS')

        # T‑wave amplitude (mV – absolútna hodnota)
        self.twa = ctrl.Antecedent(
            np.linspace(*MF["T_amp"]["universe"], 200), 'TWA')

        # Výstupná premenná (0–1 závažnosť arytmie)
        self.arr = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'Arrhythmia')

        # ------------------------------------------------------------------ #
        # 2) Členovacie funkcie vytvorené z JSON (LOW / MED / HIGH)
        # ------------------------------------------------------------------ #
        # ---------- HR ----------
        self.hr['nizka'] = fuzz.trapmf(self.hr.universe, MF["HR_bpm"]["low"])
        self.hr['normalna'] = fuzz.trimf(self.hr.universe, MF["HR_bpm"]["med"])
        self.hr['vysoka'] = fuzz.trapmf(self.hr.universe, MF["HR_bpm"]["high"])

        # ---------- QRS ----------
        self.qrs['kratky'] = fuzz.trapmf(self.qrs.universe, MF["QRS_ms"]["low"])
        self.qrs['normalny'] = fuzz.trimf(self.qrs.universe, MF["QRS_ms"]["med"])
        self.qrs['dlhy'] = fuzz.trapmf(self.qrs.universe, MF["QRS_ms"]["high"])

        # ---------- T-wave amplitude ----------
        self.twa['nizka'] = fuzz.trapmf(self.twa.universe, MF["T_amp"]["low"])
        self.twa['normalna'] = fuzz.trimf(self.twa.universe, MF["T_amp"]["med"])
        self.twa['vysoka'] = fuzz.trapmf(self.twa.universe, MF["T_amp"]["high"])

        """
        self.hr = ctrl.Antecedent(np.arange(20, 221, 1), 'HR')  # bpm
        self.qrs = ctrl.Antecedent(np.arange(40, 401, 1), 'QRS')  # ms
        self.twa = ctrl.Antecedent(np.arange(0.0, 1.21, 0.01), 'TWA')  # mV (absolútna hodnota)

        self.arr = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'Arrhythmia')

        # --- Definícia trojuholníkových členstiev (trimf) -----------------
        # ---------- HR ----------
        self.hr['nizka'] = fuzz.trimf(self.hr.universe, [40, 50, 60])
        self.hr['normalna'] = fuzz.trimf(self.hr.universe, [55, 70, 85])
        self.hr['vysoka'] = fuzz.trimf(self.hr.universe, [80, 90, 120])

        # ---------- QRS ----------
        self.qrs['kratky'] = fuzz.trimf(self.qrs.universe, [50, 60, 70])
        self.qrs['normalny'] = fuzz.trimf(self.qrs.universe, [65, 80, 100])
        self.qrs['dlhy'] = fuzz.trimf(self.qrs.universe, [90, 100, 120])

        # ---------- T-wave ----------
        self.twa['nizka'] = fuzz.trimf(self.twa.universe, [0.0, 0.1, 0.2])
        self.twa['normalna'] = fuzz.trimf(self.twa.universe, [0.15, 0.3, 0.45])
        self.twa['vysoka'] = fuzz.trimf(self.twa.universe, [0.6, 0.9, 1.2])

        # ---------- VÝSTUP ----------
        #self.arr['normalna'] = fuzz.trimf(self.arr.universe, [0.0, 0.2, 0.4])
        #self.arr['mierna'] = fuzz.trimf(self.arr.universe, [0.35, 0.55, 0.75])
        #self.arr['zavazna'] = fuzz.trimf(self.arr.universe, [0.7, 0.85, 1.0])
        """
        self.arr['normalna'] = fuzz.trimf(self.arr.universe, [0.0, 0.2, 0.4])
        self.arr['mierna'] = fuzz.trimf(self.arr.universe, [0.4, 0.6, 0.8])
        self.arr['zavazna'] = fuzz.trimf(self.arr.universe, [0.8, 0.93, 1.0])
        """

                        
                        
                        
                        ('nizka', 'kratky', 'nizka', 'normalna'),
                ('nizka', 'kratky', 'normalna', 'normalna'),
                ('nizka', 'kratky', 'vysoka', 'mierna'),
                ('nizka', 'normalny', 'nizka', 'normalna'),
                ('nizka', 'normalny', 'normalna', 'mierna'),
                ('nizka', 'normalny', 'vysoka', 'mierna'),
                ('nizka', 'dlhy', 'nizka', 'mierna'),
                ('nizka', 'dlhy', 'normalna', 'zavazna'),
                ('nizka', 'dlhy', 'vysoka', 'zavazna'),
                ('normalna', 'kratky', 'nizka', 'normalna'),
                ('normalna', 'kratky', 'normalna', 'normalna'),
                ('normalna', 'kratky', 'vysoka', 'mierna'),
                ('normalna', 'normalny', 'nizka', 'normalna'),
                ('normalna', 'normalny', 'normalna', 'normalna'),
                ('normalna', 'normalny', 'vysoka', 'mierna'),
                ('normalna', 'dlhy', 'nizka', 'mierna'),
                ('normalna', 'dlhy', 'normalna', 'zavazna'),
                ('normalna', 'dlhy', 'vysoka', 'zavazna'),
                ('vysoka', 'kratky', 'nizka', 'mierna'),
                ('vysoka', 'kratky', 'normalna', 'mierna'),
                ('vysoka', 'kratky', 'vysoka', 'zavazna'),
                ('vysoka', 'normalny', 'nizka', 'mierna'),
                ('vysoka', 'normalny', 'normalna', 'zavazna'),
                ('vysoka', 'normalny', 'vysoka', 'zavazna'),
                ('vysoka', 'dlhy', 'nizka', 'zavazna'),
                ('vysoka', 'dlhy', 'normalna', 'zavazna'),
                ('vysoka', 'dlhy', 'vysoka', 'zavazna'),

                        """
        # --- Pravidlá vo forme IF‑THEN -----------------------------------
        # Každý tuple (hr, q, t, s) opisuje textové označenia MF.
        rules = [
            ctrl.Rule(self.hr[hr] & self.qrs[q] & self.twa[t], self.arr[s],
                      label=f"HR {hr}  &  QRS {q}  &  TWA {t}  →  arytmia {s}")
            for hr, q, t, s in [

                ('nizka', 'kratky', 'nizka', 'normalna'),
                ('nizka', 'kratky', 'normalna', 'normalna'),
                ('nizka', 'kratky', 'vysoka', 'mierna'),
                ('nizka', 'normalny', 'nizka', 'normalna'),
                ('nizka', 'normalny', 'normalna', 'mierna'),
                ('nizka', 'normalny', 'vysoka', 'mierna'),
                ('nizka', 'dlhy', 'nizka', 'mierna'),
                ('nizka', 'dlhy', 'normalna', 'normalna'),
                ('nizka', 'dlhy', 'vysoka', 'zavazna'),
                ('normalna', 'kratky', 'nizka', 'normalna'),
                ('normalna', 'kratky', 'normalna', 'normalna'),
                ('normalna', 'kratky', 'vysoka', 'mierna'),
                ('normalna', 'normalny', 'nizka', 'normalna'),
                ('normalna', 'normalny', 'normalna', 'normalna'),
                ('normalna', 'normalny', 'vysoka', 'mierna'),
                ('normalna', 'dlhy', 'nizka', 'normalna'),
                ('normalna', 'dlhy', 'normalna', 'normalna'),
                ('normalna', 'dlhy', 'vysoka', 'zavazna'),
                ('vysoka', 'kratky', 'nizka', 'normalna'),
                ('vysoka', 'kratky', 'normalna', 'mierna'),
                ('vysoka', 'kratky', 'vysoka', 'zavazna'),
                # ('vysoka', 'normalny', 'nizka', 'normalna'),
                ('vysoka', 'normalny', 'nizka', 'mierna'),
                ('vysoka', 'normalny', 'normalna', 'zavazna'),
                ('vysoka', 'normalny', 'vysoka', 'zavazna'),
                ('vysoka', 'dlhy', 'nizka', 'zavazna'),  # normalna povodne
                ('vysoka', 'dlhy', 'normalna', 'normalna'),
                ('vysoka', 'dlhy', 'vysoka', 'zavazna'),

            ]
        ]
        rules.append(
            ctrl.Rule(self.twa['vysoka'] & self.qrs['dlhy'], self.arr['mierna'],
                      label="QRS dlhy & TWA veľmi vysoká → mierna")

        )
        rules.append(
            ctrl.Rule((self.hr['vysoka'] | self.hr['normalna']) &
                      self.qrs['dlhy'] & self.twa['normalna'],
                      self.arr['zavazna'],
                      label="QRS dlhy & TWA normalna → zavazna")
        )
        rules.append(ctrl.Rule(self.hr['vysoka'] & self.qrs['dlhy'], self.arr['mierna']))
        # Vrátime skonfigurovaný riadiaci systém a samotný consequent(arytmiu) pre ďalšie výpočty.
        diagnosis_ctrl = ctrl.ControlSystem(rules)  # objekt ControlSystem (celý fuzzy systém),
        return diagnosis_ctrl, self.arr  # výstupnú premennú Arrhythmia (

    def _get_sim(self):
        if self._sim is None:
            self._sim = ctrl.ControlSystemSimulation(self.diagnosis_ctrl)
        else:
            self._sim.reset()
        return self._sim

    # --------------------------------------------------------------
    # Metóda predict()
    #  - vstup  : čísla HR (bpm), QRS (ms), TWA (mV)
    #  - výstup : slovník s centroidom, slovným labelom, µ členstvami
    #             a zoznamom najviac aktivovaných pravidiel
    # --------------------------------------------------------------
    def predict(self, hr, qrs, twa, top_n: int = 27):

        diagnosis_sim = self._get_sim()
        # Vytvoríme simulátor pre náš fuzzy systém; každý výpočet
        # (tj. každý úder) má vlastnú inštanciu.
        # 1) Vložíme vstupy do simulácie a spustíme inferenciu.
        for name, val, anteced in [('HR', hr, self.hr),
                                   ('QRS', qrs, self.qrs),
                                   ('TWA', twa, self.twa)]:
            # Ak je vstup NaN, nahradíme ho hodnotou tesne pod minimom
            # univerza – takýto bod do MF nepatrí a nijako neovplyvní inferenciu.
            diagnosis_sim.input[name] = np.nan_to_num(val, nan=anteced.universe.min() - 1)
        # Spustíme inferenčný mechanizmus (pravidlá + agregácia) a
        # defuzifikáciu (centroid metóda). Výsledok sa uloží do
        # diagnosis_sim.output.
        # ------------------------------------------------------------------
        # Spustíme samotnú inferenciu (vyhodnotenie pravidiel + agregáciu) a
        # defuzzifikáciu.  Po zavolaní `compute()` sa do `diagnosis_sim.output`
        # zapíše kľúč "Arrhythmia" s crisp hodnotou výslednej fuzzy množiny.
        diagnosis_sim.compute()
        # NOTE: V niektorých okrajových prípadoch (napr. chýbajúce alebo NaN
        # vstupy mimo definovaných intervalov) sa môže stať, že scikit‑fuzzy
        # nedokáže vyhodnotiť žiadne pravidlo a kľúč "Arrhythmia" v outpute
        # vôbec nevytvorí.  Ošetrujeme to blokom nižšie.
        if 'Arrhythmia' not in diagnosis_sim.output:
            # Ak kľúč chýba, vraciame štruktúru s značkou „invalid“ a nulovými
            # členstvami, aby ďalší kód nespadol na KeyError.
            return {
                "centroid": np.nan,
                "label": "invalid",
                "memberships": {k: 0.0 for k in self.arr.terms},
                "rules": []
            }
        # Defuzifikovaný skalar (centroid) výstupnej množiny.
        out_val = diagnosis_sim.output['Arrhythmia']

        # 2) Defuzifikovaný centroid prevodíme na členstvá (µ) pre každú výstupnú triedu.
        # Pre každú výstupnú triedu vypočítame, ako veľmi jej centroid zodpovedá
        # (klassická reverse‑lookup pomocou interp_membership).
        μ_out = {lbl: fuzz.interp_membership(self.arr.universe,
                                             self.arr[lbl].mf,
                                             out_val)
                 for lbl in self.arr.terms}

        if sum(μ_out.values()) == 0:
            # zabezpečte, že sa *vždy* vracia dict v rovnakej štruktúre
            return {
                "centroid": np.nan,
                "label": "invalid",
                "conf": 0.0,
                "memberships": {k: 0.0 for k in self.arr.terms},
                "rules": []
            }

        # 3) Zhromaždíme µ každého pravidla.
        #    Rôzne verzie scikit‑fuzzy ukladajú silu inak, preto
        #    ošetrujeme 0.5.x (aggregate_firing) aj 0.4.x (firing_strength).
        fired = []
        for rule in self.diagnosis_ctrl.rules:

            # 0.5.x ────────────────────────────────────────────────
            if hasattr(rule, "aggregate_firing"):
                try:
                    strength = float(rule.aggregate_firing[diagnosis_sim])
                except (KeyError, TypeError):
                    strength = 0.0

            # 0.4.x ────────────────────────────────────────────────
            elif hasattr(rule, "firing_strength"):
                strength = float(rule.firing_strength)

            # fallback ─────────────────────────────────────────────
            else:
                strength = 0.0

            if strength > 0:
                fired.append({"rule": rule.label, "μ": round(strength, 3)})

        fired = sorted(fired, key=lambda d: d["μ"], reverse=True)[:top_n]

        # —— 1) zober triedu s najvyšším μ ————————————————
        label = max(μ_out, key=μ_out.get)  # 'normalna' / 'mierna' / 'zavazna'
        confidence = μ_out[label]

        # —— 3) vráť výsledok ————————————————————————————————
        return {
            "centroid": round(float(out_val), 3),
            "label": label,
            "conf": round(float(confidence), 3),
            "memberships": {k: round(float(v), 3) for k, v in μ_out.items()},
            "rules": fired
        }

    # tiež do FuzzyClassifier

    # --------------------------------------------------------------
    # Pomocné vizualizačné metódy
    # --------------------------------------------------------------
    def plot_membership_functions(self):
        """
        Vykreslí všetky vstupné a výstupné fuzzy množiny
        (HR, QRS, TWA, Arrhythmia) do jedného 2×2 grafu.
        """
        plt.figure(figsize=(10, 6))

        vars_and_titles = [
            (self.hr, "Srdcová frekvencia (HR) [bpm]"),
            (self.qrs, "QRS interval [ms]"),
            (self.twa, "Amplitúda T‑vlny (|mV|)"),
            (self.arr, "Arytmia (výstup)")
        ]

        for i, (var, title) in enumerate(vars_and_titles, 1):
            plt.subplot(2, 2, i)
            for term, mf in var.terms.items():
                plt.plot(var.universe, mf.mf, label=term)
            plt.title(title)
            plt.xlabel("Hodnota")
            plt.ylabel("μ")
            plt.ylim(0, 1.05)
            plt.grid(True)
            if i == 4:
                plt.legend(loc="upper center", ncol=3, fontsize="small")
        plt.tight_layout()
        plt.show()

    def demo_inference(self, hr, qrs, twa, top_n: int = 5):
        """
        Ukážka krokov Mamdaniho inferencie pre konkrétne vstupy.
        Vypíše tabuľku členstiev, aktivované pravidlá a zobrazí
        graf MFs s vyznačením vstupných hodnôt a crisp výstupu.
        """
        res = self.predict(hr, qrs, twa, top_n=top_n)

        # --- textový výstup ------------------------------------------------
        print("\n=== DEMO Mamdani ===")
        print(f"Vstupy: HR={hr} bpm, QRS={qrs} ms, TWA={twa} mV")
        print(f"Crisp výstup (centroid): {res['centroid']:.3f}  →  {res['label']}")
        print("Členstvá výstupu:", res["memberships"])
        print("Top pravidlá:")
        for r in res["rules"]:
            print(f"  μ={r['μ']:.3f} → {r['rule']}")

        # --- výpočet agregovanej fuzzy množiny (MAX z orezaných MF) -----
        universe = self.arr.universe
        aggregated = np.zeros_like(universe, dtype=float)

        # Každé pravidlo priradí klipovanú MF jednej výstupnej triede
        for r in res["rules"]:
            term_name = r["rule"].split("→")[-1].split()[-1]  # z labelu vyparsuj 'normalna'/'mierna'/'zavazna'
            term_name = term_name.strip()
            if term_name not in self.arr.terms:
                continue
            mf_vals = self.arr[term_name].mf
            clipped = np.minimum(mf_vals, r["μ"])
            aggregated = np.maximum(aggregated, clipped)

        # --- graf: agregovaná množina + centroid -------------------------
        plt.figure(figsize=(6, 3))

        # 1) Pôvodné MF (čiarkované, priesvitné)
        for term, mf in self.arr.terms.items():
            plt.plot(universe, mf.mf,
                     color=COLORS.get(term, None),
                     linestyle="--", linewidth=1.2, alpha=0.45)

        # 2) Klipované MF z každého aktivovaného pravidla
        for r in res["rules"]:
            term_name = r["rule"].split()[-1]  # normalna / mierna / zavazna
            if term_name not in self.arr.terms:
                continue
            mf_vals = self.arr[term_name].mf
            clipped = np.minimum(mf_vals, r["μ"])
            plt.fill_between(universe, 0, clipped,
                             facecolor=COLORS.get(term_name, None),
                             alpha=DEFAULT_ALPHA)

        # 3) Centroid
        plt.axvline(res["centroid"], color="k", linestyle="--",
                    linewidth=1.5, label=f"Centroid = {res['centroid']:.3f}")

        plt.title("Agregovaná fuzzy množina + centroid")
        plt.xlabel("Hodnota arytmie")
        plt.ylabel("μ")
        plt.ylim(0, 1.05)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # --- grafické zobrazenie ------------------------------------------
        # 1) MFs + vstupné body
        self.plot_membership_functions()

        # 2) Výstupné MF + centroid
        plt.figure(figsize=(6, 3))
        for term, mf in self.arr.terms.items():
            plt.plot(self.arr.universe, mf.mf, label=term)
        plt.axvline(res["centroid"], color="k", linestyle="--",
                    label=f"Centroid = {res['centroid']:.3f}")
        plt.title("Výstupné množiny + centroid")
        plt.xlabel("Hodnota arytmie")
        plt.ylabel("μ")
        plt.ylim(0, 1.05)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------------
    # Vizualizácia jednej premennej so zvýraznením konkrétnej hodnoty
    # --------------------------------------------------------------
    def plot_single_membership(self, var, value, title=None, xlabel=None):
        """
        var   : ctrl.Antecedent alebo Consequent (napr. self.qrs)
        value : hodnota, ktorú chceš vyznačiť (napr. QRS = 95 ms)
        title : voliteľný titulok grafu
        xlabel: popis osi x
        """
        if title is None:
            title = f"Stupeň príslušnosti pre {var.label}"
        if xlabel is None:
            xlabel = var.label

        universe = var.universe
        plt.figure(figsize=(6, 3))

        for term, mf_obj in var.terms.items():
            mf = mf_obj.mf
            μ = fuzz.interp_membership(universe, mf, value)

            color = COLORS.get(term, None)
            # samotná krivka (plná čiara, hrubšia)
            plt.plot(universe, mf, color=color, linewidth=DEFAULT_LINEWIDTH, label=term)

            # vyfarbenie len po úroveň µ
            plt.fill_between(universe, 0, np.minimum(mf, μ),
                             facecolor=color, alpha=DEFAULT_ALPHA)

            # bod v mieste (value, μ)
            plt.plot([value], [μ], "ko")

        plt.axvline(value, color="k", linestyle="--", linewidth=1.3)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Membership")
        plt.ylim(0, 1.05)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------------
    # Pomocná funkcia: univerzálne kreslenie jednej premennej
    # --------------------------------------------------------------

    @staticmethod
    def plot_membership_function(clf,
                                 var_name: str,
                                 value: float,
                                 *,
                                 title: str | None = None,
                                 xlabel: str | None = None):
        """
        clf      : inštancia FuzzyClassifier
        var_name : 'hr', 'qrs', 'twa' alebo 'arr'
        value    : číslo, ktoré chceš vyznačiť
        title    : voliteľný titulok grafu
        xlabel   : voliteľný popis osi x
        """
        # mapovanie reťazec -> atribút triedy
        var = getattr(clf, var_name.lower())
        if title is None:
            title = f"Stupeň príslušnosti pre {var.label.upper()}"
        if xlabel is None:
            xlabel = var.label.upper()

        # využijeme metódu definovanú v triede
        clf.plot_single_membership(var=var, value=value,
                                   title=title, xlabel=xlabel)

    # classifiers/fuzzy_classifier.py  (dovnútra FuzzyClassifier)
    def set_twa_mf(self, low: float, high: float) -> None:
        """
        Rýchlo prepíše MF pre TWA ('normalna', 'vysoka').
            low  – hrana medzi 'normalna' a 'vysoka' (≈ 0.4–0.6)
            high – maximum trojuholníka pre 'vysoka'
        """
        # stále zachovaj 3 termy, len posúvame uzly!
        self.twa['normalna'].mf = fuzz.trimf(self.twa.universe,
                                             [0.15, (0.15 + low) / 2, low])
        self.twa['vysoka'].mf = fuzz.trimf(self.twa.universe,
                                           [low, (low + high) / 2, high])


if __name__ == "__main__":
    clf = FuzzyClassifier()
    clf.demo_inference(hr=120, qrs=180, twa=0.55, top_n=3)
    """ 
    print(f"\nVýstup = {res['centroid']:.3f}  →  {res['label']}")
    print("Členstvá:",
          "  ".join(f"{k}: {v:.2f}" for k, v in res['memberships'].items()))

    print("Top pravidlá:")
    for r in res["rules"]:
        print(f"  μ = {r['μ']:.3f}  →  {r['rule']}")
    
    # ------------------------------------------------------------------
    # Demo vizualizácie Mamdaniho inferencie pre hodnoty z textu

    clf.plot_membership_function(clf, "hr", hr_val,
                                 title="Stupeň príslušnosti pre srdcovú frekvenciu (HR)",
                                 xlabel="HR [bpm]")

    clf.plot_membership_function(clf, "qrs", qrs_val,
                                 title="Stupeň príslušnosti pre QRS interval (QRS)",
                                 xlabel="QRS [ms]")

    clf.plot_membership_function(clf, "twa", twa_val,
                                 title="Stupeň príslušnosti pre amplitúdu T-vlny (TWA)",
                                 xlabel="T-wave |mV|")

    # ak chceš aj výstupnú premennú
    clf.plot_membership_function(clf, "arr", res["centroid"],
                                 title="Výstupné fuzzy množiny + centroid",
                                 xlabel="Arytmia (0–1)")

    clf.demo_inference(hr=55, qrs=95, twa=0.5, top_n=2)
    """

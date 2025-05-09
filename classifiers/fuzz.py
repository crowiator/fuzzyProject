# classifiers/fuzzy_classifier.py
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl, interp_membership
import pandas as pd
from collections import Counter
from preprocessing.annotation_mapping import ANNOTATION_TO_FUZZY, EXCLUDED_ANN
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
COLORS = dict(
    normalna="#1f77b4",  # blue
    mierna="#ff7f0e",  # orange
    zavazna="#2ca02c",  # green
)
DEFAULT_LINEWIDTH = 1.8
DEFAULT_ALPHA = 0.6


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
    def __init__(self, cost_sensitive: bool = True,
                 class_weight: dict[str, float] | None = None):
        """
        cost_sensitive – ak True, pri výbere labelu sa použije µ_c · w_c
        class_weight   – voliteľne vlastné váhy; ak None dopočítajú sa vo fit()
        """
        self.cost_sensitive = cost_sensitive
        self.class_weight = class_weight or {}
        self.diagnosis_ctrl, self.arr = self._init_fuzzy_system()
        self._sim: ctrl.ControlSystemSimulation | None = None
    def _init_fuzzy_system(self):
        # Nastavíme univerzá pre tri vstupy (HR, QRS, TWA) a jeden výstup (Arrhythmia).
        # Hodnoty sú vybraté tak, aby pokrývali typické fyziologické rozsahy.
        # uprava
        # ------------------------------------------------------------------ #
        # 1) Vstupné a výstupné premenné (univerzá)
        # ------------------------------------------------------------------ #
        """
           self.hr = ctrl.Antecedent(np.arange(20, 221, 1), 'HR')  # bpm
        self.qrs = ctrl.Antecedent(np.arange(40, 401, 1), 'QRS')  # ms
        self.twa = ctrl.Antecedent(np.arange(0.0, 1.21, 0.01), 'TWA')  # mV (absolútna hodnota)

        self.arr = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'Arrhythmia')

        # --- Definícia trojuholníkových členstiev (trimf) -----------------
        # ---------- HR ----------
        self.hr['nizka'] = fuzz.trimf(self.hr.universe, [20, 45, 65])
        self.hr['normalna'] = fuzz.trimf(self.hr.universe, [60, 80, 100])
        self.hr['vysoka'] = fuzz.trimf(self.hr.universe, [90, 130, 220])

        # ---------- QRS ----------
        self.qrs['kratky'] = fuzz.trimf(self.qrs.universe, [40, 55, 85])
        self.qrs['normalny'] = fuzz.trimf(self.qrs.universe, [80, 140, 210])
        self.qrs['dlhy'] = fuzz.trimf(self.qrs.universe, [200, 250, 400])

        # ---------- T-wave ----------
        self.twa['nizka'] = fuzz.trimf(self.twa.universe, [-0.30, 0.00, 0.20])
        self.twa['normalna'] = fuzz.trimf(self.twa.universe, [0.15, 0.30, 0.50])
        self.twa['vysoka'] = fuzz.trimf(self.twa.universe, [0.45, 0.70, 1.00])

        :return:
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
        # self.arr['normalna'] = fuzz.trimf(self.arr.universe, [0.0, 0.2, 0.4])
        # self.arr['mierna'] = fuzz.trimf(self.arr.universe, [0.35, 0.55, 0.75])
        # self.arr['zavazna'] = fuzz.trimf(self.arr.universe, [0.7, 0.85, 1.0])
        # self.arr['normalna'] = fuzz.trimf(self.arr.universe, [0.0, 0.20, 0.40])
        # self.arr['mierna'] = fuzz.trimf(self.arr.universe, [0.35, 0.55, 0.75])
        # self.arr['mierna'] = fuzz.trimf(self.arr.universe, [0.27, 0.55, 0.78])
        # self.arr['mierna'] = fuzz.trimf(self.arr.universe, [0.27, 0.50, 0.68])
        # self.arr['zavazna'] = fuzz.trimf(self.arr.universe, [0.70, 0.85, 1.00])
        # self.arr['zavazna'] = fuzz.trimf(self.arr.universe, [0.58, 0.78, 1.00])
        """
        self.arr['normalna'] = fuzz.trimf(self.arr.universe, [0.0, 0.25, 0.45])
        self.arr['mierna'] = fuzz.trimf(self.arr.universe, [0.45, 0.60, 0.75])
        self.arr['zavazna'] = fuzz.trimf(self.arr.universe, [0.70, 0.85, 1.00])
        """
        self.arr['normalna'] = fuzz.trimf(self.arr.universe, [0.0, 0.25, 0.5])
        self.arr['mierna'] = fuzz.trimf(self.arr.universe, [0.50, 0.70, 0.85])
        self.arr['zavazna'] = fuzz.trimf(self.arr.universe, [0.75, 0.90, 1.00])

        """
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
                        #('vysoka', 'normalny', 'nizka', 'normalna'),
                        ('vysoka', 'normalny', 'nizka', 'mierna'),
                        ('vysoka', 'normalny', 'normalna', 'zavazna'),
                        ('vysoka', 'normalny', 'vysoka', 'zavazna'),
                        ('vysoka', 'dlhy', 'nizka', 'zavazna'), #normalna povodne
                        ('vysoka', 'dlhy', 'normalna', 'normalna'),
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

    def fit(self, y: np.ndarray):
        """
        y – 1-D array textových labelov ('normalna','mierna','zavazna')
        nič netrénuje, len nastaví váhy pre cost-sensitive voľbu
        """
        if not self.cost_sensitive or self.class_weight:
            return self
        uniq, cnt = np.unique(y, return_counts=True)
        total = len(y)
        self.class_weight = {c: total / n for c, n in zip(uniq, cnt)}
        return self

    # ----------------------------------------------
    # pomocná metóda – vyberie výstupný label
    def _select_label(self, mu: dict[str, float]) -> tuple[str, float]:
        """Vracia (label, confidence) – confidence = pôvodné µ"""
        if not self.cost_sensitive or not self.class_weight:
            lab = max(mu, key=mu.get)
            return lab, mu[lab]
        lab = max(mu, key=lambda c: mu[c] * self.class_weight.get(c, 1.0))
        return lab, mu[lab]

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
            return np.nan, "Invalid", μ_out



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

        label, confidence = self._select_label(μ_out)

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

    # --------------------------------------------------------------
    # Vektorová inferencia: pole tvaru (n_samples, 3)  →  array labelov
    # --------------------------------------------------------------
    def predict_many(self, X: np.ndarray, top_n: int = 0) -> np.ndarray:
        """
        X – 2-D ndarray [HR, QRS, TWA] po riadkoch
        vracia 1-D ndarray textových labelov
        """
        labels = []
        for hr, qrs, twa in X:
            out = self.predict(float(hr), float(qrs), float(twa), top_n=top_n)
            labels.append(out["label"])
        return np.array(labels, dtype=str)




if __name__ == "__main__":
    # ---------- 1) načítaj črty a anotácie ---------------------------------
    df = pd.read_csv("../exported_features/ecg_104_features_full.csv")  # HR, QRS, TWA, Annotation
    print(df)
    # mapuj MIT-BIH symbol -> normalna/mierna/zavazna
    df["severity"] = df["Annotation"].apply(
        lambda s: ANNOTATION_TO_FUZZY.get(s, "normalna")
        if s not in EXCLUDED_ANN else "normalna"
    )
    print(df)
    # ---------- 2) spočítaj váhy w_c = 1/π_c --------------------------------
    cnt = Counter(df["severity"])
    total = len(df)
    class_weight = {c: round((total / n) ** 0.5, 2) for c, n in cnt.items()}
    print("Počty tried:", cnt)
    print("Váhy w_c   :", class_weight)

    # ---------- 3) inicializuj fuzzyho -------------------------------------
    clf = FuzzyClassifier(cost_sensitive=True, class_weight=class_weight)

    # ---------- 4) dávková predikcia ---------------------------------------
    X = df[["HR_bpm", "QRS_ms", "T_amp"]].values
    df["Prediction"] = clf.predict_many(X)

    # ---------- 5) rýchly report a uloženie --------------------------------
    print("\n=== Classification report (record 104) ===")
    print(classification_report(df["severity"], df["Prediction"], digits=3))
    print("Confusion matrix:\n", confusion_matrix(df["severity"], df["Prediction"]))

    df.to_csv("ecg_104_pred.csv", index=False)
    print("\nVýstup uložený v  ecg_104_pred.csv")
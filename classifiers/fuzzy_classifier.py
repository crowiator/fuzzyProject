# classifiers/fuzzy_classifier.py
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl, interp_membership

class FuzzyClassifier:
    def __init__(self):
        self.diagnosis_ctrl, self.arrhythmia = self._init_fuzzy_system()

    def _init_fuzzy_system(self):
        hr = ctrl.Antecedent(np.arange(30, 160, 1), 'HR')
        qrs = ctrl.Antecedent(np.arange(40, 200, 1), 'QRS')
        twa = ctrl.Antecedent(np.arange(0.0, 1.2, 0.01), 'TWA')
        arrhythmia = ctrl.Consequent(np.arange(-0.1, 1.1, 0.01), 'Arrhythmia')

        hr['low'] = fuzz.trimf(hr.universe, [40, 50, 60])
        hr['normal'] = fuzz.trimf(hr.universe, [55, 70, 85])
        hr['high'] = fuzz.trimf(hr.universe, [80, 90, 120])

        qrs['short'] = fuzz.trimf(qrs.universe, [50, 60, 70])
        qrs['normal'] = fuzz.trimf(qrs.universe, [65, 80, 100])
        qrs['long'] = fuzz.trimf(qrs.universe, [90, 100, 120])

        twa['low'] = fuzz.trimf(twa.universe, [0.0, 0.1, 0.2])
        twa['normal'] = fuzz.trimf(twa.universe, [0.15, 0.3, 0.45])
        twa['high'] = fuzz.trimf(twa.universe, [0.4, 0.5, 0.6])

        arrhythmia['normal'] = fuzz.trimf(arrhythmia.universe, [0.0, 0.25, 0.5])
        arrhythmia['moderate'] = fuzz.trimf(arrhythmia.universe, [0.3, 0.5, 0.7])
        arrhythmia['severe'] = fuzz.trimf(arrhythmia.universe, [0.6, 0.75, 1.0])

        rules = [
            ctrl.Rule(hr[hr_level] & qrs[qrs_level] & twa[twa_level], arrhythmia[severity])
            for hr_level, qrs_level, twa_level, severity in [
                ('low', 'short', 'low', 'normal'),
                ('low', 'short', 'normal', 'normal'),
                ('low', 'short', 'high', 'moderate'),
                ('low', 'normal', 'low', 'normal'),
                ('low', 'normal', 'normal', 'moderate'),
                ('low', 'normal', 'high', 'moderate'),
                ('low', 'long', 'low', 'moderate'),
                ('low', 'long', 'normal', 'severe'),
                ('low', 'long', 'high', 'severe'),
                ('normal', 'short', 'low', 'normal'),
                ('normal', 'short', 'normal', 'normal'),
                ('normal', 'short', 'high', 'moderate'),
                ('normal', 'normal', 'low', 'normal'),
                ('normal', 'normal', 'normal', 'normal'),
                ('normal', 'normal', 'high', 'moderate'),
                ('normal', 'long', 'low', 'moderate'),
                ('normal', 'long', 'normal', 'severe'),
                ('normal', 'long', 'high', 'severe'),
                ('high', 'short', 'low', 'moderate'),
                ('high', 'short', 'normal', 'moderate'),
                ('high', 'short', 'high', 'severe'),
                ('high', 'normal', 'low', 'moderate'),
                ('high', 'normal', 'normal', 'severe'),
                ('high', 'normal', 'high', 'severe'),
                ('high', 'long', 'low', 'severe'),
                ('high', 'long', 'normal', 'severe'),
                ('high', 'long', 'high', 'severe'),
            ]
        ]

        diagnosis_ctrl = ctrl.ControlSystem(rules)
        return diagnosis_ctrl, arrhythmia

    def is_valid(self, hr, qrs, twa):
        # presne podľa pôvodného článku
        return (30 <= hr <= 120) and (50 <= qrs <= 120) and (0.0 <= twa <= 0.6)

    def predict(self, hr, qrs, twa):
        if not self.is_valid(hr, qrs, twa):
            # explicitné označenie mimo rozsahu hodnôt ako Invalid
            return np.nan, "Invalid", {"normal": 0, "moderate": 0, "severe": 0}

        sim = ctrl.ControlSystemSimulation(self.diagnosis_ctrl)
        sim.input['HR'] = hr
        sim.input['QRS'] = qrs
        sim.input['TWA'] = twa
        sim.compute()
        if 'Arrhythmia' not in sim.output:
            return np.nan, "Invalid", {"normal": 0, "moderate": 0, "severe": 0}

        output_val = sim.output['Arrhythmia']

        memberships = {
            label: interp_membership(self.arrhythmia.universe, self.arrhythmia[label].mf, output_val)
            for label in ['normal', 'moderate', 'severe']
        }

        category = max(memberships, key=memberships.get).capitalize()

        return output_val, category, memberships
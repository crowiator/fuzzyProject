import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from skfuzzy import interp_membership


# Definovanie fuzzy premenných pre vstupy
#https://www.jetir.org/papers/JETIR2411585.pdf#:~:text=Traditional%20diagnostic%20methods%20often%20rely,Fuzzy%20inference%20systems
hr = ctrl.Antecedent(np.arange(30, 160, 1), 'HR')        # Heart Rate (bpm) 40–120
qrs = ctrl.Antecedent(np.arange(40, 200, 1), 'QRS')      # QRS interval (ms)
twa = ctrl.Antecedent(np.arange(0.0, 1.2, 0.01), 'TWA')  # T-wave amplitude (mV)

# 2. Výstupná fuzzy premenná
arrhythmia = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'Arrhythmia')

# 3. Fuzzy množiny
hr['low'] = fuzz.trimf(hr.universe, [40, 50, 60])
hr['normal'] = fuzz.trimf(hr.universe, [55, 70, 85])
hr['high'] = fuzz.trimf(hr.universe, [80, 90, 130])

qrs['short'] = fuzz.trimf(qrs.universe, [50, 60, 70])
qrs['normal'] = fuzz.trimf(qrs.universe, [65, 80, 100])
qrs['long'] = fuzz.trimf(qrs.universe, [90, 100, 120])

twa['low'] = fuzz.trimf(twa.universe, [0.0, 0.1, 0.2])
twa['normal'] = fuzz.trimf(twa.universe, [0.15, 0.3, 0.45])
twa['high'] = fuzz.trimf(twa.universe, [0.4, 0.5, 0.6])

# Vystup
arrhythmia['normal'] = fuzz.trimf(arrhythmia.universe, [0.0, 0.25, 0.5])
arrhythmia['moderate'] = fuzz.trimf(arrhythmia.universe, [0.3, 0.5, 0.7])
arrhythmia['severe'] = fuzz.trimf(arrhythmia.universe, [0.6, 0.75, 1.0])

# Ručne prepísané pravidlá podľa tabuľky (riadok po riadku):
rules = [
    ctrl.Rule(hr['low'] & qrs['short'] & twa['low'], arrhythmia['normal']),
    ctrl.Rule(hr['low'] & qrs['short'] & twa['normal'], arrhythmia['normal']),
    ctrl.Rule(hr['low'] & qrs['short'] & twa['high'], arrhythmia['moderate']),
    ctrl.Rule(hr['low'] & qrs['normal'] & twa['low'], arrhythmia['normal']),
    ctrl.Rule(hr['low'] & qrs['normal'] & twa['normal'], arrhythmia['moderate']),
    ctrl.Rule(hr['low'] & qrs['normal'] & twa['high'], arrhythmia['moderate']),
    ctrl.Rule(hr['low'] & qrs['long'] & twa['low'], arrhythmia['moderate']),
    ctrl.Rule(hr['low'] & qrs['long'] & twa['normal'], arrhythmia['severe']),
    ctrl.Rule(hr['low'] & qrs['long'] & twa['high'], arrhythmia['severe']),

    ctrl.Rule(hr['normal'] & qrs['short'] & twa['low'], arrhythmia['normal']),
    ctrl.Rule(hr['normal'] & qrs['short'] & twa['normal'], arrhythmia['normal']),
    ctrl.Rule(hr['normal'] & qrs['short'] & twa['high'], arrhythmia['moderate']),
    ctrl.Rule(hr['normal'] & qrs['normal'] & twa['low'], arrhythmia['normal']),
    ctrl.Rule(hr['normal'] & qrs['normal'] & twa['normal'], arrhythmia['normal']),
    ctrl.Rule(hr['normal'] & qrs['normal'] & twa['high'], arrhythmia['moderate']),
    ctrl.Rule(hr['normal'] & qrs['long'] & twa['low'], arrhythmia['moderate']),
    ctrl.Rule(hr['normal'] & qrs['long'] & twa['normal'], arrhythmia['severe']),
    ctrl.Rule(hr['normal'] & qrs['long'] & twa['high'], arrhythmia['severe']),

    ctrl.Rule(hr['high'] & qrs['short'] & twa['low'], arrhythmia['moderate']),
    ctrl.Rule(hr['high'] & qrs['short'] & twa['normal'], arrhythmia['moderate']),
    ctrl.Rule(hr['high'] & qrs['short'] & twa['high'], arrhythmia['severe']),
    ctrl.Rule(hr['high'] & qrs['normal'] & twa['low'], arrhythmia['moderate']),
    ctrl.Rule(hr['high'] & qrs['normal'] & twa['normal'], arrhythmia['severe']),
    ctrl.Rule(hr['high'] & qrs['normal'] & twa['high'], arrhythmia['severe']),
    ctrl.Rule(hr['high'] & qrs['long'] & twa['low'], arrhythmia['severe']),
    ctrl.Rule(hr['high'] & qrs['long'] & twa['normal'], arrhythmia['severe']),
    ctrl.Rule(hr['high'] & qrs['long'] & twa['high'], arrhythmia['severe']),
]

# 5. Fuzzy kontrolér
diagnosis_ctrl = ctrl.ControlSystem(rules)
fuzzy_classifier = ctrl.ControlSystemSimulation(diagnosis_ctrl)

# 6. Funkcia na predikciu
def predict_arrhythmia(hr_value, qrs_value, twa_value, rule_based=True, threshold=0.3, debug=False):
    sim = ctrl.ControlSystemSimulation(diagnosis_ctrl)
    sim.input['HR'] = hr_value
    sim.input['QRS'] = qrs_value
    sim.input['TWA'] = twa_value

    try:
        sim.compute()
        output = sim.output['Arrhythmia']

        if rule_based:
            # Fuzzy členstvo
            normal_mu = interp_membership(arrhythmia.universe, arrhythmia['normal'].mf, output)
            moderate_mu = interp_membership(arrhythmia.universe, arrhythmia['moderate'].mf, output)
            severe_mu = interp_membership(arrhythmia.universe, arrhythmia['severe'].mf, output)

            if debug:
                print(f"→ HR={hr_value:.1f}, QRS={qrs_value:.1f}, TWA={twa_value:.3f} | output={output:.3f}")
                print(f"   μ(Normal)={normal_mu:.2f}, μ(Moderate)={moderate_mu:.2f}, μ(Severe)={severe_mu:.2f}")

            # Logika s prahom na členstvá
            if severe_mu > threshold:
                category = "Severe"
            elif moderate_mu > threshold:
                category = "Moderate"
            elif normal_mu > threshold:
                category = "Normal"
            else:
                category = "Unknown"
        else:
            # Klasifikácia len na základe výstupu
            if output < 0.33:
                category = "Normal"
            elif output < 0.66:
                category = "Moderate"
            else:
                category = "Severe"

            if debug:
                print(f"→ HR={hr_value:.1f}, QRS={qrs_value:.1f}, TWA={twa_value:.3f} | output={output:.3f} → {category}")

    except Exception as e:
        output = np.nan
        category = "Unknown"
        if debug:
            print(f"❌ Exception during fuzzy computation: {e}")

    return output, category




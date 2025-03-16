import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# 1️⃣ Definovanie vstupných a výstupných fuzzy premenných
hr = ctrl.Antecedent(np.arange(50, 180, 1), 'HeartRate')
hrv = ctrl.Antecedent(np.arange(10, 200, 1), 'HRV')
amplitude = ctrl.Antecedent(np.arange(0.2, 2.0, 0.01), 'Amplitude')
classification = ctrl.Consequent(np.arange(0, 5, 1), 'Class')

# 2️⃣ Presné členovacie funkcie pre vstupy
hr['low'] = fuzz.gaussmf(hr.universe, 60, 10)
hr['normal'] = fuzz.gaussmf(hr.universe, 90, 15)
hr['high'] = fuzz.gaussmf(hr.universe, 130, 20)

hrv['low'] = fuzz.gaussmf(hrv.universe, 30, 15)
hrv['normal'] = fuzz.gaussmf(hrv.universe, 90, 20)
hrv['high'] = fuzz.gaussmf(hrv.universe, 150, 25)

amplitude['low'] = fuzz.sigmf(amplitude.universe, 0.5, -10)
amplitude['normal'] = fuzz.sigmf(amplitude.universe, 1.0, 10)
amplitude['high'] = fuzz.sigmf(amplitude.universe, 1.5, 10)

# 3️⃣ Členovacie funkcie pre výstup
classification['normal'] = fuzz.trimf(classification.universe, [0, 0, 1])
classification['bradycardia'] = fuzz.trimf(classification.universe, [0, 1, 2])
classification['tachycardia'] = fuzz.trimf(classification.universe, [1, 2, 3])
classification['arrhythmia'] = fuzz.trimf(classification.universe, [2, 3, 4])
classification['severe'] = fuzz.trimf(classification.universe, [3, 4, 4])

# 4️⃣ Vylepšené fuzzy pravidlá pre presnejšiu klasifikáciu
rule1 = ctrl.Rule(hr['low'] & hrv['high'], classification['bradycardia'])
rule2 = ctrl.Rule(hr['high'] & hrv['low'], classification['tachycardia'])
rule3 = ctrl.Rule(hrv['low'] & amplitude['high'], classification['arrhythmia'])
rule4 = ctrl.Rule(hr['normal'] & hrv['normal'] & amplitude['normal'], classification['normal'])
rule5 = ctrl.Rule(hr['high'] & hrv['high'] & amplitude['high'], classification['severe'])
rule6 = ctrl.Rule(hr['low'] & hrv['low'] & amplitude['low'], classification['bradycardia'])
rule7 = ctrl.Rule(hr['high'] & hrv['normal'] & amplitude['low'], classification['tachycardia'])

# 5️⃣ Vytvorenie fuzzy kontrolného systému
system_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])
classifier = ctrl.ControlSystemSimulation(system_ctrl)

# 6️⃣ Testovanie klasifikátora na náhodných vstupoch
def classify_ekg(hr_value, hrv_value, amp_value):
    classifier.input['HeartRate'] = hr_value
    classifier.input['HRV'] = hrv_value
    classifier.input['Amplitude'] = amp_value
    classifier.compute()
    return classifier.output['Class']

# Testovací príklad
hr_test, hrv_test, amp_test = 120, 40, 1.5
result = classify_ekg(hr_test, hrv_test, amp_test)
print(f'Predikovaná trieda: {result:.2f}')

# 7️⃣ Vizualizácia fuzzy množín
hr.view()
hrv.view()
amplitude.view()
classification.view()
plt.show()
# 8️⃣ Pridanie vizualizácie hodnoty na grafe
x_value = hr_test
mu_low = fuzz.interp_membership(hr.universe, hr['low'].mf, x_value)
mu_normal = fuzz.interp_membership(hr.universe, hr['normal'].mf, x_value)
mu_high = fuzz.interp_membership(hr.universe, hr['high'].mf, x_value)

plt.figure(figsize=(8, 5))
plt.plot(hr.universe, hr['low'].mf, 'r', linewidth=2, label="low")
plt.plot(hr.universe, hr['normal'].mf, 'b', linewidth=2, label="normal")
plt.plot(hr.universe, hr['high'].mf, 'k', linewidth=2, label="high")

plt.axvline(x_value, color='gray', linestyle='dashed', alpha=0.6)
plt.scatter([x_value], [mu_low], color='red', s=100, zorder=3)
plt.scatter([x_value], [mu_normal], color='blue', s=100, zorder=3)
plt.scatter([x_value], [mu_high], color='black', s=100, zorder=3)

plt.text(x_value + 2, mu_low, f"{mu_low:.2f}", color='red', fontsize=12)
plt.text(x_value + 2, mu_normal, f"{mu_normal:.2f}", color='blue', fontsize=12)
plt.text(x_value + 2, mu_high, f"{mu_high:.2f}", color='black', fontsize=12)

plt.xlabel("Heart Rate (bpm)")
plt.ylabel("Membership Degree")
plt.legend()
plt.title("Fuzzy Membership Function for Heart Rate")
plt.grid()
plt.show()


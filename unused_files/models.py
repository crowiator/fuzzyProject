import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class FuzzyModel:
    def __init__(self, input_vars, output_vars, rules):
        self.input_vars = {}
        self.output_vars = {}
        self.rules = []

        # Vytvorenie fuzzy premenných pre vstupy
        for name, (universe, mf_definitions) in input_vars.items():
            var = ctrl.Antecedent(np.array(universe), name)
            for mf_name, (mf_type, params) in mf_definitions.items():
                self._add_membership_function(var, mf_name, mf_type, params)
            self.input_vars[name] = var

        # Vytvorenie fuzzy premenných pre výstupy
        for name, (universe, mf_definitions) in output_vars.items():
            var = ctrl.Consequent(np.array(universe), name)
            for mf_name, (mf_type, params) in mf_definitions.items():
                self._add_membership_function(var, mf_name, mf_type, params)
            self.output_vars[name] = var

        # Pridanie pravidiel
        for rule in rules:
            self.rules.append(ctrl.Rule(*rule))

        # Vytvorenie kontrolného systému
        self.control_system = ctrl.ControlSystem(self.rules)
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)

    def _add_membership_function(self, var, name, mf_type, params):
        if mf_type == 'gauss':
            var[name] = fuzz.gaussmf(var.universe, *params)
        elif mf_type == 'sigmoid':
            var[name] = fuzz.sigmf(var.universe, *params)
        elif mf_type == 'trimf':
            var[name] = fuzz.trimf(var.universe, params)
        else:
            raise ValueError(f'Neznámy typ členovacej funkcie: {mf_type}')

    def classify(self, **inputs):
        for var_name, value in inputs.items():
            if var_name in self.input_vars:
                self.simulation.input[var_name] = value
            else:
                raise ValueError(f'Neznáma vstupná premenná: {var_name}')

        self.simulation.compute()
        return {name: self.simulation.output[name] for name in self.output_vars}

    def extract_fuzzy_features(self, **inputs):
        fuzzy_features = {}
        for var_name, value in inputs.items():
            if var_name in self.input_vars:
                var = self.input_vars[var_name]
                for mf_name in var.terms:
                    mf = var.terms[mf_name].mf
                    fuzzy_features[f'{var_name}_{mf_name}'] = fuzz.interp_membership(var.universe, mf, value)
            else:
                raise ValueError(f'Neznáma vstupná premenná: {var_name}')
        return fuzzy_features

    def visualize(self):
        for var in self.input_vars.values():
            var.view()
        for var in self.output_vars.values():
            var.view()
        plt.show()


# **Definícia modelu pre klasifikáciu EKG**
input_variables = {
    'HeartRate': ([50, 180, 1], {
        'low': ('gauss', [60, 10]),
        'normal': ('gauss', [90, 15]),
        'high': ('gauss', [130, 20])
    }),
    'HRV': ([10, 200, 1], {
        'low': ('gauss', [30, 15]),
        'normal': ('gauss', [90, 20]),
        'high': ('gauss', [150, 25])
    }),
    'Amplitude': ([0.2, 2.0, 0.01], {
        'low': ('sigmoid', [0.5, -10]),
        'normal': ('sigmoid', [1.0, 10]),
        'high': ('sigmoid', [1.5, 10])
    })
}

output_variables = {
    'Class': ([0, 5, 1], {
        'normal': ('trimf', [0, 0, 1]),
        'bradycardia': ('trimf', [0, 1, 2]),
        'tachycardia': ('trimf', [1, 2, 3]),
        'arrhythmia': ('trimf', [2, 3, 4]),
        'severe': ('trimf', [3, 4, 4])
    })
}

# **Trénovanie modelu s fuzzy featurami**
X = []
y = []
np.random.seed(42)
for _ in range(1000):
    hr = np.random.randint(50, 180)
    hrv = np.random.randint(10, 200)
    amp = np.random.uniform(0.2, 2.0)
    fuzzy_features = FuzzyModel(input_variables, output_variables, []).extract_fuzzy_features(HeartRate=hr, HRV=hrv,
                                                                                              Amplitude=amp)
    X.append(list(fuzzy_features.values()))
    y.append(np.random.randint(0, 5))  # Simulované výstupy

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'SVM': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'{name} Accuracy: {accuracy_score(y_test, y_pred):.2f}')

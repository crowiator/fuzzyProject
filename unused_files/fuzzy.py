import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt


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

rules = [
    (input_variables['HeartRate'][1]['low'] & input_variables['HRV'][1]['high'],
     output_variables['Class'][1]['bradycardia']),
    (input_variables['HeartRate'][1]['high'] & input_variables['HRV'][1]['low'],
     output_variables['Class'][1]['tachycardia']),
    (input_variables['HRV'][1]['low'] & input_variables['Amplitude'][1]['high'],
     output_variables['Class'][1]['arrhythmia']),
    (input_variables['HeartRate'][1]['normal'] & input_variables['HRV'][1]['normal'] & input_variables['Amplitude'][1][
        'normal'], output_variables['Class'][1]['normal']),
    (input_variables['HeartRate'][1]['high'] & input_variables['HRV'][1]['high'] & input_variables['Amplitude'][1][
        'high'], output_variables['Class'][1]['severe']),
    (input_variables['HeartRate'][1]['low'] & input_variables['HRV'][1]['low'] & input_variables['Amplitude'][1]['low'],
     output_variables['Class'][1]['bradycardia']),
    (input_variables['HeartRate'][1]['high'] & input_variables['HRV'][1]['normal'] & input_variables['Amplitude'][1][
        'low'], output_variables['Class'][1]['tachycardia'])
]

# **Vytvorenie univerzálneho fuzzy modelu**
fuzzy_model = FuzzyModel(input_variables, output_variables, rules)

# **Testovanie univerzálneho modelu**
test_result = fuzzy_model.classify(HeartRate=120, HRV=40, Amplitude=1.5)
print(f'Predikovaná trieda: {test_result}')

# **Vizualizácia fuzzy množín**
fuzzy_model.visualize()
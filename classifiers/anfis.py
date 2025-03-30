from simpful import *

# ANFIS systém vytvorený iba raz
FS = FuzzySystem()

# Definuj fuzzy množiny pre vstupy
HR_TLV = AutoTriangle(3, terms=["Low", "Normal", "High"], universe_of_discourse=[30, 160])
QRS_TLV = AutoTriangle(3, terms=["Short", "Normal", "Long"], universe_of_discourse=[40, 180])
TWA_TLV = AutoTriangle(3, terms=["Low", "Normal", "High"], universe_of_discourse=[0.0, 1.0])

FS.add_linguistic_variable("HR", HR_TLV)
FS.add_linguistic_variable("QRS_interval", QRS_TLV)
FS.add_linguistic_variable("T_wave", TWA_TLV)

# Výstup (klasifikačné fuzzy množiny)
C1 = TriangleFuzzySet(0, 0, 1, term="Normal")
C2 = TriangleFuzzySet(0, 1, 2, term="Moderate")
C3 = TriangleFuzzySet(1, 2, 2, term="Severe")
FS.add_linguistic_variable("class", LinguisticVariable([C1, C2, C3], universe_of_discourse=[0, 2]))

# Definuj pravidlá (statické)
rules = [
    "IF (HR IS Low) AND (QRS_interval IS Short) AND (T_wave IS Low) THEN (class IS Normal)",
    "IF (HR IS Low) AND (QRS_interval IS Short) AND (T_wave IS Normal) THEN (class IS Normal)",
    "IF (HR IS Low) AND (QRS_interval IS Short) AND (T_wave IS High) THEN (class IS Moderate)",

    "IF (HR IS Low) AND (QRS_interval IS Normal) AND (T_wave IS Low) THEN (class IS Normal)",
    "IF (HR IS Low) AND (QRS_interval IS Normal) AND (T_wave IS Normal) THEN (class IS Moderate)",
    "IF (HR IS Low) AND (QRS_interval IS Normal) AND (T_wave IS High) THEN (class IS Moderate)",

    "IF (HR IS Low) AND (QRS_interval IS Long) AND (T_wave IS Low) THEN (class IS Moderate)",
    "IF (HR IS Low) AND (QRS_interval IS Long) AND (T_wave IS Normal) THEN (class IS Severe)",
    "IF (HR IS Low) AND (QRS_interval IS Long) AND (T_wave IS High) THEN (class IS Severe)",

    "IF (HR IS Normal) AND (QRS_interval IS Short) AND (T_wave IS Low) THEN (class IS Normal)",
    "IF (HR IS Normal) AND (QRS_interval IS Short) AND (T_wave IS Normal) THEN (class IS Normal)",
    "IF (HR IS Normal) AND (QRS_interval IS Short) AND (T_wave IS High) THEN (class IS Moderate)",

    "IF (HR IS Normal) AND (QRS_interval IS Normal) AND (T_wave IS Low) THEN (class IS Normal)",
    "IF (HR IS Normal) AND (QRS_interval IS Normal) AND (T_wave IS Normal) THEN (class IS Normal)",
    "IF (HR IS Normal) AND (QRS_interval IS Normal) AND (T_wave IS High) THEN (class IS Moderate)",

    "IF (HR IS Normal) AND (QRS_interval IS Long) AND (T_wave IS Low) THEN (class IS Moderate)",
    "IF (HR IS Normal) AND (QRS_interval IS Long) AND (T_wave IS Normal) THEN (class IS Severe)",
    "IF (HR IS Normal) AND (QRS_interval IS Long) AND (T_wave IS High) THEN (class IS Severe)",

    "IF (HR IS High) AND (QRS_interval IS Short) AND (T_wave IS Low) THEN (class IS Moderate)",
    "IF (HR IS High) AND (QRS_interval IS Short) AND (T_wave IS Normal) THEN (class IS Moderate)",
    "IF (HR IS High) AND (QRS_interval IS Short) AND (T_wave IS High) THEN (class IS Severe)",

    "IF (HR IS High) AND (QRS_interval IS Normal) AND (T_wave IS Low) THEN (class IS Moderate)",
    "IF (HR IS High) AND (QRS_interval IS Normal) AND (T_wave IS Normal) THEN (class IS Severe)",
    "IF (HR IS High) AND (QRS_interval IS Normal) AND (T_wave IS High) THEN (class IS Severe)",

    "IF (HR IS High) AND (QRS_interval IS Long) AND (T_wave IS Low) THEN (class IS Severe)",
    "IF (HR IS High) AND (QRS_interval IS Long) AND (T_wave IS Normal) THEN (class IS Severe)",
    "IF (HR IS High) AND (QRS_interval IS Long) AND (T_wave IS High) THEN (class IS Severe)"
]
FS.add_rules(rules)

# 🔄 Funkcia, ktorú môžeme volať z main.py / RF modelov
def get_anfis_score(hr, qrs, twa):
    FS.set_variable("HR", hr)
    FS.set_variable("QRS_interval", qrs)
    FS.set_variable("T_wave", twa)
    result = FS.Mamdani_inference(["class"])
    return result["class"]

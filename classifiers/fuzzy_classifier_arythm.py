import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import re
# Výstupné triedy: skratka → plný názov
class_labels = {
    "N": "Normal",
    "ST": "Sinus Tachycardia",
    "AT": "Atrial Tachycardia",
    "AF": "Atrial Flutter",
    "AFb": "Atrial Fibrillation",
    "VT": "Ventricular Tachycardia",
    "SB": "Sinus Bradycardia",
    "1B": "First AV Block",
    "2B1": "Second AV Block Type 1",
    "2B2": "Second AV Block Type 2",
    "3B": "Third AV Block",
    "PAC": "Premature Atrial Contraction",
    "PVC": "Premature Ventricular Contraction"
}
# Jemnejší rozsah: 0 až 200 s krokom 0.2 bpm
vr_range = np.linspace(0, 200, 1001)

# μ_slow
vr_slow = np.piecewise(vr_range,
    [vr_range <= 55,
     (vr_range > 55) & (vr_range < 60),
     vr_range >= 60],
    [1,
     lambda x: (60 - x) / 5,
     0])

# μ_normal
vr_normal = np.piecewise(vr_range,
    [vr_range <= 55,
     (vr_range > 55) & (vr_range < 60),
     (vr_range >= 60) & (vr_range <= 100),
     (vr_range > 100) & (vr_range < 105),
     vr_range >= 105],
    [0,
     lambda x: (x - 55) / 5,
     1,
     lambda x: (105 - x) / 5,
     0])

# μ_high
vr_high = np.piecewise(vr_range,
    [vr_range <= 100,
     (vr_range > 100) & (vr_range < 105),
     (vr_range >= 105) & (vr_range <= 155),
     (vr_range > 155) & (vr_range < 160),
     vr_range >= 160],
    [0,
     lambda x: (x - 100) / 5,
     1,
     lambda x: (160 - x) / 5,
     0])

# μ_very_high
vr_very_high = np.piecewise(vr_range,
    [vr_range <= 155,
     (vr_range > 155) & (vr_range < 160),
     vr_range >= 160],
    [0,
     lambda x: (x - 155) / 5,
     1])

# Vesmír diskurzu pre PRI: 0 až 300 ms
pri_range = np.linspace(0, 300, 1001)

# μ_Narrow
pri_narrow = np.piecewise(pri_range,
    [pri_range <= 115,
     (pri_range > 115) & (pri_range < 120),
     pri_range >= 120],
    [1,
     lambda x: (120 - x) / 5,
     0])

# μ_Normal
pri_normal = np.piecewise(pri_range,
    [pri_range <= 115,
     (pri_range > 115) & (pri_range < 120),
     (pri_range >= 120) & (pri_range <= 200),
     (pri_range > 200) & (pri_range < 205),
     pri_range >= 205],
    [0,
     lambda x: (x - 115) / 5,
     1,
     lambda x: (205 - x) / 5,
     0])

# μ_Broad
pri_broad = np.piecewise(pri_range,
    [pri_range <= 200,
     (pri_range > 200) & (pri_range < 205),
     pri_range >= 205],
    [0,
     lambda x: (x - 200) / 5,
     1])

#QRS
qrs_range = np.linspace(0, 200, 1001)

# μ_Narrow
qrs_narrow = np.piecewise(qrs_range,
    [qrs_range <= 55,
     (qrs_range > 55) & (qrs_range < 60),
     qrs_range >= 60],
    [1,
     lambda x: (60 - x) / 5,
     0])

# μ_Normal
qrs_normal = np.piecewise(qrs_range,
    [qrs_range <= 55,
     (qrs_range > 55) & (qrs_range < 60),
     (qrs_range >= 60) & (qrs_range <= 100),
     (qrs_range > 100) & (qrs_range < 105),
     qrs_range >= 105],
    [0,
     lambda x: (x - 55) / 5,
     1,
     lambda x: (105 - x) / 5,
     0])

# μ_Broad
qrs_broad = np.piecewise(qrs_range,
    [qrs_range <= 100,
     (qrs_range > 100) & (qrs_range < 105),
     qrs_range >= 105],
    [0,
     lambda x: (x - 100) / 5,
     1])
# RR interval
rr_range = np.linspace(0.3, 1.5, 1000)  # R-R interval range in seconds

# μ_Short
rr_short = np.piecewise(rr_range,
    [rr_range <= 0.55,
     (rr_range > 0.55) & (rr_range < 0.6),
     rr_range >= 0.6],
    [1,
     lambda x: (0.6 - x) / 0.05,
     0])

# μ_Normal
rr_normal = np.piecewise(rr_range,
    [rr_range <= 0.55,
     (rr_range > 0.55) & (rr_range < 0.6),
     (rr_range >= 0.6) & (rr_range <= 1.0),
     (rr_range > 1.0) & (rr_range < 1.05),
     rr_range >= 1.05],
    [0,
     lambda x: (x - 0.55) / 0.05,
     1,
     lambda x: (1.05 - x) / 0.05,
     0])

# μ_Wide
rr_wide = np.piecewise(rr_range,
    [rr_range <= 1.0,
     (rr_range > 1.0) & (rr_range < 1.05),
     rr_range >= 1.05],
    [0,
     lambda x: (x - 1.0) / 0.05,
     1])
# Rozsah AR (Atrial Rate)
ar_range = np.linspace(0, 400, 1000)

# µ_slow
ar_slow = np.piecewise(ar_range,
    [ar_range <= 50,
     (ar_range > 50) & (ar_range < 60),
     ar_range >= 60],
    [1,
     lambda x: (60 - x) / 10,
     0])

# µ_normal
ar_normal = np.piecewise(ar_range,
    [ar_range <= 50,
     (ar_range > 50) & (ar_range < 60),
     (ar_range >= 60) & (ar_range <= 100),
     (ar_range > 100) & (ar_range < 110),
     ar_range >= 110],
    [0,
     lambda x: (x - 50) / 10,
     1,
     lambda x: (110 - x) / 10,
     0])

# µ_little_high
ar_lh = np.piecewise(ar_range,
    [ar_range <= 105,
     (ar_range > 105) & (ar_range < 110),
     (ar_range >= 110) & (ar_range <= 150),
     (ar_range > 150) & (ar_range < 155),
     ar_range >= 155],
    [0,
     lambda x: (x - 105) / 5,
     1,
     lambda x: (155 - x) / 5,
     0])

# µ_high
ar_high = np.piecewise(ar_range,
    [ar_range <= 150,
     (ar_range > 150) & (ar_range < 160),
     (ar_range >= 160) & (ar_range <= 240),
     (ar_range > 240) & (ar_range < 250),
     ar_range >= 250],
    [0,
     lambda x: (x - 150) / 10,
     1,
     lambda x: (250 - x) / 10,
     0])

# µ_very_high
ar_vh = np.piecewise(ar_range,
    [ar_range <= 245,
     (ar_range > 245) & (ar_range < 250),
     (ar_range >= 250) & (ar_range <= 350),
     (ar_range > 350) & (ar_range < 355),
     ar_range >= 355],
    [0,
     lambda x: (x - 245) / 5,
     1,
     lambda x: (355 - x) / 5,
     0])

# µ_extremely_high
ar_eh = np.piecewise(ar_range,
    [ar_range <= 350,
     (ar_range > 350) & (ar_range < 360),
     ar_range >= 360],
    [0,
     lambda x: (x - 350) / 5,
     1])
# P–P interval (v sekundách)
pp_range = np.linspace(0.4, 1.2, 1000)

# µ_short
pp_short = np.piecewise(pp_range,
    [pp_range <= 0.55,
     (pp_range > 0.55) & (pp_range < 0.6),
     pp_range >= 0.6],
    [1,
     lambda x: (0.6 - x) / 0.05,
     0])

# µ_normal
pp_normal = np.piecewise(pp_range,
    [pp_range <= 0.55,
     (pp_range > 0.55) & (pp_range < 0.6),
     (pp_range >= 0.6) & (pp_range <= 1.0),
     (pp_range > 1.0) & (pp_range < 1.05),
     pp_range >= 1.05],
    [0,
     lambda x: (x - 0.55) / 0.05,
     1,
     lambda x: (1.05 - x) / 0.05,
     0])

# µ_wide
pp_wide = np.piecewise(pp_range,
    [pp_range <= 1.0,
     (pp_range > 1.0) & (pp_range < 1.05),
     pp_range >= 1.05],
    [0,
     lambda x: (x - 1.0) / 0.05,
     1])


# Rozsah hodnôt pre P:QRS pomer
pqrs = np.linspace(0, 2, 1000)

# μ_Low(x) podľa článku
pqrs_low = np.piecewise(pqrs,
                      [pqrs <= 0.55,
                       (pqrs > 0.55) & (pqrs < 0.6),
                       pqrs >= 0.6],
                      [1,
     lambda x: (0.6 - x) / 0.05,
     0])

# μ_High(x) podľa článku
pqrs_high = np.piecewise(pqrs,
                       [pqrs <= 1.0,
                        (pqrs > 1.0) & (pqrs < 1.05),
                        pqrs >= 1.05],
                       [0,
     lambda x: (x - 1.0) / 0.05,
     1])

# μ_Desirable(x) — stredová symetrická fuzzy množina okolo 1
pqrs_desirable = np.piecewise(pqrs,
                            [(pqrs >= 0.8) & (pqrs <= 1.0),
                             (pqrs > 1.0) & (pqrs <= 1.2),
                             (pqrs < 0.8) | (pqrs > 1.2)],
                            [lambda x: (x - 0.8) / 0.2,
     lambda x: (1.2 - x) / 0.2,
     0])

# Rozsah pre RI ratio (napr. od -2 po 6)
ri_range = np.linspace(-2, 6, 1000)

# μ_Low(x)
ri_low = np.piecewise(ri_range,
    [ri_range <= -2,
     (ri_range > -2) & (ri_range <= 1),
     (ri_range > 1) & (ri_range <= 4),
     ri_range > 4],
    [1,
     lambda x: 1 - 2*((x + 2)/6)**2,
     lambda x: 2*((x - 4)/6)**2,
     0])

# μ_High(x)
ri_high = np.piecewise(ri_range,
    [ri_range <= -2,
     (ri_range > -2) & (ri_range <= 1),
     (ri_range > 1) & (ri_range <= 4),
     ri_range > 4],
    [0,
     lambda x: 2*((x + 2)/6)**2,
     lambda x: 1 - 2*((x - 4)/6)**2,
     1])

# μ_Desirable(x): trojuholníková funkcia so stredom v 1
ri_desirable = np.piecewise(ri_range,
    [ri_range <= 0.8,
     (ri_range > 0.8) & (ri_range <= 1.2),
     ri_range > 1.2],
    [lambda x: (x - 0.5) / (0.3),
     lambda x: (1.5 - x) / (0.3),
     0])
#PI2 : PI1 ratio
pi_range = np.linspace(-2, 6, 1000)

# μ_Low(x)
pi_low = np.piecewise(pi_range,
                      [pi_range <= -2,
                       (pi_range > -2) & (pi_range <= 1),
                       (pi_range > 1) & (pi_range <= 4),
                       pi_range > 4],
                      [1,
     lambda x: 1 - 2 * ((x + 2)/6)**2,
     lambda x: 2 * ((x - 4)/6)**2,
     0])

# μ_High(x)
pi_high = np.piecewise(pi_range,
                       [pi_range <= -2,
                        (pi_range > -2) & (pi_range <= 1),
                        (pi_range > 1) & (pi_range <= 4),
                        pi_range > 4],
                       [0,
     lambda x: 2 * ((x + 2)/6)**2,
     lambda x: 1 - 2 * ((x - 4)/6)**2,
     1])

# μ_Desirable(x): trojuholníková funkcia vrcholom v 1
pi_desirable = np.piecewise(pi_range,
                            [pi_range <= 0.7,
                             (pi_range > 0.7) & (pi_range <= 1.3),
                             pi_range > 1.3],
                            [lambda x: (x - 0.5) / (0.2),
     lambda x: (1.5 - x) / (0.2),
     0])
# T wave
twa = np.linspace(-6, 6, 1000)

# μ_Negative
twa_neg = np.piecewise(twa,
                       [twa <= -3,
                        (twa > -3) & (twa <= 0),
                        (twa > 0) & (twa <= 3),
                        twa > 3],
                       [1,
     lambda x: 1 - 2 * ((x + 3)/6)**2,
     lambda x: 2 * ((x - 3)/6)**2,
     0])

# μ_Positive
twa_pos = np.piecewise(twa,
                      [twa <= -3,
                       (twa > -3) & (twa <= 0),
                       (twa > 0) & (twa <= 3),
                       twa > 3],
                      [0,
     lambda x: 2 * ((x + 3)/6)**2,
     lambda x: 1 - 2 * ((x - 3)/6)**2,
     1])

# μ_Isolated (tvarovaný ako trojuholník so špičkou v 0)
twa_iso = np.piecewise(twa,
                      [twa <= -0.1,
                       (twa > -0.1) & (twa <= 0),
                       (twa > 0) & (twa <= 0.1),
                       twa > 0.1],
                      [0,
     lambda x: (x + 0.1) / 0.1,
     lambda x: (0.1 - x) / 0.1,
     0])

rules = [
    "IF (VR IS Normal) AND (PRI IS Normal) AND (QRSd IS Normal) AND (AR IS Normal) AND (PP IS Normal) AND (PQRS IS Desirable) AND (RR IS Normal) AND (RI IS Desirable) AND (PI IS Desirable) AND (T IS Positive) THEN (class IS Normal)",
    "IF (VR IS Slow) AND (PRI IS Normal) AND (QRSd IS Normal) AND (AR IS Slow) AND (PP IS Wide) AND (PQRS IS Desirable) AND (RR IS Wide) AND (RI IS Desirable) AND (PI IS Desirable) AND (T IS Positive) THEN (class IS SB)",
    "IF (VR IS Slow) AND (PRI IS Broad) AND (QRSd IS Normal) AND (AR IS Normal) AND (PP IS Normal) AND (PQRS IS Desirable) AND (RR IS Wide) AND (RI IS Desirable) AND (PI IS Desirable) AND (T IS Positive) THEN (class IS 1B)",
    "IF (VR IS Normal) AND (PRI IS Broad) AND (QRSd IS Normal) AND (AR IS Normal) AND (PP IS Normal) AND (PQRS IS Desirable) AND (RR IS Normal) AND (RI IS Desirable) AND (PI IS Desirable) AND (T IS Positive) THEN (class IS 1B)",
    "IF (VR IS Normal) AND (PRI IS Broad) AND (QRSd IS Normal) AND (AR IS Normal) AND (PP IS Normal) AND (PQRS IS High) AND (RR IS Normal) AND (RI IS High) AND (PI IS Desirable) AND (T IS Positive) THEN (class IS 2B1)",
    "IF (VR IS Normal) AND (PRI IS Normal) AND (QRSd IS Broad) AND (AR IS Normal) AND (PP IS Normal) AND (PQRS IS High) AND (RR IS Wide) AND (RI IS High) AND (PI IS Desirable) AND (T IS Positive) THEN (class IS 2B1)",
    "IF (VR IS Slow) AND (PRI IS Broad) AND (QRSd IS Normal) AND (AR IS Normal) AND (PP IS Normal) AND (PQRS IS High) AND (RR IS Wide) AND (RI IS High) AND (PI IS Desirable) AND (T IS Positive) THEN (class IS 2B1)",
    "IF (VR IS Slow) AND (PRI IS Normal) AND (QRSd IS Broad) AND (AR IS Normal) AND (PP IS Normal) AND (PQRS IS High) AND (RR IS Wide) AND (RI IS High) AND (PI IS Desirable) AND (T IS Positive) THEN (class IS 2B2)",
    "IF (VR IS High) AND (QRSd IS Broad) AND (PQRS IS Low) AND (RR IS Short) THEN (class IS VT)",
    "IF (VR IS Slow) AND (QRSd IS Broad) AND (AR IS Normal) AND (PP IS Normal) AND (PQRS IS High) AND (RR IS Wide) AND (RI IS Low) AND (PI IS Low) AND (T IS Positive) THEN (class IS 3B)",
    "IF (VR IS Slow) AND (QRSd IS Broad) AND (AR IS Normal) AND (PP IS Normal) AND (PQRS IS High) AND (RR IS Wide) AND (RI IS Low) AND (PI IS High) AND (T IS Positive) THEN (class IS 3B)",
    "IF (VR IS Slow) AND (QRSd IS Broad) AND (AR IS Normal) AND (PP IS Normal) AND (PQRS IS High) AND (RR IS Wide) AND (RI IS High) AND (PI IS High) AND (T IS Positive) THEN (class IS 3B)",
    "IF (VR IS Slow) AND (QRSd IS Broad) AND (AR IS Normal) AND (PP IS Normal) AND (PQRS IS High) AND (RR IS Wide) AND (RI IS High) AND (PI IS Low) AND (T IS Positive) THEN (class IS 3B)",
    "IF (VR IS Slow) AND (QRSd IS Broad) AND (AR IS Little high) AND (PP IS Short) AND (PQRS IS High) AND (RR IS Wide) AND (RI IS Low) AND (PI IS Low) AND (T IS Positive) THEN (class IS 3B)",
    "IF (VR IS Slow) AND (QRSd IS Broad) AND (AR IS Little high) AND (PP IS Short) AND (PQRS IS High) AND (RR IS Wide) AND (RI IS Low) AND (PI IS High) AND (T IS Positive) THEN (class IS 3B)",
    "IF (VR IS Slow) AND (QRSd IS Broad) AND (AR IS Little high) AND (PP IS Short) AND (PQRS IS High) AND (RR IS Wide) AND (RI IS High) AND (PI IS High) AND (T IS Positive) THEN (class IS 3B)",
    "IF (VR IS Slow) AND (QRSd IS Broad) AND (AR IS Little high) AND (PP IS Short) AND (PQRS IS High) AND (RR IS Wide) AND (RI IS High) AND (PI IS Low) AND (T IS Positive) THEN (class IS 3B)",
    "IF (PRI IS Normal) AND (QRSd IS Normal) AND (PP IS Normal) AND (PQRS IS Desirable) AND (RR IS Normal) AND (RI IS Low) AND (PI IS Low) AND (T IS Positive) THEN (class IS PAC)",
    "IF (PRI IS Normal) AND (QRSd IS Normal) AND (PP IS Normal) AND (PQRS IS Desirable) AND (RR IS Normal) AND (RI IS Low) AND (PI IS High) AND (T IS Positive) THEN (class IS PAC)",
    "IF (PRI IS Norma) AND (QRSd IS Normal) AND (PP IS Normal) AND (PQRS IS Desirable) AND (RR IS Normal) AND (RI IS High) AND (PI IS High) AND (T IS Positive) THEN (class IS PAC)",
    "IF (PRI IS Normal) AND (QRSd IS Normal) AND (PP IS Normal) AND (PQRS IS Desirable) AND (RR IS Normal) AND (RI IS High) AND (PI IS Low) AND (T IS Positive) THEN (class IS PAC)",
    "IF (PRI IS Normal) AND (QRSd IS Normal) AND (PP IS Wide) AND (PQRS IS Desirable) AND (RR IS Wide) AND (RI IS Low) AND (PI IS Low) AND (T IS Positive) THEN (class IS PAC)",
    "IF (PRI IS Normal) AND (QRSd IS Normal) AND (PP IS Wide) AND (PQRS IS Desirable) AND (RR IS Wide) AND (RI IS Low) AND (PI IS High) AND (T IS Positive) THEN (class IS PAC)",
    "IF (PRI IS Normal) AND (QRSd IS Normal) AND (PP IS Wide) AND (PQRS IS Desirable) AND (RR IS Wide) AND (RI IS High) AND (PI IS High) AND (T IS Positive) THEN (class IS PAC)",
    "IF (PRI IS Normal) AND (QRSd IS Normal) AND (PP IS Wide) AND (PQRS IS Desirable) AND (RR IS Wide) AND (RI IS High) AND (PI IS Low) AND (T IS Positive) THEN (class IS PAC)",
    "IF (PRI IS Normal) AND (QRSd IS Normal) AND (PP IS Short) AND (PQRS IS Desirable) AND (RR IS Short) AND (RI IS Low) AND (PI IS Low) AND (T IS Positive) THEN (class IS PAC)",
    "IF (PRI IS Normal) AND (QRSd IS Normal) AND (PP IS Short) AND (PQRS IS Desirable) AND (RR IS Short) AND (RI IS Low) AND (PI IS High) AND (T IS Positive) THEN (class IS PAC)",
    "IF (PRI IS Normal) AND (QRSd IS Normal) AND (PP IS Short) AND (PQRS IS Desirable) AND (RR IS Short) AND (RI IS High) AND (PI IS High) AND (T IS Positive) THEN (class IS PAC)",
    "IF (PRI IS Normal) AND (QRSd IS Normal) AND (PP IS Short) AND (PQRS IS Desirable) AND (RR IS Short) AND (RI IS High) AND (PI IS Low) AND (T IS Positive) THEN (class IS PAC)",
    "IF (VR IS Very high) AND (QRSd IS Normal) AND (AR IS Extremely high) AND (PQRS IS High) AND (RR IS Short) AND (RI IS Low) AND (T IS Positive) THEN (class IS AFb)",
    "IF (VR IS Very high) AND (QRSd IS Normal) AND (AR IS Extremely high) AND (PQRS IS High) AND (RR IS Short) AND (RI IS High) AND (T IS Positive) THEN (class IS AFb)",
    "IF (VR IS High) AND (PRI IS Normal) AND (QRSd IS Normal) AND (AR IS Little high) AND (PP IS Short) AND (PQRS IS Desirable) AND (RR IS Short) AND (RI IS Desirable) AND (PI IS Desirable) THEN (class IS ST)",
    "IF (VR IS Very high) AND (PRI IS Normal) AND (QRSd IS Normal) AND (AR IS High) AND (PP IS Short) AND (PQRS IS Desirable) AND (RR IS Short) AND (RI IS Desirable) AND (PI IS Desirable) THEN (class IS ST)",
    "IF (VR IS Very high) AND (PRI IS Normal) AND (QRSd IS Normal) AND (AR IS Very high) AND (PP IS Short) AND (PQRS IS Desirable) AND (RR IS Short) AND (RI IS Desirable) AND (PI IS Desirable) THEN (class IS ST)",
    "IF (VR IS Very high) AND (PRI IS Normal) AND (QRSd IS Normal) AND (AR IS Extremely high) AND (PP IS Short) AND (PQRS IS Desirable) AND (RR IS Short) AND (RI IS Desirable) AND (PI IS Desirable) THEN (class IS ST)",
    "IF (VR IS Very high) AND (PRI IS Narrow) AND (QRSd IS Normal) AND (AR IS High) AND (PP IS Short) AND (PQRS IS Desirable) THEN (class IS AT)",
    "IF (VR IS Very high) AND (PRI IS Narrow) AND (QRSd IS Normal) AND (AR IS Very high) AND (PP IS Short) AND (PQRS IS Desirable) THEN (class IS AT)",
    "IF (VR IS Normal) AND (QRSd IS Normal) AND (AR IS Very high) AND (PP IS Short) AND (PQRS IS High) AND (RR IS Normal) AND (RI IS Desirable) AND (PI IS Desirable) THEN (class IS AF)",
    "IF (VR IS Normal) AND (QRSd IS Normal) AND (AR IS Extremely high) AND (PP IS Short) AND (PQRS IS High) AND (RR IS Normal) AND (RI IS Desirable) AND (PI IS Desirable) THEN (class IS AF)",
    "IF (VR IS High) AND (QRSd IS Normal) AND (AR IS Very high) AND (PP IS Short) AND (PQRS IS High) AND (RR IS Short) AND (RI IS Desirable) AND (PI IS Desirable) THEN (class IS AF)",
    "IF (VR IS High) AND (QRSd IS Normal) AND (AR IS Extremely high) AND (PP IS Short) AND (PQRS IS High) AND (RR IS Short) AND (RI IS Desirable) AND (PI IS Desirable) THEN (class IS AF)",
"IF (VR IS High) AND (QRSd IS Normal) AND (AR IS Extremely high) AND (PQRS IS High) AND (RR IS Short) THEN (class IS AFb)",
    "IF (VR IS High) AND (QRSd IS Normal) AND (AR IS Extremely high) AND (PQRS IS High) AND (RR IS Short) THEN (class IS AFb)",
    "IF (VR IS Very high) AND (QRSd IS Normal) AND (AR IS Extremely high) AND (PQRS IS High) AND (RR IS Short) THEN (class IS AFb)",
    "IF (VR IS Very high) AND (QRSd IS Normal) AND (AR IS Extremely high) AND (PQRS IS High) AND (RR IS Short) THEN (class IS AFb)",
    "IF (VR IS High) AND (QRSd IS Broad) AND (PQRS IS Low) AND (RR IS Short) AND (T IS Negative) THEN (class IS VT)",
    "IF (VR IS Very high) AND (QRSd IS Broad) AND (PQRS IS Low) AND (RR IS Short) AND (T IS Negative) THEN (class IS VT)",
    "IF (QRSd IS Broad) AND (T IS Negative) THEN (class IS PVC)",
    "IF (QRSd IS Broad) AND (T IS Negative) THEN (class IS PVC)",
    "IF (QRSd IS Broad) AND (T IS Negative) THEN (class IS PVC)",
    "IF (QRSd IS Broad) AND (T IS Negative) THEN (class IS PVC)",
    "IF (VR IS High) AND (QRSd IS Broad) AND (PQRS IS Low) AND (RR IS Short) AND (T IS Negative) THEN (class IS VT)",
    "IF (VR IS High) AND (QRSd IS Broad) AND (PQRS IS Low) AND (RR IS Short) AND (T IS Negative) THEN (class IS VT)",
    "IF (VR IS Very high) AND (QRSd IS Broad) AND (PQRS IS Low) AND (RR IS Short) AND (T IS Negative) THEN (class IS VT)",
    "IF (VR IS Very high) AND (QRSd IS Broad) AND (PQRS IS Low) AND (RR IS Short) AND (T IS Negative) THEN (class IS VT)"
]

def get_membership_functions():
    return {
        "VR": {
            "Slow": lambda x: np.interp(x, vr_range, vr_slow),
            "Normal": lambda x: np.interp(x, vr_range, vr_normal),
            "High": lambda x: np.interp(x, vr_range, vr_high),
            "Very high": lambda x: np.interp(x, vr_range, vr_very_high),
        },
        "PRI": {
            "Narrow": lambda x: np.interp(x, pri_range, pri_narrow),
            "Normal": lambda x: np.interp(x, pri_range, pri_normal),
            "Broad": lambda x: np.interp(x, pri_range, pri_broad),
        },
        "QRSd": {
            "Narrow": lambda x: np.interp(x, qrs_range, qrs_narrow),
            "Normal": lambda x: np.interp(x, qrs_range, qrs_normal),
            "Broad": lambda x: np.interp(x, qrs_range, qrs_broad),
        },
        "RR": {
            "Short": lambda x: np.interp(x, rr_range, rr_short),
            "Normal": lambda x: np.interp(x, rr_range, rr_normal),
            "Wide": lambda x: np.interp(x, rr_range, rr_wide),
        },
        "AR": {
            "Slow": lambda x: np.interp(x, ar_range, ar_slow),
            "Normal": lambda x: np.interp(x, ar_range, ar_normal),
            "Little high": lambda x: np.interp(x, ar_range, ar_lh),
            "High": lambda x: np.interp(x, ar_range, ar_high),
            "Very high": lambda x: np.interp(x, ar_range, ar_vh),
            "Extremely high": lambda x: np.interp(x, ar_range, ar_eh),
        },
        "PP": {
            "Short": lambda x: np.interp(x, pp_range, pp_short),
            "Normal": lambda x: np.interp(x, pp_range, pp_normal),
            "Wide": lambda x: np.interp(x, pp_range, pp_wide),
        },
        "PQRS": {
            "Low": lambda x: np.interp(x, pqrs, pqrs_low),
            "Desirable": lambda x: np.interp(x, pqrs, pqrs_desirable),
            "High": lambda x: np.interp(x, pqrs, pqrs_high),
        },
        "RI": {
            "Low": lambda x: np.interp(x, ri_range, ri_low),
            "Desirable": lambda x: np.interp(x, ri_range, ri_desirable),
            "High": lambda x: np.interp(x, ri_range, ri_high),
        },
        "PI": {
            "Low": lambda x: np.interp(x, pi_range, pi_low),
            "Desirable": lambda x: np.interp(x, pi_range, pi_desirable),
            "High": lambda x: np.interp(x, pi_range, pi_high),
        },
        "T": {
            "Negative": lambda x: np.interp(x, twa, twa_neg),
            "Isolated": lambda x: np.interp(x, twa, twa_iso),
            "Positive": lambda x: np.interp(x, twa, twa_pos),
        }
    }

# --- Pomocné funkcie ---
def extract_conditions(rule):
    """Ziskaj podmienky a cieľ z pravidla."""
    cond_match = re.search(r"IF\s*(.*?)\s*THEN", rule)
    then_match = re.search(r"THEN\s*\(class IS ([^)]+)\)", rule)
    if not cond_match or not then_match:
        return None, None
    condition_part = cond_match.group(1)
    conditions = re.findall(r"\((\w+) IS (\w+)\)", condition_part)
    conclusion = then_match.group(1)
    return conditions, conclusion


def fuzzy_inference(input_values, memberships):
    """
    Vykoná inferenciu nad vstupmi pomocou fuzzy pravidiel.
    - input_values: dict s číselnými vstupmi (napr. {"VR": 72, ...})
    - memberships: dict s menami člení a ich funckiami (napr. memberships["VR"]["Slow"] = funkcia)

    Return: predikovaná trieda (napr. "Normal")
    """
    scores = {}  # agregácia mäkkých pravidiel pre každú triedu

    for rule in rules:
        conditions, output_class = extract_conditions(rule)
        if not conditions:
            continue

        degrees = []
        for var, label in conditions:
            mu_func = memberships.get(var, {}).get(label)
            val = input_values.get(var)
            if mu_func is None or val is None:
                degrees.append(0)
            else:
                degrees.append(mu_func(val))

        activation = min(degrees)  # Použieme MIN ako AND operátor
        scores[output_class] = max(scores.get(output_class, 0), activation)  # MAX agregácia pre triedu

    # Vyber najviac aktivovanú triedu
    if scores:
        return max(scores.items(), key=lambda x: x[1])[0]
    return "Unknown"


# --- Príklad použitia ---
if __name__ == "__main__":
    from fuzzy_classifier_arythm import fuzzy_inference, get_membership_functions, class_labels

    test_cases = [
        {
            "name": "Case 1",
            "input": {
                "VR": 110, "PRI": 90, "QRSd": 100, "AR": 410,
                "RR": 0.46, "PP": 0.15, "PQRS": 2,
                "RI": 0.7, "PI": 0.9, "T": 1
            },
            "expected": "AFb"
        },
        {
            "name": "Case 2",
            "input": {
                "VR": 114, "PRI": 120, "QRSd": 296, "AR": 20,
                "RR": 0.52, "PP": 0.5, "PQRS": 0,
                "RI": 1.2, "PI": 1.0, "T": -1
            },
            "expected": "VT"
        },
        {
            "name": "Case 3",
            "input": {
                "VR": 40, "PRI": 120, "QRSd": 150, "AR": 91,
                "RR": 1.5, "PP": 0.66, "PQRS": 1.2,
                "RI": 1.5, "PI": 1.0, "T": 1
            },
            "expected": "3B"
        },
        {
            "name": "Case 4",
            "input": {
                "VR": 104.89, "PRI": 164, "QRSd": 94, "AR": 107.14,
                "RR": 0.572, "PP": 0.56, "PQRS": 1.0,
                "RI": 1.2, "PI": 1.0, "T": 1
            },
            "expected": "2B2"
        },
        {
            "name": "Case 5",
            "input": {
                "VR": 132, "QRSd": 80,
                "RR": 0.457,  "PQRS": 1.6,
                "RI": 1.4, "T": 1
            },
            "expected": "ST"
        },
        {
            "name": "Case 6",
            "input": {
                "VR": 57.25, "PRI": 185, "QRSd": 73.2, "AR": 57.25,
                "RR": 1.048, "PP": 1.048, "PQRS": 1.0,
                "RI": 1.0, "PI": 1.0, "T": 1
            },
            "expected": "Normal"
        },
        {
            "name": "Case 7",
            "input": {
                "VR": 61.37, "PRI": 168, "QRSd": 68, "AR": 61.37,
                "RR": 0.97, "PP": 0.97, "PQRS": 1.0,
                "RI": 1.0, "PI": 1.0, "T": 1
            },
            "expected": "Normal"
        },
        {
            "name": "Case 8",
            "input": {
                "VR": 91.3, "PRI": 152, "QRSd": 88, "AR": 90.2,
                "RR": 0.66, "PP": 0.67, "PQRS": 1.0,
                "RI": 1.0, "PI": 1.0, "T": 1
            },
            "expected": "Normal"
        },
        {
            "name": "Case 9",
            "input": {
                "VR": 41.5, "PRI": 150, "QRSd": 98, "AR": 40,
                "RR": 1.45, "PP": 1.49, "PQRS": 1.0,
                "RI": 1.0, "PI": 1.0, "T": 1
            },
            "expected": "SB"
        },
        {
            "name": "Case 10",
            "input": {
                "VR": 61.3, "PRI": 134, "QRSd": 65, "AR": 59,
                "RR": 0.98, "PP": 0.99, "PQRS": 1.0,
                "RI": 1.0, "PI": 1.0, "T": 1
            },
            "expected": "Normal"
        }
    ]

    memberships = get_membership_functions()

    for case in test_cases:
        predicted = fuzzy_inference(case["input"], memberships)
        label = class_labels.get(predicted, predicted)
        expected = class_labels.get(case["expected"], case["expected"])
        print(f"{case['name']} → Predicted: {label}, Expected: {expected}")

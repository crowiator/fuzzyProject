import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
import skfuzzy as fuzz
from sklearn.metrics import f1_score
from deap import base, creator, tools, algorithms
from skfuzzy import control as ctrl
import multiprocessing as mp
from classifiers.fuzzy_classifier import FuzzyClassifier
from joblib import dump
class DataLoader:
    @staticmethod
    def load_csv_folder(folder: str = "mitdb_fuzzy_results"):
        dfs = [pd.read_csv(f) for f in Path(folder).glob("*.csv")]
        df = pd.concat(dfs, ignore_index=True)

        valid_labels = {"normalna", "mierna", "zavazna"}
        df = df[df["TrueLabel"].isin(valid_labels)]

        df = df.rename(columns={"HR_bpm": "HR", "QRS_ms": "QRS", "T_amp": "T"})
        X = df[["HR", "QRS", "T"]].values
        y = df["TrueLabel"].map({"normalna": 0, "mierna": 1, "zavazna": 2}).values

        assert not np.isnan(y).any(), "NaN labels found after preprocessing!"
        return X, y

    @staticmethod
    def load_csv(path: str):
        df = pd.read_csv(path)
        df["TrueLabel"] = df["TrueLabel"].fillna("unknown").replace("", "unknown")
        df = df.rename(columns={"HR_bpm": "HR", "QRS_ms": "QRS", "T_amp": "T"})
        X = df[["HR", "QRS", "T"]].values
        y = df["TrueLabel"].map({"normalna": 0, "mierna": 1, "zavazna": 2}).fillna(3).astype(int).values
        return X, y


class FuzzyOptimizer:
    def __init__(self, base_clf: FuzzyClassifier, X, y_true):
        self.base_clf = base_clf
        self.X = X
        self.y_true = y_true

        self.GENE_INFO = [
            ("HR", 1), ("HR", 2), ("HR", 4),
            ("QRS", 1), ("QRS", 2), ("QRS", 4),
            ("TWA", 1), ("TWA", 2), ("TWA", 4),
            ("ARR", 1), ("ARR", 4), ("ARR", 5),
        ]
        self.BOUNDS = [(-0.25, 0.25)] * len(self.GENE_INFO)

    @staticmethod
    def get_abc(term):
        m, u = term.mf, term.parent.universe
        idx_pos = np.where(m > 0)[0]
        return np.array([u[idx_pos[0]], u[np.argmax(m)], u[idx_pos[-1]]])

    @staticmethod
    def shift_node(term, abc_new):
        term.mf = fuzz.trimf(term.parent.universe, np.sort(abc_new))

    def build_clf_from_genome(self, genome):
        clf = deepcopy(self.base_clf)
        for delta, (varname, pt_idx) in zip(genome, self.GENE_INFO):
            var = getattr(clf, varname.lower()) if varname != "ARR" else clf.arr
            term_name = list(var.terms.keys())[pt_idx // 3]
            term = var[term_name]
            abc = self.get_abc(term)
            abc[pt_idx % 3] += delta * np.ptp(var.universe)
            self.shift_node(term, abc)
        clf.diagnosis_ctrl = ctrl.ControlSystem(clf.diagnosis_ctrl.rules)
        return clf

    def evaluate(self, individual):
        clf = self.build_clf_from_genome(individual)
        label_map = {"normalna": 0, "mierna": 1, "zavazna": 2}
        preds = []
        for h, q, t in self.X:
            res = clf.predict(h, q, t)
            if isinstance(res, dict):
                lbl = res.get("label", "unknown")
            elif isinstance(res, (list, tuple)) and len(res) >= 2:
                lbl = res[1]
            else:
                lbl = "unknown"
            preds.append(label_map.get(lbl, 3))

        macro = f1_score(self.y_true, preds, labels=[0, 1, 2], average="macro", zero_division=0)
        return macro,

    def run_ga(self, pop_size=50, n_gen=50, n_proc=8):
        if "FitnessMax" not in creator.__dict__:
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        for i, (lo, hi) in enumerate(self.BOUNDS):
            toolbox.register(f"gene{i}", np.random.uniform, lo, hi)

        toolbox.register("individual", tools.initCycle,
                         creator.Individual,
                         tuple(getattr(toolbox, f"gene{i}")
                               for i in range(len(self.BOUNDS))), n=1)
        toolbox.register("population", tools.initRepeat,
                         list, toolbox.individual)
        toolbox.register("evaluate", self.evaluate)
        toolbox.register("mate", tools.cxBlend, alpha=.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=.1, indpb=.3)
        toolbox.register("select", tools.selTournament, tournsize=3)

        ctx = mp.get_context("fork")  # alebo "spawn" ak preferuješ
        pool = ctx.Pool(processes=n_proc)
        toolbox.register("map", pool.map)

        pop = toolbox.population(pop_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean);
        stats.register("max", np.max)

        algorithms.eaSimple(pop, toolbox, cxpb=.6, mutpb=.4, ngen=n_gen,
                            stats=stats, halloffame=hof, verbose=True)

        pool.close();
        pool.join()  # <- dôležité
        return hof[0], hof[0].fitness.values[0]
def export_mf_nodes(clf, csv_path="mf_nodes.csv"):
    rows = []
    for var_name in ["hr", "qrs", "twa", "arr"]:
        var = getattr(clf, var_name) if var_name != "arr" else clf.arr
        for term_name, term in var.terms.items():
            u = term.parent.universe
            m = term.mf
            a, c = u[m>0][0], u[m>0][-1]
            b = u[np.argmax(m)]
            rows.append([var_name.upper(), term_name, a, b, c])
    pd.DataFrame(rows, columns=["var", "term", "a", "b", "c"]).to_csv(csv_path, index=False)

             # uloží mf_nodes.csv
if __name__ == "__main__":
    mp.set_start_method("fork", force=True)   # macOS safe start

    X, y_true = DataLoader.load_csv_folder(
        "mitdb_fuzzy_results/")

    optimizer = FuzzyOptimizer(FuzzyClassifier(), X, y_true)
    best, score = optimizer.run_ga(pop_size=10, n_gen=5, n_proc=4)

    print("\nBest genome Δ%:", np.round(best, 4))
    print("Macro-F1:", round(score, 4))

    # ----------  DOPLNENÝ BLOK – baseline porovnanie -------------
    baseline_clf = FuzzyClassifier()  # pôvodný model
    label_map = {"normalna": 0, "mierna": 1, "zavazna": 2}

    y_pred_base = [label_map.get(baseline_clf.predict(h, q, t)["label"], 3)
                   for h, q, t in X]

    baseline_f1 = f1_score(y_true, y_pred_base,
                           labels=[0, 1, 2],
                           average="macro",
                           zero_division=0)

    print("Macro-F1 baseline:", round(baseline_f1, 4))
    print("Zlepšenie:", round(score - baseline_f1, 4))
    np.save("best_genome.npy", np.asarray(best, dtype=np.float32))
    print("→ Genóm uložený do best_genome.npy")
    baseline_clf = FuzzyClassifier()  # pôvodný model
    label_map = {"normalna": 0, "mierna": 1, "zavazna": 2}

    y_pred_base = [label_map.get(baseline_clf.predict(h, q, t)["label"], 3)
                   for h, q, t in X]

    baseline_f1 = f1_score(y_true, y_pred_base,
                           labels=[0, 1, 2],
                           average="macro",
                           zero_division=0)

    print("Macro-F1 baseline:", round(baseline_f1, 4))
    print("Zlepšenie:", round(score - baseline_f1, 4))
    np.save("best_genome.npy", np.asarray(best, dtype=np.float32))
    print("→ Genóm uložený do best_genome.npy")

    tuned_clf = optimizer.build_clf_from_genome(best)  # ← oprava tu!
    dump(tuned_clf, "fuzzy_tuned.pkl")
    print("→ Tunovaný FuzzyClassifier uložený do fuzzy_tuned.pkl")

    export_mf_nodes(tuned_clf)
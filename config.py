# config.py – Globálne nastavenia projektu
from pathlib import Path
# Cesta k dátam
DATA_PATH = "./data/mit/"

# Zoznam záznamov z MIT-BIH databázy (možno rozšíriť podľa potreby)
ALL_RECORDS = [
    "100","101","102","103","104","105","106","107","108","109",
    "111","112","113","114","115","116","117","118","119",
    "121","122","123","124","200","201","202","203","205","207",
    "208","209","210","212","213","214","215","217","219","220",
    "221","222","223","228","230","231","232","233","234"
]
RECORD_NAMES2 = [
    "100",
]

# Parametre signálového spracovania
LOWPASS_CUTOFF = 40
DWT_THRESHOLD = 0.15
QRS_THRESHOLD = 0.35
MOVING_WINDOW_SIZE = 10

# Segmentácia – trvanie okolo R-vlny (v sekundách)
SEGMENT_PRE_R = 0.2
SEGMENT_POST_R = 0.4

# Tolerancia pre matching R-vrcholu (v ms)
MATCHING_TOLERANCE = 50

# Testovací pomer pre train/test split
TEST_SIZE = 0.2

# Random seed (pre reprodukovateľnosť)
RANDOM_STATE = 42



# Hlavný priečinok projektu
BASE_DIR = Path(__file__).resolve().parent
MIT_DATA_PATH = BASE_DIR / "data" / "mit"
RESULTS_DIR = BASE_DIR / "results"
REPORTS_DIR = RESULTS_DIR / "reports"
DATA_CACHE_DIR = RESULTS_DIR / "data"

# Cache súbory
FUZZY_FEATURE_CACHE = DATA_CACHE_DIR / "fuzzy_feature_cache.npz"
FUZZY_FEATURE_CACHE_CAMFEA = DATA_CACHE_DIR / "fuzzy_feature_cache_camfea.npz"
CNN_SEGMENT_CACHE = DATA_CACHE_DIR / "cnn_segments_fuzzy.npz"
CNN_RESULTS_CSV = REPORTS_DIR / "cnn_vs_hybrid_comparison.csv"
QRS_COMPARISON_CSV = REPORTS_DIR / "qrs_comparison.csv"


MIT_LOCAL_DIR = Path(__file__).resolve().parent / "data" / "mit"
##new thigns
LEAD = "MLII"
OUT_DIR = Path("mitdb_fuzzy_results")   # každé CSV pôjde sem
FEAT_DIR = Path("exported_features")
MF_CFG_PATH = BASE_DIR / "mf_params_all.json"

OUT_TRAIN_FUZZY = BASE_DIR / "exported_features_fuzzy"
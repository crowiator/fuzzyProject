# config.py – Globálne nastavenia projektu
from pathlib import Path
# Cesta k dátam
DATA_PATH = "./data/mit/"

# Zoznam záznamov z MIT-BIH databázy (možno rozšíriť podľa potreby)
RECORD_NAMES = [
    "100", "101", "102", "103", "104", "106", "107", "108", "109",
    "111", "112", "113", "115", "116", "117", "118", "119",
    "121", "122", "123", "124", "200", "201", "202", "205",
    "207", "208", "209",  "212", "213", "214", "217", "219",
    "220", "221", "222", "223", "228", "230", "231", "232",
    "233", "234"
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
CNN_SEGMENT_CACHE = DATA_CACHE_DIR / "cnn_segments_fuzzy.npz"
CNN_RESULTS_CSV = REPORTS_DIR / "cnn_vs_hybrid_comparison.csv"
QRS_COMPARISON_CSV = REPORTS_DIR / "qrs_comparison.csv"
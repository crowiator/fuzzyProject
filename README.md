# FuzzyProject

## Popis

Tento projekt je zameraný na klasifikáciu biomedicínskych signálov (EKG) pomocou fuzzy logického klasifikátora, ktorý využíva hybridný prístup s tradičnými klasifikačnými algoritmami.

## Projekt obsahuje

- Jednoduchý fuzzy klasifikátor založený na fuzzy logike.
- Tradičné klasifikačné algoritmy (Decision Tree, Random Forest, SVM, kNN).
- Hybridné modely kombinujúce fuzzy výstupy s tradičnými metódami.

## Štruktúra projektu

```
fuzzyProject/
├── classifiers/
│   ├── fuzzy_classifier.py
│   └── traditional_models.py
├── data/
├── preprocessing/
│   ├── annotation_mapping.py
│   ├── feature_extraction.py
│   ├── filtering.py
│   ├── fuzzy_feature_loader.py
│   └── load.py
├── results/
│   ├── figures/
│   ├── plots/
│   ├── reports/
│   │   ├── hybrid/
│   │   ├── traditional/
│   │   ├── best_params.csv
│   │   └── fuzzy_classification_results.csv
├── utils/
├── venv/
├── config.py
├── fuzzy_predictions.csv
├── main.py
├── optimalization.py
└── vv.py
```

## Požiadavky

Projekt používa Python. Požadované balíčky nájdete v súbore `requirements.txt`.

## Spustenie

Aktivujte virtuálne prostredie:

```bash
source venv/bin/activate
```

Spustite hlavný skript:

```bash
python main.py
```

## Hlavné súbory

- `main.py` - Spúšťa celkový proces klasifikácie a hodnotenia modelov.
- `fuzzy_classifier.py` - Implementácia fuzzy klasifikátora.
- `traditional_models.py` - Tradičné klasifikačné algoritmy a ich trénovanie.
- `fuzzy_feature_loader.py` - Načítanie alebo extrakcia fuzzy čŕt.
- `optimalization.py` - Skript pre optimalizáciu parametrov modelov.

## Výsledky

Výsledky klasifikácie sa ukladajú do priečinka `results/reports` vo formáte CSV a PNG (Confusion Matrix, Classification Report).

## Poznámky

- Pred spustením modelov skontrolujte dostupnosť datasetu v priečinku `data`.
- Výstupy klasifikátora a metriky sú prístupné v adresári `results`.

## Autor

Samuel Vrana

# MINLP-Heart-Disease-Fairness

Sparse, budgeted and fair logistic regression for heart-disease risk, on two datasets (UCI Heart Disease and Z-Alizadeh Sani). Easy problem: L2 logistic regression on all tests. Hard problem: the same loss with at most K tests, a test budget B, and a fairness bound ε between men and women (MINLP).

## Layout
- `data/` raw data (UCI Heart Disease files, Z-Alizadeh Sani xlsx)
- `src/data_prep.py` loading, cleaning, encoding, splits, assumed test costs
- `src/core.py` logistic fit, convex fairness constraint, metrics
- `src/easy.py` easy problem on both datasets, with optimality checks (gradient, Hessian, scikit-learn)
- `src/hard.py` exact subset enumeration and greedy search
- `src/experiments.py` all experiments for one dataset
- `src/plots.py` figures
- `results/` JSON results and `results/figures/`
- `Heart Disease Risk Optimization.pptx`, `Final Report - Heart Disease Risk Optimization.docx` final slides and report

## Run
```
pip install numpy pandas scipy scikit-learn matplotlib openpyxl
cd src
python easy.py                # easy problem + verification (seconds)
python experiments.py uci     # exact; first run builds a cached table (about 1 minute)
python experiments.py za      # greedy search (about 6 minutes)
python plots.py
```

Test costs are assumed illustrative values, not sourced prices.

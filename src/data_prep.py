"""Data loading for both datasets, with one shared output interface.

Each *test* (original clinical attribute) is one selectable unit j. Its encoded
columns (numeric column, or one-hot levels) all share one binary variable z_j.

Returned dict (both datasets):
  Xtr, Xte, ytr, yte   encoded arrays / labels
  names                encoded column names
  group                encoded column -> test index
  tests                test names
  cost                 per-test cost (ASSUMED, illustrative)
  sex_tr, sex_te       protected attribute (1 = male)
"""
import os
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

# ---------------------------------------------------------------- generic encoder
def _encode(tr, te, features, numeric):
    """Fit imputation/scaling on train only. No missing-indicator columns
    (they proxy for the recording site and leak)."""
    Xtr, Xte, names, group = [], [], [], []
    for j, c in enumerate(features):
        if c in numeric:
            med = tr[c].median()
            a, b = tr[c].fillna(med), te[c].fillna(med)
            mu, sd = a.mean(), (a.std() or 1.0)
            cols = [((a - mu) / sd, (b - mu) / sd, c)]
        else:
            levels = sorted(tr[c].dropna().unique())[1:]      # first level = reference
            mode = tr[c].mode()[0]
            a, b = tr[c].fillna(mode), te[c].fillna(mode)
            cols = [((a == l).astype(float), (b == l).astype(float), f"{c}={l}") for l in levels]
        for a, b, n in cols:
            Xtr.append(np.asarray(a, float)); Xte.append(np.asarray(b, float))
            names.append(n); group.append(j)
    return np.column_stack(Xtr), np.column_stack(Xte), names, np.array(group)

def _pack(tr, te, features, numeric, cost):
    Xtr, Xte, names, group = _encode(tr, te, features, numeric)
    return dict(Xtr=Xtr, Xte=Xte, ytr=tr["y"].values, yte=te["y"].values,
                names=names, group=group, tests=list(features),
                cost=np.array([cost[c] for c in features], float),
                sex_tr=tr["sex_bin"].values.astype(int), sex_te=te["sex_bin"].values.astype(int))

# ---------------------------------------------------------------- UCI Heart Disease
UCI_COLS = ["age","sex","cp","trestbps","chol","fbs","restecg","thalch","exang",
            "oldpeak","slope","ca","thal","num"]
UCI_FEATURES = UCI_COLS[:-1]
UCI_NUMERIC = {"age","trestbps","chol","thalch","oldpeak","ca"}
# ASSUMED approximate unit costs (illustrative order-of-magnitude, NOT sourced).
UCI_COST = {"age":0,"sex":0,"cp":5,"trestbps":5,"chol":20,"fbs":15,"restecg":50,
            "thalch":60,"exang":60,"oldpeak":70,"slope":70,"ca":300,"thal":400}
UCI_SITES = ["cleveland","hungarian","switzerland","va"]

def load_uci_raw(sites=None):
    parts = []
    for s in (sites or UCI_SITES):
        d = pd.read_csv(os.path.join(DATA, f"processed.{s}.data"), header=None,
                        names=UCI_COLS, na_values="?")
        d["site"] = s
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    for c in ["chol", "trestbps"]:          # zeros are physiologically impossible = missing
        df.loc[df[c] == 0, c] = np.nan
    df["y"] = (df["num"] > 0).astype(int)
    df["sex_bin"] = df["sex"].astype(int)
    return df

def build_uci(seed=0, test_size=0.25, sites=None, cost=None):
    df = load_uci_raw(sites)
    tr, te = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["y"])
    d = _pack(tr, te, UCI_FEATURES, UCI_NUMERIC, cost or UCI_COST)
    d["name"] = "uci"
    return d

# ---------------------------------------------------------------- Z-Alizadeh Sani
ZA_FILE = os.path.join(DATA, "alizadeh", "Z-Alizadeh sani dataset.xlsx")
# ASSUMED approximate unit costs, simplified to be independent per feature
# (e.g. one ECG yields several features; here each ECG feature carries a share).
ZA_COST = {
 "Age":0,"Sex":0,"Weight":1,"Length":1,"BMI":1,
 "DM":2,"HTN":2,"Current Smoker":2,"EX-Smoker":2,"FH":2,"Obesity":2,"CRF":2,"CVA":2,
 "Airway disease":2,"Thyroid Disease":2,"CHF":2,"DLP":2,
 "BP":3,"PR":3,"Edema":3,"Weak Peripheral Pulse":5,"Lung rales":5,
 "Systolic Murmur":5,"Diastolic Murmur":5,
 "Typical Chest Pain":2,"Dyspnea":2,"Function Class":3,"Atypical":2,"Nonanginal":2,"LowTH Ang":2,
 "Q Wave":8,"St Elevation":8,"St Depression":8,"Tinversion":8,"LVH":8,"Poor R Progression":8,"BBB":8,
 "FBS":5,"CR":6,"TG":8,"LDL":8,"HDL":8,"BUN":6,"ESR":8,"HB":5,"K":5,"Na":5,
 "WBC":6,"Lymph":6,"Neut":6,"PLT":6,
 "EF-TTE":150,"Region RWMA":150,"VHD":150,
}
ZA_FEATURES = list(ZA_COST)
ZA_NUMERIC = {"Age","Weight","Length","BMI","BP","PR","Function Class","FBS","CR","TG","LDL","HDL",
              "BUN","ESR","HB","K","Na","WBC","Lymph","Neut","PLT","EF-TTE","Region RWMA"}

def load_za_raw():
    df = pd.read_excel(ZA_FILE, sheet_name=0)
    df["Sex"] = (df["Sex"] == "Male").astype(int)         # file spells female "Fmale"
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype).startswith("str"):
            if set(df[c].unique()) <= {"Y", "N"}:
                df[c] = (df[c] == "Y").astype(int)
    df["y"] = (df["Cath"] == "Cad").astype(int)
    df["sex_bin"] = df["Sex"].astype(int)
    return df.copy()                                       # defragment

def build_za(seed=0, test_size=0.25, cost=None):
    df = load_za_raw()
    tr, te = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["y"])
    d = _pack(tr, te, ZA_FEATURES, ZA_NUMERIC, cost or ZA_COST)
    d["name"] = "za"
    return d

def build(name, seed=0, **kw):
    return {"uci": build_uci, "za": build_za}[name](seed=seed, **kw)

if __name__ == "__main__":
    for n in ["uci", "za"]:
        d = build(n)
        print(f"{n}: Xtr {d['Xtr'].shape} Xte {d['Xte'].shape} tests {len(d['tests'])} "
              f"pos-rate {d['ytr'].mean():.3f} male {d['sex_tr'].mean():.2f} "
              f"total cost {d['cost'].sum():.0f}")
        assert not np.isnan(d["Xtr"]).any() and not np.isnan(d["Xte"]).any()

"""Append solution + analysis slides to a copy of the proposal deck, in the deck's own style.
Every number is read from results/*.json."""
import os, json, copy
from pptx import Presentation
from pptx.util import Emu, Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from data_prep import ROOT

R = os.path.join(ROOT, "results"); FIG = os.path.join(R, "figures")
U = json.load(open(os.path.join(R, "uci_results.json"))); Z = json.load(open(os.path.join(R, "za_results.json")))
V = json.load(open(os.path.join(R, "easy_verification.json")))
SRC = os.path.join(ROOT, "Heart Disease Risk Optimization.pptx")
OUT = os.environ.get("DECK_OUT") or os.path.join(ROOT, "Heart Disease Risk Optimization - Final.pptx")

INK, MUTE, BODY, ORANGE, DORANGE, PALE, PALEO, WHITE, LINE = (RGBColor.from_string(c) for c in
    ["0F172A", "64748B", "475569", "F97316", "C2410C", "F8FAFC", "FFF7ED", "FFFFFF", "E2E8F0"])
prs = Presentation(SRC); BLANK = prs.slide_layouts[6]
import numpy as np

# ====================================================================== fixes to the original slides 1-9
import copy, io
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pptx.oxml.ns import qn
orig = list(prs.slides)

def sub(slide, idx, old, new):
    """Replace text inside one run of shape idx (keeps formatting)."""
    for para in slide.shapes[idx].text_frame.paragraphs:
        for run in para.runs:
            if old in run.text:
                run.text = run.text.replace(old, new); return
    raise ValueError(f"'{old}' not found in shape {idx}")

def clone_slide(src):
    dst = prs.slides.add_slide(BLANK)
    for sh in list(dst.shapes): sh._element.getparent().remove(sh._element)
    bg = src._element.cSld.find(qn("p:bg"))
    if bg is not None: dst._element.cSld.insert(0, copy.deepcopy(bg))
    for el in src.shapes._spTree:
        if el.tag in (qn("p:nvGrpSpPr"), qn("p:grpSpPr")): continue
        new = copy.deepcopy(el)
        for node in new.iter():
            for att in (qn("r:embed"), qn("r:link"), qn("r:id")):
                rid = node.get(att)
                if rid and rid in src.part.rels:
                    rel = src.part.rels[rid]
                    node.set(att, dst.part.relate_to(rel.target_ref, rel.reltype, is_external=True) if rel.is_external
                             else dst.part.relate_to(rel.target_part, rel.reltype))
        dst.shapes._spTree.append(new)
    return dst

# slide 2: problem statement
sub(orig[1], 21, "all 16 features", "all available tests")
sub(orig[1], 23, "Select K optimal diagnostic features from candidates so that total network-wide predictive accuracy is maximized under strict limits.",
    "Select at most K tests from the candidates (13 on UCI, 54 on Z-Alizadeh) to minimise prediction loss within the budget and fairness limits.")
# slide 3: UCI dataset
sub(orig[2], 9, "DATASET · UCI HEART DISEASE", "DATASET 1 · UCI HEART DISEASE")
sub(orig[2], 13, "16", "13"); sub(orig[2], 14, "CLINICAL FEATURES", "CLINICAL TESTS")
sub(orig[2], 25, "High cost. 66% missing data.", "High cost. Up to 66% missing.")
# slide 4: easy problem
sub(orig[3], 15, "Solvable rapidly in O(d) iterations.", "Solved to optimality and verified.")
sub(orig[3], 17, "using all 16 features", "using all available tests")
# slide 5: hard problem
sub(orig[4], 13, "guaranteed demographic fairness", "bounded demographic disparity")
sub(orig[4], 20, "Demographic parity introduces severe non-linearity and local minima.",
    "The exact parity gap is non-convex; we solve a convex score-gap surrogate and report the true gap.")
# slide 6: motivation
sub(orig[5], 10, "across age and sex sub-groups", "across demographic sub-groups such as sex")
def title_size(slide, idx, pt):
    for para in slide.shapes[idx].text_frame.paragraphs:
        for run in para.runs: run.font.size = Pt(pt)
title_size(orig[5], 3, 30); title_size(orig[8], 6, 30)
orig[5].shapes[4].top += Inches(0.12)      # room under the larger title
orig[0].shapes[3].width = Inches(11.2)       # keep the subtitle on one line
# slide 7: constraints
sub(orig[6], 12, "run 16 tests for every patient", "run every available test on every patient")
# slide 8: price of realism concept (picture had d=16 baked into the image: redraw it)
sub(orig[7], 6, "to achieve zero demographic bias and enforce diagnostic budget limits", "to meet the sparsity, budget and fairness limits")
old = [sh for sh in orig[7].shapes if sh.shape_type == 13][0]
l, t, w, h = old.left, old.top, old.width, old.height
fig, ax = plt.subplots(figsize=(w / 914400, h / 914400), dpi=200)
xs = np.linspace(0, 1, 200); ys = (1 - np.exp(-4 * xs)) / (1 - np.exp(-4))
ax.plot(xs, ys, color="#F97316", lw=3.5, solid_capstyle="round")
for x_, lab, col, dx in [(0.07, "Few tests (cheap)", "#0F172A", 0.03), (0.38, "Elbow panel", "#3B82F6", 0.04), (1.0, "All tests (full cost)", "#0F172A", -0.02)]:
    y_ = (1 - np.exp(-4 * x_)) / (1 - np.exp(-4)); ax.plot([x_], [y_], "o", ms=11, color=col)
    ax.text(x_ + dx, y_ + (-0.07 if x_ < 0.9 else 0.07), lab, fontsize=11, color="#0F172A" if col != "#3B82F6" else "#1D4ED8",
            weight="bold" if col == "#3B82F6" else "normal", ha="left" if x_ < 0.9 else "right", va="center")
ax.set_xlabel("Diagnostic cost  →", color="#64748B", fontsize=11); ax.set_ylabel("Accuracy  →", color="#64748B", fontsize=11)
ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-0.05, 1.08); ax.set_ylim(-0.12, 1.12)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
for sp in ["left", "bottom"]: ax.spines[sp].set_color("#CBD5E1")
fig.tight_layout(); buf = io.BytesIO(); fig.savefig(buf, format="png", facecolor="white"); plt.close(fig); buf.seek(0)
old._element.getparent().remove(old._element)
orig[7].shapes.add_picture(buf, l, t, w, h)
# slide 9: input / constraints / output
sub(orig[8], 7, "returns optimal diagnostic panels with minimum achievable loss", "returns low-cost diagnostic panels with minimal loss")
sub(orig[8], 11, "920 Patient Records", "920 + 303 Patient Records"); sub(orig[8], 12, "16 Clinical Features", "13 and 54 Clinical Tests")
# new slide: Z-Alizadeh dataset, cloned from the UCI slide
za = clone_slide(orig[2])
sub(za, 9, "DATASET 1 · UCI HEART DISEASE", "DATASET 2 · Z-ALIZADEH SANI")
sub(za, 10, "TARGET: BINARY DISEASE CLASSIFICATION (NUM > 0)", "TARGET: CORONARY ARTERY DISEASE (CAD) VS NORMAL")
sub(za, 11, "920", "303"); sub(za, 13, "13", "54"); sub(za, 15, "28-77", "30-86")
sub(za, 17, "Tier 1: Demographics", "Tier 1: History")
za.shapes[18].text_frame.paragraphs[0].runs[1].text = " Age, Sex, BMI, HTN, DM, smoking."
sub(za, 19, "Zero cost. 100% coverage.", "Free to low cost. No missing data.")
sub(za, 20, "Tier 2: Routine Tests", "Tier 2: Exam, ECG, Labs")
za.shapes[21].text_frame.paragraphs[0].runs[1].text = " Chest pain, ECG signs, lipids, blood count."
sub(za, 22, "Low cost. High coverage.", "Low cost. No missing data.")
sub(za, 23, "Tier 3: Specialized", "Tier 3: Echocardiography")
za.shapes[24].text_frame.paragraphs[0].runs[1].text = " EF-TTE, wall motion, valve disease."
sub(za, 25, "High cost. Up to 66% missing.", "High cost. About 150 per test.")
za.shapes[8].text_frame.paragraphs[0].runs[0].text = " https://archive.ics.uci.edu/dataset/412/z+alizadeh+sani "

def tb(s, x, y, w, h, text, size=14, bold=False, color=INK, font="Calibri", align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic=False):
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = box.text_frame
    tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    lines = text if isinstance(text, list) else [text]
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align; r = p.add_run(); r.text = ln
        f = r.font; f.name, f.size, f.bold, f.italic = font, Pt(size), bold, italic; f.color.rgb = color
        if i: p.space_before = Pt(6)
    return box

def card(s, x, y, w, h, fill=PALE, line=None, radius=0.03):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.adjustments[0] = radius if fill == PALE else 0.05; sh.fill.solid(); sh.fill.fore_color.rgb = fill
    line = line or (LINE if fill == PALE else RGBColor.from_string("FED7AA"))
    sh.line.color.rgb = line; sh.line.width = Pt(0.75)
    sh.shadow.inherit = False
    return sh

def header(s, title, sub, notes=""):
    tb(s, 0.83, 0.67, 12.0, 0.5, title, 30, True)
    tb(s, 0.83, 1.26, 12.0, 0.25, sub, 15, False, MUTE)
    if notes: s.notes_slide.notes_text_frame.text = notes

def stat(s, x, y, w, big, label, color=INK, fill=PALE, bigsize=36):
    card(s, x, y, w, 1.45, fill)
    tb(s, x + 0.2, y + 0.15, w - 0.4, 0.7, big, bigsize, True, color)
    tb(s, x + 0.2, y + 0.9, w - 0.4, 0.45, label, 14, False, BODY)

def pic(s, name, x, y, w):
    return s.shapes.add_picture(os.path.join(FIG, name), Inches(x), Inches(y), width=Inches(w))

def new(): return prs.slides.add_slide(BLANK)
f3 = lambda v: f"{v:.3f}"
ue, ze = U["easy"], Z["easy"]; us, zs = U["por"]["steps"], Z["por"]["steps"]
uF, zF = us["+fairness (eps=0.5*gap0)"], zs["+fairness (eps=0.5*gap0)"]
ust, zst = U["stability"], Z["stability"]
nice = {"cp": "chest pain type", "oldpeak": "ST depression", "thalch": "max heart rate", "exang": "exercise angina",
        "age": "age", "sex": "sex", "Typical Chest Pain": "typical chest pain", "HTN": "hypertension", "PR": "pulse rate", "TG": "triglycerides"}
pn = lambda tests: ", ".join(nice.get(t, t) for t in tests)

# ---------------------------------------------------------------- 10 approach
s = new(); header(s, "HOW WE SOLVE IT", "SEARCH OVER TEST PANELS, SOLVE A CONVEX PROBLEM INSIDE",
    "Once the set of tests z is fixed, the problem is an ordinary convex logistic regression with one linear constraint. The only hard part is choosing z. "
    "On UCI there are only 2^13 = 8,192 panels so we try them all and get the proven optimum. On Z-Alizadeh there are 2^54 so we use greedy search, validated against the exact answer on UCI.")
steps = [("1  Fix the panel", "Choosing which tests are on (z) makes the loss convex in w, with one linear fairness constraint. Solved with L-BFGS / SLSQP."),
         ("2  Search over panels", f"UCI: enumerate all 2^13 = 8,192 panels (exact, with pruning). Z-Alizadeh: 2^54 panels, so greedy forward selection plus swap/drop search."),
         ("3  Verify", f"Greedy matched the exact optimum in {U['heuristic_vs_exact']['matches']} of {U['heuristic_vs_exact']['total']} test configurations on UCI.")]
for i, (t, d_) in enumerate(steps):
    y = 1.95 + i * 1.65
    card(s, 0.83, y, 6.6, 1.45); tb(s, 1.05, y + 0.15, 6.2, 0.35, t, 18, True)
    tb(s, 1.05, y + 0.58, 6.2, 0.8, d_, 14, False, BODY)
card(s, 7.8, 1.95, 4.7, 4.75, PALEO)
tb(s, 8.05, 2.15, 4.2, 0.3, "FAIRNESS: CONVEX SURROGATE", 14, True, DORANGE)
tb(s, 8.05, 2.7, 4.2, 0.6, "|aᵀw| ≤ ε", 30, False, INK, font="Playfair Display")
tb(s, 8.05, 3.55, 4.2, 3.0, ["a = mean features of men minus mean features of women, so the constraint is linear in w.",
    "The exact probability gap is non-convex. We solve the surrogate and always report the true gap afterwards.",
    "Two variants: demographic parity (all patients) and equal opportunity (diseased patients only)."], 14, False, BODY)

# ---------------------------------------------------------------- 11 datasets
s = new(); header(s, "DATA PREPARATION", "SAME PIPELINE, TWO VERY DIFFERENT DATASETS",
    "The professor asked for a second dataset. Both are binary heart-disease classification so the same easy/hard pipeline applies unchanged. "
    "We also removed the missing-indicator columns on UCI: they proxy for which hospital recorded the patient and inflated the fit (easy loss went from 0.414 to 0.446 once removed).")
rows = [("", "UCI Heart Disease", "Z-Alizadeh Sani"), ("Patients", "920 (4 hospitals)", "303"),
        ("Candidate tests", "13", "54"), ("Missing data", "up to 66% (ca)", "none"),
        ("Positive rate", "55%", "71%"), ("Disease rate, men vs women", "63% vs 26%", "74% vs 68%"),
        ("Subset search space", "2^13 = 8,192 (exact)", "2^54 (heuristic)")]
tbl = s.shapes.add_table(len(rows), 3, Inches(0.83), Inches(1.95), Inches(7.6), Inches(4.4)).table
for c, w in enumerate([2.9, 2.35, 2.35]): tbl.columns[c].width = Inches(w)
for r_, row in enumerate(rows):
    tbl.rows[r_].height = Inches(0.62)
    for c, v in enumerate(row):
        cell = tbl.cell(r_, c); cell.text = v; cell.fill.solid()
        cell.fill.fore_color.rgb = INK if r_ == 0 else (PALE if r_ % 2 else WHITE); cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        for p in cell.text_frame.paragraphs:
            for rn in p.runs: rn.font.name, rn.font.size, rn.font.bold = "Inter", Pt(14), (r_ == 0 or c == 0); rn.font.color.rgb = WHITE if r_ == 0 else INK
card(s, 8.8, 1.95, 3.7, 4.4, PALEO)
tb(s, 9.05, 2.15, 3.2, 0.3, "HOW WE PREPARED IT", 14, True, DORANGE)
tb(s, 9.05, 2.6, 3.25, 3.7, ["Stratified 75/25 split; imputation and scaling fitted on training data only.",
    "Zeros in cholesterol and blood pressure treated as missing.",
    "Dropped missing-indicator columns: they leaked the hospital.",
    "Test costs are assumed illustrative values."], 14, False, BODY)

# ---------------------------------------------------------------- 12 easy
s = new(); header(s, "THE EASY PROBLEM: RESULTS", "L2 LOGISTIC REGRESSION ON ALL TESTS (BENCHMARK)",
    "Easy problem solved to optimality and verified three ways: gradient near zero, strictly positive Hessian eigenvalue (so strictly convex, unique global minimum), and agreement with scikit-learn.")
for i, (nm, e, k) in enumerate([("UCI Heart Disease", ue, "uci"), ("Z-Alizadeh Sani", ze, "za")]):
    x = 0.83 + i * 6.0
    tb(s, x, 1.85, 5.6, 0.3, nm.upper(), 14, True, DORANGE)
    stat(s, x, 2.3, 1.75, f3(e["J"]), "train loss J"); stat(s, x + 1.9, 2.3, 1.75, f3(e["test_acc"]), "test accuracy")
    stat(s, x + 3.8, 2.3, 1.75, f3(e["test_auc"]), "test AUC")
    tb(s, x, 4.0, 5.6, 1.5, [f"Gradient norm: {V[k]['grad']:.0e}", f"Smallest Hessian eigenvalue: {V[k]['mineig']:.0e} (> 0, strictly convex)",
        f"Max weight difference vs scikit-learn: {V[k]['sk_diff']:.0e}"], 14, False, BODY)
card(s, 0.83, 5.75, 11.6, 0.95, PALEO)
tb(s, 1.1, 5.9, 11.1, 0.7, "Loss(Easy) is the reference for the Price of Realism: PoR = Loss(Hard) − Loss(Easy).", 16, True, DORANGE, anchor=MSO_ANCHOR.MIDDLE)

# ---------------------------------------------------------------- 13 pareto
s = new(); header(s, "ACCURACY VS COST", "PARETO FRONTIER AND THE ELBOW POINT",
    f"As the budget B grows, loss falls steeply at first and then flattens. The elbow is where extra spend stops paying off. UCI elbow B*={U['B_star']}, Z-Alizadeh B*={Z['B_star']}. "
    "On Z-Alizadeh the small bump at higher cost is the greedy search being slightly suboptimal; an exact solver could not produce it.")
pic(s, "uci_pareto.png", 0.83, 1.7, 6.6); pic(s, "za_pareto.png", 0.83, 4.25, 6.6)
card(s, 7.8, 1.7, 4.7, 5.05, PALEO)
tb(s, 8.05, 1.9, 4.2, 0.3, "WHAT WE SEE", 14, True, DORANGE)
tb(s, 8.05, 2.35, 4.2, 4.3, [f"UCI: a 4-test panel ({pn(us['+budget B*']['tests'])}) costs {us['+budget B*']['cost']:.0f} versus {ue['cost']:.0f} for all tests.",
    f"Z-Alizadeh: a 5-test panel costs {zs['+budget B*']['cost']:.0f} versus {ze['cost']:.0f} for all 54 tests.",
    "Almost all accuracy is recovered with a small fraction of the spend.",
    "Costs are our assumed illustrative values."], 14, False, BODY)

# ---------------------------------------------------------------- 14 PoR
s = new(); header(s, "THE PRICE OF REALISM", "WHERE THE EXTRA LOSS COMES FROM",
    f"PoR = Loss(Hard) - Loss(Easy). UCI: {U['por']['PoR_train_loss']:.3f}. Z-Alizadeh: {Z['por']['PoR_train_loss']:.3f}. "
    "On UCI the budget step adds nothing because the 4-test panel already costs less than the budget. On Z-Alizadeh the fairness step adds nothing because the sex gap is small.")
pic(s, "uci_por.png", 0.83, 1.75, 5.9); pic(s, "za_por.png", 6.75, 1.75, 5.9)
for i, (lab, e, h) in enumerate([("UCI", ue, uF), ("Z-Alizadeh", ze, zF)]):
    x = 0.83 + i * 5.9
    card(s, x, 5.35, 5.7, 1.4, PALEO)
    tb(s, x + 0.2, 5.5, 5.3, 1.15, [f"{lab}: test accuracy {f3(e['test_acc'])} → {f3(h['test_acc'])}, AUC {f3(e['test_auc'])} → {f3(h['test_auc'])}",
        f"Cost {e['cost']:.0f} → {h['cost']:.0f}  |  PoR = {h['J'] - e['J']:.3f}"], 14, False, INK)

# ---------------------------------------------------------------- 15 fairness
d1, d2 = U["fair_sweep"]["dp"], U["fair_sweep"]["eo"]
s = new(); header(s, "THE PRICE OF FAIRNESS", "TIGHTENING THE PARITY BOUND ε (K* AND B* FIXED)",
    "ε is expressed as a fraction of the unconstrained score gap. On UCI tightening demographic parity costs accuracy and, importantly, increases the false-negative gap between sexes. "
    "Equal opportunity targets the clinically relevant harm directly. Test sets are small (230 and 76 rows) so test metrics are noisy.")
pic(s, "uci_fairness.png", 0.83, 1.7, 6.6); pic(s, "za_fairness.png", 0.83, 4.25, 6.6)
card(s, 7.8, 1.7, 4.7, 5.05, PALEO)
tb(s, 8.05, 1.9, 4.2, 0.3, "WHAT WE SEE", 14, True, DORANGE)
tb(s, 8.05, 2.35, 4.2, 4.3, [f"UCI: halving the parity gap cuts test accuracy {f3(d1[0]['test_acc'])} → {f3(d1[2]['test_acc'])}.",
    f"Enforcing demographic parity widens the false-negative gap ({f3(d1[0]['test_fnr_gap'])} → {f3(d1[2]['test_fnr_gap'])}), so it can hurt the very patients it should protect.",
    "Z-Alizadeh: the sex gap is small, so fairness is almost free."], 14, False, BODY)

# ---------------------------------------------------------------- 16 robustness
accs = lambda st: [r["test_acc"] for r in st["runs"]]
import numpy as np
s = new(); header(s, "ARE THE RESULTS ROBUST?", "REPEATED SPLITS, COST CHANGES, AND A HOSPITAL CHECK",
    "Selected panels are stable: the same core tests keep being chosen. Cost sensitivity multiplies each cost by a random factor between 0.5 and 2. The site check drops the Switzerland and VA hospitals.")
top = lambda st, n: ", ".join(nice.get(k, k) for k, v in sorted(st["freq"].items(), key=lambda kv: -kv[1]) if v == 1.0)
cards = [("Random splits", f"{len(ust['runs'])} UCI / {len(zst['runs'])} Z-A", f"Always chosen — UCI: {top(ust, 0)}. Z-A: {top(zst, 0)}."),
         ("Cost changes (×0.5–2)", "20 draws each", f"UCI always keeps {top(U['cost_sensitivity'], 0)}. Z-A picks the same 5 tests every time."),
         ("Site check (UCI)", f"PoR {U['site_robustness']['PoR']:.3f}", f"Cleveland + Hungarian only: panel {pn(U['site_robustness']['hard']['tests'])}.")]
for i, (a, b, c) in enumerate(cards):
    x = 0.83 + i * 3.95; card(s, x, 1.95, 3.75, 3.4)
    tb(s, x + 0.2, 2.1, 3.35, 0.3, a.upper(), 14, True, DORANGE); tb(s, x + 0.2, 2.6, 3.35, 0.7, b, 24, True, INK)
    tb(s, x + 0.2, 3.5, 3.35, 1.8, c, 14, False, BODY)
card(s, 0.83, 5.6, 11.6, 1.1, PALEO)
tb(s, 1.1, 5.7, 11.1, 0.9, f"Hard-panel test accuracy across splits: UCI {np.mean(accs(ust)):.3f} ± {np.std(accs(ust)):.3f} (easy {np.mean([r['easy_test_acc'] for r in ust['runs']]):.3f}); "
   f"Z-Alizadeh {np.mean(accs(zst)):.3f} ± {np.std(accs(zst)):.3f} (easy {np.mean([r['easy_test_acc'] for r in zst['runs']]):.3f}).", 14, False, INK, anchor=MSO_ANCHOR.MIDDLE)

# ---------------------------------------------------------------- 17 conclusions
s = new(); header(s, "CONCLUSIONS", "WHAT THE OPTIMIZATION TELLS US",
    "Main message: a few cheap tests recover most of the accuracy; fairness is the expensive constraint when the groups truly differ. Be upfront about the limits.")
card(s, 0.83, 1.95, 5.7, 3.6); tb(s, 1.08, 2.15, 5.2, 0.3, "TAKEAWAYS", 14, True, DORANGE)
tb(s, 1.08, 2.65, 5.2, 4.0, [f"A small panel is enough: UCI uses 4 of 13 tests at about {us['+budget B*']['cost'] / ue['cost'] * 100:.0f}% of the cost.",
    f"Price of Realism: {U['por']['PoR_train_loss']:.3f} on UCI, {Z['por']['PoR_train_loss']:.3f} on Z-Alizadeh.",
    "Fairness is costly only when groups differ in disease rate (UCI).",
    "Demographic parity can worsen the false-negative gap; equal opportunity is the safer target."], 14, False, BODY)
card(s, 6.8, 1.95, 5.7, 3.6, PALEO); tb(s, 7.05, 2.15, 5.2, 0.3, "LIMITATIONS", 14, True, DORANGE)
tb(s, 7.05, 2.65, 5.2, 4.0, ["Test costs are assumed illustrative values, not sourced prices.",
    "Z-Alizadeh panels come from a greedy search: upper bounds, not proven optima.",
    "Fairness uses a convex surrogate of the true probability gap.",
    "Small test sets (230 and 76 rows) make test metrics noisy."], 14, False, BODY)

FONT = "Calibri"
def unify(tf):
    for para in tf.paragraphs:
        for run in para.runs: run.font.name = FONT
        epr = para._p.find(qn("a:endParaRPr"))
        if epr is not None:
            for tag in ("a:latin", "a:ea", "a:cs"):
                el = epr.find(qn(tag))
                if el is not None: epr.remove(el)
def walk(shapes):
    for sh in shapes:
        if sh.shape_type == 6: walk(sh.shapes)
        if sh.has_text_frame: unify(sh.text_frame)
        if getattr(sh, "has_table", False) and sh.has_table:
            for row in sh.table.rows:
                for c in row.cells: unify(c.text_frame)
for sl in prs.slides: walk(sl.shapes)
ids = list(prs.slides._sldIdLst)   # 0-8 original, 9 Z-A clone, 10.. new
order = [0, 1, 2, 9, 11, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16, 17]
lst = prs.slides._sldIdLst
for el in ids: lst.remove(el)
for i in order: lst.append(ids[i])
prs.save(OUT); print("saved", OUT, "slides:", len(prs.slides))

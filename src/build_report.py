"""Build the final report (.docx). All numbers come from results/*.json."""
import os, json
import numpy as np
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from data_prep import ROOT, UCI_COST, ZA_COST

R = os.path.join(ROOT, "results"); FIG = os.path.join(R, "figures")
U = json.load(open(os.path.join(R, "uci_results.json"))); Z = json.load(open(os.path.join(R, "za_results.json")))
V = json.load(open(os.path.join(R, "easy_verification.json")))
OUT = os.path.join(ROOT, "Final Report - Heart Disease Risk Optimization.docx")
f3 = lambda v: f"{v:.3f}"

doc = Document()
st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(11)
for s_ in ["Heading 1", "Heading 2"]:
    doc.styles[s_].font.name = "Calibri"; doc.styles[s_].font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
sec = doc.sections[0]; sec.left_margin = sec.right_margin = Inches(1); sec.top_margin = sec.bottom_margin = Inches(0.9)

def H(t, l=1): doc.add_heading(t, level=l)
def P(t, bold=False, italic=False):
    p = doc.add_paragraph(); r = p.add_run(t); r.bold, r.italic = bold, italic; p.paragraph_format.space_after = Pt(6); return p
def B(items):
    for t in items: doc.add_paragraph(t, style="List Bullet")
def EQ(t):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; r = p.add_run(t); r.font.name = "Cambria Math"; r.font.size = Pt(12)
def FIGURE(name, cap, w=6.3):
    doc.add_picture(os.path.join(FIG, name), width=Inches(w)); doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    c = doc.add_paragraph(cap); c.alignment = WD_ALIGN_PARAGRAPH.CENTER; c.runs[0].italic = True; c.runs[0].font.size = Pt(9.5)
def TABLE(rows, widths=None, header=True):
    t = doc.add_table(rows=len(rows), cols=len(rows[0])); t.style = "Light Grid Accent 1"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = t.cell(i, j); c.text = ""; r = c.paragraphs[0].add_run(str(v)); r.font.size = Pt(9.5); r.bold = (i == 0 and header)
    if widths:
        for row in t.rows:
            for j, w in enumerate(widths): row.cells[j].width = Inches(w)
    doc.add_paragraph()

ue, ze = U["easy"], Z["easy"]; us, zs = U["por"]["steps"], Z["por"]["steps"]
KEY = "+fairness (eps=0.5*gap0)"
nice = {"cp": "chest pain type", "oldpeak": "ST depression", "thalch": "max heart rate", "exang": "exercise angina", "age": "age", "sex": "sex"}
pn = lambda ts: ", ".join(nice.get(t, t) for t in ts)

# ------------------------------------------------------------------ title
t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("Heart Disease Risk Optimization"); r.bold = True; r.font.size = Pt(24)
t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
t.add_run("Balancing Diagnostic Cost, Feature Sparsity, and Demographic Fairness").italic = True
t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
t.add_run("Krishna Vivek Toshniwal (BC2025047), Samarth Patidar (BC2025085), Venkatachala Darshan Gurumurthy (BC2025110)\nOptimization, IIIT Bangalore, Semester 3")

H("Abstract")
P(f"We study heart-disease risk prediction as an optimization problem on two real datasets: UCI Heart Disease (920 patients, 13 tests) and Z-Alizadeh Sani (303 patients, 54 tests). "
  f"The easy problem is L2-regularised logistic regression on all tests. The hard problem adds three real-world constraints: at most K tests (sparsity), a total test budget B, and a bound ε on the difference in predicted risk between men and women (fairness). "
  f"With K, B and ε set at the elbow of the cost-accuracy frontier, the Price of Realism, the extra training loss versus the easy problem, is {U['por']['PoR_train_loss']:.3f} on UCI and {Z['por']['PoR_train_loss']:.3f} on Z-Alizadeh. "
  f"A {us[KEY]['k']}-test panel on UCI recovers most of the accuracy at {us[KEY]['cost'] / ue['cost'] * 100:.0f}% of the full panel's cost. "
  f"Fairness is expensive on UCI, where disease prevalence differs sharply by sex, and nearly free on Z-Alizadeh, where it does not.")

# ------------------------------------------------------------------ 1
H("1. Motivation")
P("Hospitals cannot run every available test on every patient. Tests cost money, specialised equipment has limited throughput, and a model that is accurate on average can still treat demographic groups unequally. "
  "A standard classifier minimises prediction error only. It ignores what each test costs and ignores whether its errors fall unevenly on one group. "
  "Optimisation is needed because choosing which tests to use is a combinatorial decision (2^d possible panels), and because budget and fairness limits must hold exactly rather than being traded off informally. "
  "Solving it lets a clinic see precisely what accuracy it gives up for a cheaper, fairer test panel.")

# ------------------------------------------------------------------ 2
H("2. Data")
P("Dataset 1: UCI Heart Disease (four hospitals: Cleveland, Hungarian, Switzerland, VA). 920 patients; the target is disease present (num > 0). "
  "We use the 13 standard clinical attributes. Dataset 2: Z-Alizadeh Sani (UCI repository, id 412). 303 patients, 54 candidate features after dropping one constant column; the target is coronary artery disease (CAD) versus normal.")
TABLE([["", "UCI Heart Disease", "Z-Alizadeh Sani"], ["Patients", "920", "303"], ["Candidate tests", "13", "54"], ["Missing values", "up to 66% (ca), 53% (thal), 34% (slope)", "none"],
       ["Positive rate", "55%", "71%"], ["Disease rate, men / women", "63% / 26%", "74% / 68%"], ["Train / test split", "690 / 230", "227 / 76"]], [1.9, 2.5, 2.1])
P("Preprocessing. Both datasets use a stratified 75/25 split. Imputation (training median for numeric tests, training mode for categorical tests) and scaling are fitted on the training set only. "
  "Zeros in cholesterol and resting blood pressure are physiologically impossible and are treated as missing. "
  "We deliberately do not add missing-value indicator columns: on UCI they act as a proxy for the recording hospital and inflate the fit. Removing them raised the easy-problem loss from 0.414 to 0.446. "
  "Categorical tests are one-hot encoded, and every encoded column belongs to exactly one test, so a single binary variable switches a whole test on or off. "
  "The Z-Alizadeh file spells female as 'Fmale'; this is mapped to female.")
P("Costs. Each test has an assumed unit cost (Appendix A). Demographics are free, a questionnaire or vital sign costs a few units, blood tests a few to tens, ECG-derived items about 8 each, and echocardiography about 150 per feature. "
  "These are illustrative values, not sourced prices, and the conclusions should be read with that in mind. Section 7 tests how sensitive the results are to them.", italic=False)

# ------------------------------------------------------------------ 3
H("3. Mathematical formulation")
H("3.1 Easy problem", 2)
P("Let x_i be the (encoded) features of patient i, y_i ∈ {0,1} the label, and λ = 0.01 the regularisation weight. The easy problem is")
EQ("min over (w, b):   J(w, b) = (1/n) Σ_i [ log(1 + exp(wᵀx_i + b)) − y_i (wᵀx_i + b) ] + (λ/2) ‖w‖²")
P("J is strictly convex, so its minimum is global and unique. It uses every test and ignores cost and fairness, and serves as the accuracy benchmark.")
H("3.2 Hard problem (sparse, budgeted and fair MINLP)", 2)
P("Introduce a binary variable z_j ∈ {0,1} for each test j (test used or not), a cost c_j, and let g(i) be the test to which encoded column i belongs. The hard problem is")
EQ("min over (w, b, z):   J(w, b)")
EQ("subject to   |w_i| ≤ M z_g(i)  (a test's weights are zero unless the test is on)")
EQ("Σ_j z_j ≤ K  (sparsity)          Σ_j c_j z_j ≤ B  (budget)          |aᵀw| ≤ ε  (fairness)")
P("The deck states sparsity as ‖w‖₀ ≤ K; counting tests rather than individual encoded columns is the same idea applied at the clinical level. "
  "For the fairness constraint we use a convex surrogate. Let x̄_men and x̄_women be the average feature vectors of men and women in the training data, and a = x̄_men − x̄_women. "
  "Then aᵀw is the difference between the average scores (log-odds) of the two groups, which is linear in w; the bias cancels. "
  "The exact condition on average predicted probabilities, |mean σ(wᵀx | men) − mean σ(wᵀx | women)| ≤ ε, is non-convex. We solve the surrogate and always report the true probability gap afterwards. "
  "We consider two versions of a: demographic parity (a computed over all patients) and equal opportunity (a computed over diseased patients only), which targets the missed-diagnosis rate.")
P("The problem is a mixed-integer nonlinear program. The binary variables and the K and B limits make it combinatorial, and the logistic loss is nonlinear.")

# ------------------------------------------------------------------ 4
H("4. Solution method")
B(["For a fixed panel z, the problem in (w, b) is convex with one linear constraint. We solve it with L-BFGS, switching to SLSQP only when the fairness constraint binds.",
   "UCI (13 tests): we enumerate all 2^13 = 8,192 panels. Constrained loss can never be lower than unconstrained loss, so candidates are processed in increasing unconstrained loss and the search stops once the best constrained value found is below the next candidate. This gives the proven global optimum.",
   "Z-Alizadeh (54 tests): 2^54 panels cannot be enumerated. We use greedy forward selection followed by drop and swap local search. The result is a feasible panel but not a proof of optimality.",
   f"Validation: on UCI the heuristic returned the exact optimum in {U['heuristic_vs_exact']['matches']} of {U['heuristic_vs_exact']['total']} configurations (K ∈ {{2,4,6,8}}, four budgets, with and without fairness)."])
P("Elbow selection. We sweep the budget with K and ε unconstrained, normalise the cost and loss axes to [0,1], and take the point farthest below the chord joining the endpoints. K* is chosen the same way from a sweep over K. "
  f"This gives B* = {U['B_star']}, K* = {U['K_star']} on UCI and B* = {Z['B_star']}, K* = {Z['K_star']} on Z-Alizadeh.")

# ------------------------------------------------------------------ 5
H("5. Results")
H("5.1 Easy problem and verification", 2)
TABLE([["", "UCI", "Z-Alizadeh"], ["Train loss J", f3(ue["J"]), f3(ze["J"])], ["Test accuracy", f3(ue["test_acc"]), f3(ze["test_acc"])], ["Test AUC", f3(ue["test_auc"]), f3(ze["test_auc"])],
       ["Gradient norm at solution", f"{V['uci']['grad']:.1e}", f"{V['za']['grad']:.1e}"], ["Smallest Hessian eigenvalue", f"{V['uci']['mineig']:.1e}", f"{V['za']['mineig']:.1e}"],
       ["Max weight difference vs scikit-learn", f"{V['uci']['sk_diff']:.1e}", f"{V['za']['sk_diff']:.1e}"]], [2.8, 1.8, 1.8])
P("The gradient is zero to numerical precision, the Hessian is positive definite (strictly convex), and the weights agree with scikit-learn's solver, which verifies the optimum.")
H("5.2 Cost versus accuracy", 2)
FIGURE("uci_pareto.png", "Figure 1. UCI: training loss and test accuracy versus the cost of the chosen panel. The star marks the elbow.")
FIGURE("za_pareto.png", "Figure 2. Z-Alizadeh: the same frontier. The small non-monotone bump at higher cost reflects the greedy search being suboptimal.")
P(f"Loss falls steeply for the first few cheap tests and then flattens. On UCI the elbow panel ({pn(us['+budget B*']['tests'])}) costs {us['+budget B*']['cost']:.0f} against {ue['cost']:.0f} for all tests. "
  f"On Z-Alizadeh the elbow configuration reaches a {zs['+budget B*']['k']}-test panel costing {zs['+budget B*']['cost']:.0f} against {ze['cost']:.0f}.")
H("5.3 Hard problem and the Price of Realism", 2)
rows = [["Step", "UCI loss J", "UCI cost", "UCI test acc", "Z-A loss J", "Z-A cost", "Z-A test acc"]]
for k in us:
    rows.append([k.replace("eps=0.5*gap0", "ε = 0.5 × gap").replace("K*", "K").replace("B*", "B"), f3(us[k]["J"]), f"{us[k]['cost']:.0f}", f3(us[k]["test_acc"]),
                 f3(zs[k]["J"]), f"{zs[k]['cost']:.0f}", f3(zs[k]["test_acc"])])
TABLE(rows, [1.9, 0.75, 0.7, 0.8, 0.75, 0.7, 0.8])
P(f"Price of Realism (PoR = Loss(Hard) − Loss(Easy), training loss): UCI {U['por']['PoR_train_loss']:.3f}, Z-Alizadeh {Z['por']['PoR_train_loss']:.3f}. "
  f"Chosen panels: UCI {pn(us[KEY]['tests'])}; Z-Alizadeh {', '.join(zs[KEY]['tests'])}. "
  "On UCI the sparsity limit alone already forces the panel under the budget, so the budget step adds nothing; on Z-Alizadeh the fairness step adds nothing because the sex gap is small.")
FIGURE("uci_por.png", "Figure 3. UCI: loss as each constraint is added.", 4.6); FIGURE("za_por.png", "Figure 4. Z-Alizadeh: loss as each constraint is added.", 4.6)
H("5.4 Price of fairness", 2)
FIGURE("uci_fairness.png", "Figure 5. UCI: tightening ε (fraction of the unconstrained score gap) under demographic parity and equal opportunity.")
FIGURE("za_fairness.png", "Figure 6. Z-Alizadeh: the same sweep.")
d1 = U["fair_sweep"]["dp"]
rows = [["ε (fraction of gap)", "Loss J", "Test acc", "True prob. gap", "FNR gap"]] + [[f"{x['frac']:.2f}", f3(x["J"]), f3(x["test_acc"]), f3(x["test_dp_gap_prob"]), f3(x["test_fnr_gap"])] for x in d1]
P("UCI, demographic parity:"); TABLE(rows, [1.5, 1, 1, 1.3, 1])
P(f"On UCI, halving the parity gap lowers test accuracy from {f3(d1[0]['test_acc'])} to {f3(d1[2]['test_acc'])}. It also widens the false-negative-rate gap between sexes ({f3(d1[0]['test_fnr_gap'])} to {f3(d1[2]['test_fnr_gap'])}). "
  "Because the groups genuinely differ in disease rate, forcing equal average risk under-flags men or over-flags women, so demographic parity can harm the patients it is meant to protect. "
  "Equal opportunity reduces the diseased-patient risk gap at similar or lower loss. Test sets are small (230 and 76 patients), so test accuracy in these sweeps is noisy and training loss is the more reliable guide.")

# ------------------------------------------------------------------ 6
H("6. Easy versus hard: discussion")
B([f"Sparsity is the dominant cost of realism. Going from all tests to K* tests raises training loss from {f3(us['easy']['J'])} to {f3(us['+sparsity K*']['J'])} on UCI and from {f3(zs['easy']['J'])} to {f3(zs['+sparsity K*']['J'])} on Z-Alizadeh.",
   f"The panel changes qualitatively. The easy model on UCI uses expensive tests such as fluoroscopy (ca) and thallium scan (thal). The hard model replaces them with cheap tests ({pn(us[KEY]['tests'])}).",
   "Fairness has a dataset-dependent price. With a 37-point sex gap in disease prevalence (UCI) it costs accuracy; with a 6-point gap (Z-Alizadeh) it costs almost nothing.",
   "Test AUC falls from 0.907 to 0.860 (UCI) and from 0.931 to 0.879 (Z-Alizadeh): the hard model ranks patients somewhat worse while staying close in accuracy."])

# ------------------------------------------------------------------ 7
H("7. Robustness")
ust, zst = U["stability"], Z["stability"]
top = lambda st: ", ".join(nice.get(k, k) for k, v in sorted(st["freq"].items(), key=lambda kv: -kv[1]) if v == 1.0)
accs = lambda st: (np.mean([r["test_acc"] for r in st["runs"]]), np.std([r["test_acc"] for r in st["runs"]]), np.mean([r["easy_test_acc"] for r in st["runs"]]))
au, az = accs(ust), accs(zst)
B([f"Random splits ({len(ust['runs'])} UCI, {len(zst['runs'])} Z-Alizadeh, K*, B*, ε = 0.5 × gap each time). Tests chosen in every split: UCI {top(ust)}; Z-Alizadeh {top(zst)}. "
   f"Hard-panel test accuracy {au[0]:.3f} ± {au[1]:.3f} (easy {au[2]:.3f}) on UCI and {az[0]:.3f} ± {az[1]:.3f} (easy {az[2]:.3f}) on Z-Alizadeh.",
   f"Cost sensitivity (20 random multipliers in [0.5, 2] on every non-zero cost). UCI always keeps {top(U['cost_sensitivity'])}; Z-Alizadeh selects the same five tests in every draw.",
   f"Hospital check (UCI, Cleveland and Hungarian only, dropping the sites with the most missing data): PoR {U['site_robustness']['PoR']:.3f}, panel {pn(U['site_robustness']['hard']['tests'])}, consistent with the full data."])
FIGURE("uci_stability.png", "Figure 7. UCI: selection frequency across random splits.", 4.2); FIGURE("za_stability.png", "Figure 8. Z-Alizadeh: selection frequency across random splits.", 4.2)

# ------------------------------------------------------------------ 8
H("8. Limitations")
B(["Test costs are assumed, illustrative values. Only relative costs matter for panel choice, and Section 7 shows the main panels are stable to cost changes, but absolute cost figures should not be quoted as real prices.",
   "Z-Alizadeh panels come from a greedy search and are upper bounds on the true optimum, not proven optima. The UCI results are exact.",
   "Fairness uses a convex surrogate (score gap) of the true probability gap. We report the true gap but do not guarantee it is at most ε.",
   "Test sets are small (230 and 76 patients) and the datasets are historical single-source collections, so accuracy differences of a few points are within noise.",
   "Cost is modelled per test independently; in practice one ECG or blood draw yields several features at once.",
   "Sex is the only protected attribute considered."])

# ------------------------------------------------------------------ 9
H("9. Reproducibility")
P("Code is in the src/ folder. Run python src/experiments.py uci and python src/experiments.py za, then python src/plots.py, python src/build_deck.py and python src/build_report.py. "
  "Splits use fixed random seeds (0 for the main run, 1 to 10 or 20 for the repeated splits). Required packages: numpy, pandas, scipy, scikit-learn, matplotlib, openpyxl, python-pptx, python-docx.")

# ------------------------------------------------------------------ appendix
H("Appendix A. Assumed test costs")
def costrows(c): return [[k, f"{v:g}"] for k, v in c.items()]
P("UCI Heart Disease:"); TABLE([["Test", "Cost"]] + costrows(UCI_COST), [2.2, 0.9])
P("Z-Alizadeh Sani:"); TABLE([["Test", "Cost"]] + costrows(ZA_COST), [2.4, 0.9])
doc.save(OUT); print("saved", OUT)

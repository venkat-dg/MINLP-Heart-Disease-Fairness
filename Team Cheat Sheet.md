# Team cheat sheet: Heart Disease Risk Optimization

For the three of us to read before presenting or answering questions. Not for submission.

## The project in 30 seconds
We predict heart disease from clinical tests on two datasets (UCI Heart Disease, 920 patients, 13 tests; Z-Alizadeh Sani, 303 patients, 54 tests).

- **Easy problem:** L2 logistic regression using every test. Convex, one global optimum, ignores cost and fairness.
- **Hard problem:** the same loss, but you may use at most **K** tests, spend at most **B** money, and the average risk score for men and women may differ by at most **ε**. Choosing which tests to use is a yes/no decision per test, so it is a mixed-integer nonlinear program (MINLP).
- **Price of Realism (PoR)** = Loss(Hard) − Loss(Easy). It is the accuracy we give up to be cheap, small and fair.

## Numbers to remember

| | UCI | Z-Alizadeh |
|---|---|---|
| Easy loss J | 0.446 | 0.268 |
| Easy test accuracy / AUC | 0.800 / 0.907 | 0.842 / 0.931 |
| Elbow B*, K* | 150, 4 | 80, 5 |
| Hard panel | age, chest pain, max heart rate, ST depression | age, hypertension, pulse rate, typical chest pain, triglycerides |
| Hard panel cost vs full | 135 vs 1055 | 15 vs 663 |
| Hard test accuracy / AUC | 0.770 / 0.860 | 0.816 / 0.879 |
| **PoR** | **0.038** | **0.105** |
| Greedy vs exact search | matched 32 of 32 | not checkable (too big) |

Exact values are in `results/uci_results.json` and `results/za_results.json`.

## Likely questions and honest answers

**Why do we need optimization at all?**
Choosing tests is a combinatorial decision (2^d panels), and budget and fairness limits must hold exactly. Plain logistic regression minimises error only and ignores both.

**What exactly is the "hard" problem and why is it hard?**
Minimise the logistic loss over weights w, bias b and binary on/off variables z, subject to Σz ≤ K, Σcost·z ≤ B, and a fairness bound. The binary variables make it combinatorial, and the logistic loss is nonlinear, hence MINLP.

**Is the easy problem really convex? How do you know the answer is optimal?**
Yes. The loss plus the L2 term is strictly convex. We checked three things: the gradient is about 5e-8, the smallest Hessian eigenvalue is positive (about 8e-3 on UCI), and the weights match scikit-learn's to about 1e-6.

**How did you solve the hard problem if no MINLP solver was used?**
For a fixed set of tests the problem is convex, so we search over sets of tests and solve a convex problem for each. UCI has only 8,192 sets, so we tried them all (exact, with pruning). Z-Alizadeh has 2^54 sets, so we used greedy forward selection with swap and drop moves. We validated greedy against the exact answer on UCI: 32 of 32 matches.

**So is the Z-Alizadeh answer optimal?**
Not proven. It is a feasible panel and an upper bound on the true optimum. The small bump in the Pareto curve (loss at cost 200 slightly above cost 150) is the visible sign of that. Say this openly.

**Why a "surrogate" for fairness? Does it change the project?**
The exact condition (equal average predicted probability for men and women) is non-convex. We constrain the gap in the *scores* before the sigmoid, which is a single linear constraint. It makes each inner problem convex and reliable. After solving, we always report the true probability gap. The score gap and the probability gap move together because the sigmoid is monotone, but we do not guarantee the true gap is at most ε.

**What is the difference between demographic parity and equal opportunity?**
Demographic parity: men and women get the same *average predicted risk*. Equal opportunity: among patients who really have the disease, men and women get the same average risk, i.e. missed diagnoses are spread equally. The deck's constraint is demographic parity; equal opportunity is our added variant.

**Why did fairness cost so much on UCI and so little on Z-Alizadeh?**
Disease rate by sex is 63% vs 26% on UCI (a 37-point gap) and 74% vs 68% on Z-Alizadeh (6 points). Forcing equal average risk when the groups really differ throws away real signal.

**Is demographic parity a good idea on UCI?**
Questionable, and this is a finding. On UCI, tightening demographic parity widened the false-negative gap (0.064 to 0.134 at half the gap), so it can hurt the very patients it should protect. Equal opportunity targets missed diagnoses directly.

**Where do the test costs come from?**
We assumed them, as illustrative values (Appendix A of the report). They are **not** sourced prices. Do not quote absolute costs as real. We tested sensitivity by randomly multiplying every cost by 0.5 to 2: the Z-Alizadeh panel did not change at all, and UCI always kept age, chest pain and ST depression.

**Why did you drop the missing-value indicator columns?**
On UCI, missingness depends on which hospital recorded the patient (Switzerland and VA miss a lot), so the indicators act as a hospital label and inflate accuracy. Removing them raised the easy loss from 0.414 to 0.446. We also checked using only Cleveland and Hungarian: similar result (PoR 0.065).

**How was the elbow chosen?**
Sweep the budget; scale cost and loss to [0,1]; pick the point farthest below the straight line joining the endpoints. K* is chosen the same way from a sweep over K.

**Why is the UCI budget step "free" in the PoR chart?**
With K* = 4, the cheapest good 4-test panel already costs 135, under B* = 150, so the budget constraint does not bind.

**Why does the Z-Alizadeh fairness step show no change?**
The sex gap is small, so the unconstrained panel already satisfies the bound.

**How reliable are the accuracy numbers?**
Test sets are small (230 and 76 patients), so differences of a few points are noise. Use the repeated-split results (10 and 20 random splits) and training loss. Over splits: hard panel UCI 0.761 ± 0.020 vs easy 0.792; Z-Alizadeh 0.818 ± 0.026 vs easy 0.862.

**Why use training loss for PoR rather than test accuracy?**
Loss is what the optimisation minimises, so PoR is a clean comparison of optimal values. Test metrics are shown alongside as a sanity check.

**What would you do with more time?**
Use real costs, model shared costs (one ECG gives many features), run an exact mixed-integer solver on Z-Alizadeh, add more protected attributes (age group), and validate on an external dataset.

## Traps to avoid
- Don't say "globally optimal" for Z-Alizadeh. Say "best panel found, validated method".
- Don't quote costs as real money.
- Don't read too much into 1 to 2 point accuracy differences.
- Don't call the fairness constraint exact; it is a convex surrogate, with the true gap reported.

## Where things are
- Code: `src/` (`data_prep.py` data, `core.py` fit, `hard.py` solvers, `experiments.py` runs, `plots.py` figures)
- Results: `results/*.json`, figures in `results/figures/`
- Deliverables: `Heart Disease Risk Optimization - Final.pptx` (18 slides: problem and data first, then formulation, then solution and analysis), `Final Report - Heart Disease Risk Optimization.docx`

"""Statistical analysis of the observer study (reviewer R4 Maj-3).

R4 Maj-3 raises three things: the 68.5% accuracy is interpreted one-sidedly,
there is no control group, and the 0.79 vs 0.87 AUC gap is reported without a
significance test. This script supplies the numbers for all three.

Data and scoring are identical to make_roc_pr.py, which in turn follows the
original analysis script, so the AUCs here are the ones printed in Figures 8
and 9 and the accuracies are the A column of Table 5.

  1. Accuracy against chance. Each reader saw a balanced 100 real / 100
     synthetic set, so 50% is the chance level and the real images are the
     control condition inside the same session: a reader who cannot tell the
     classes apart necessarily lands at 0.5. Reader-level t-test and exact
     Wilcoxon signed-rank against 0.5, with a 95% CI on the mean.

  2. Both tails. Accuracy is also tested against the observer accuracies
     reported for GAN-generated PRs (78.2% and 82.3%), which is the comparison
     the paper's claim actually rests on.

  3. Discrimination against the control arm. Per reader, the rate of calling a
     synthetic image fake (TPR) against the rate of calling a real image fake
     (FPR), paired across readers, plus each reader's AUC against 0.5 by
     DeLong.

  4. The EC/EP gap. Exact permutation test over all 20 ways of splitting six
     readers 3/3 (reader-level, generalises to new readers), and a DeLong test
     on the two group-consensus ROC curves (case-level, generalises to new
     images). Both are reported because they answer different questions and
     have very different power.
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

QUIZ = sys.argv[1]

SCORE = {"Definitely Fake": 1, "Probably Fake": 0.75, "Unsure": 0.50,
         "Probably Real": 0.25, "Definitely Real": 0}


# --------------------------------------------------------------- fast DeLong
def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def delong(scores, labels):
    """AUCs and their covariance matrix for k score vectors on the same cases.

    scores: (k, n); labels: (n,) with 1 = positive. Sun & Xu (2014).
    """
    scores = np.atleast_2d(np.asarray(scores, dtype=float))
    labels = np.asarray(labels)
    order = np.argsort(-labels, kind="mergesort")     # positives first
    labels = labels[order]
    scores = scores[:, order]
    m = int(labels.sum())
    n = len(labels) - m
    k = scores.shape[0]

    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = _midrank(scores[r, :m])
        ty[r] = _midrank(scores[r, m:])
        tz[r] = _midrank(scores[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    cov = sx / m + sy / n
    return aucs, cov


# -------------------------------------------------------------------- data
sub = pd.read_csv(os.path.join(QUIZ, "submissions.csv"))
lab = pd.read_csv(os.path.join(QUIZ, "image_labels.csv"))
sub.columns = sub.columns.str.strip()
lab.columns = lab.columns.str.strip()

readers = sorted(sub["User Name"].unique())
per = {}
for u in readers:
    d = sub[sub["User Name"] == u].merge(lab, left_on="File",
                                         right_on="Filename", how="inner")
    d = d.sort_values("File").reset_index(drop=True)
    y = (d["Label"] == "Fake").astype(int).values
    s = d["RealFakeValueSlider"].map(SCORE).fillna(0).values

    # Table 5's fractional credit: a "probably fake" on a synthetic image is
    # 0.75 of a TP and 0.25 of a FN.
    tp = np.where(y == 1, s, 0).sum()
    fn = np.where(y == 1, 1 - s, 0).sum()
    fp = np.where(y == 0, s, 0).sum()
    tn = np.where(y == 0, 1 - s, 0).sum()
    per[u] = dict(y=y, s=s, tp=tp, tn=tn, fp=fp, fn=fn,
                  acc=(tp + tn) / (tp + tn + fp + fn),
                  tpr=tp / (tp + fn), fpr=fp / (fp + tn),
                  auc=roc_auc_score(y, s))

y_common = per[readers[0]]["y"]
for u in readers:
    assert (per[u]["y"] == y_common).all(), "case order differs between readers"

acc = np.array([per[u]["acc"] for u in readers])
auc = np.array([per[u]["auc"] for u in readers])
EC = [u for u in readers if u.startswith("EC")]
EP = [u for u in readers if u.startswith("EP")]

print("=" * 74)
print("Per reader (A column of Table 5, AUC of Figure 8)")
print("=" * 74)
for u in readers:
    p = per[u]
    a, cov = delong([p["s"]], p["y"])
    se = float(np.sqrt(cov[0, 0]))
    z = (a[0] - 0.5) / se
    pv = 2 * stats.norm.sf(abs(z))
    print("  %s  acc=%.3f  AUC=%.3f (SE %.3f)  vs 0.5: z=%.2f p=%.2e   "
          "TPR=%.3f FPR=%.3f" % (u, p["acc"], a[0], se, z, pv,
                                 p["tpr"], p["fpr"]))

# ------------------------------------------------- 1. accuracy versus chance
print()
print("=" * 74)
print("1. Mean accuracy against the 50%% chance level")
print("=" * 74)
mean_acc = acc.mean()
t, p_t = stats.ttest_1samp(acc, 0.5)
sem = stats.sem(acc)
ci = stats.t.interval(0.95, len(acc) - 1, loc=mean_acc, scale=sem)
w, p_w = stats.wilcoxon(acc - 0.5, alternative="two-sided",
                        mode="exact" if hasattr(stats, "wilcoxon") else "auto")
print("  mean accuracy      %.4f  (%.1f%%)" % (mean_acc, 100 * mean_acc))
print("  95%% CI             [%.4f, %.4f]  ([%.1f%%, %.1f%%])"
      % (ci[0], ci[1], 100 * ci[0], 100 * ci[1]))
print("  t-test vs 0.5      t(%d)=%.3f  p=%.5f" % (len(acc) - 1, t, p_t))
print("  Wilcoxon vs 0.5    W=%.1f  p=%.5f  (n=6, smallest attainable p=0.03125)"
      % (w, p_w))

# ------------------------------------ 2. accuracy versus the GAN benchmarks
print()
print("=" * 74)
print("2. Mean accuracy against published observer accuracies for GAN PRs")
print("=" * 74)
for name, ref in [("Schoenhof et al. 2024 (78.2%)", 0.782),
                  ("Fukuda et al. 2024 (82.3%)", 0.823)]:
    t2, p2 = stats.ttest_1samp(acc, ref)
    print("  vs %-30s t(%d)=%.3f  p=%.5f" % (name, len(acc) - 1, t2, p2))

# --------------------------- 3. discrimination against the control condition
print()
print("=" * 74)
print("3. Synthetic arm against the real (control) arm, paired across readers")
print("=" * 74)
tpr = np.array([per[u]["tpr"] for u in readers])
fpr = np.array([per[u]["fpr"] for u in readers])
t3, p3 = stats.ttest_rel(tpr, fpr)
w3, pw3 = stats.wilcoxon(tpr - fpr, alternative="two-sided")
print("  mean TPR (synthetic called fake)  %.3f" % tpr.mean())
print("  mean FPR (real called fake)       %.3f" % fpr.mean())
print("  difference                        %.3f" % (tpr - fpr).mean())
print("  paired t-test  t(%d)=%.3f p=%.5f | Wilcoxon W=%.1f p=%.5f"
      % (len(tpr) - 1, t3, p3, w3, pw3))

# ------------------------------------------------------- 4. the EC/EP gap
print()
print("=" * 74)
print("4. Early-career versus experienced readers, mean AUC 0.79 vs 0.87")
print("=" * 74)
a_ec = np.array([per[u]["auc"] for u in EC])
a_ep = np.array([per[u]["auc"] for u in EP])
obs = a_ep.mean() - a_ec.mean()
print("  EC mean AUC %.4f   EP mean AUC %.4f   difference %.4f"
      % (a_ec.mean(), a_ep.mean(), obs))

# exact permutation over all C(6,3) = 20 splits
diffs = []
idx = list(range(6))
for comb in itertools.combinations(idx, 3):
    g1 = auc[list(comb)]
    g2 = auc[[i for i in idx if i not in comb]]
    diffs.append(g2.mean() - g1.mean())
diffs = np.array(diffs)
p_perm = (np.abs(diffs) >= abs(obs) - 1e-12).mean()
print("  exact permutation (20 splits)   p=%.4f   "
      "(smallest attainable p with 3 vs 3 = %.2f)" % (p_perm, 2.0 / 20))
u_stat, p_mw = stats.mannwhitneyu(a_ep, a_ec, alternative="two-sided")
print("  Mann-Whitney U                  U=%.1f p=%.4f" % (u_stat, p_mw))
t4, p4 = stats.ttest_ind(a_ep, a_ec)
sd = np.sqrt(((len(a_ec) - 1) * a_ec.var(ddof=1) +
              (len(a_ep) - 1) * a_ep.var(ddof=1)) / (len(a_ec) + len(a_ep) - 2))
se_d = sd * np.sqrt(1 / len(a_ec) + 1 / len(a_ep))
ci_d = stats.t.interval(0.95, len(a_ec) + len(a_ep) - 2, loc=obs, scale=se_d)
print("  Welch/Student t-test            t(%d)=%.3f p=%.4f" % (4, t4, p4))
print("  95%% CI for the difference       [%.4f, %.4f]" % ci_d)

# case-level: DeLong on the two group-consensus scores
s_ec = np.mean([per[u]["s"] for u in EC], axis=0)
s_ep = np.mean([per[u]["s"] for u in EP], axis=0)
aucs, cov = delong([s_ec, s_ep], y_common)
var_d = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
z = (aucs[1] - aucs[0]) / np.sqrt(var_d)
p_dl = 2 * stats.norm.sf(abs(z))
ci_dl = (aucs[1] - aucs[0] - 1.96 * np.sqrt(var_d),
         aucs[1] - aucs[0] + 1.96 * np.sqrt(var_d))
print("  group-consensus AUC             EC %.4f, EP %.4f" % (aucs[0], aucs[1]))
print("  DeLong (correlated, 200 cases)  z=%.3f p=%.4f  95%% CI [%.4f, %.4f]"
      % (z, p_dl, ci_dl[0], ci_dl[1]))

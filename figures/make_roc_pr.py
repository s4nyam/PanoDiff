"""Redraw the grouped ROC and precision-recall figures (manuscript Figs. 8 and 9)
at a legible size (reviewer R3-2).

Everything that determines what the figure *shows* is taken unchanged from the
original analysis script,

    PanoDiff-QuizData/quiz_fullres_mix/user_wise_results_spreadsheet_backup.py

namely: the score mapping (predicted_label_mapping), the label convention
(Real -> 0, Fake -> 1), sklearn's roc_curve / precision_recall_curve /
average_precision_score, the averaging by interpolation onto 100 common points,
the curve order, line widths, alpha, colours (matplotlib's default cycle), axis
limits, legend positions and the titles.

One line had to be restored rather than copied. The repo copy computes the mean
AP as average_precision_score(np.zeros_like(all_recall), mean_precision), which
passes an all-zero y_true and therefore returns 0.00 (sklearn warns 'No positive
class found in y_true'). That cannot be what produced the submitted figure, which
prints 0.75 and 0.84. Those are the arithmetic means of the three individual APs
in each group -- EC (0.6944, 0.7578, 0.8126) -> 0.7550, EP (0.8469, 0.8927,
0.7693) -> 0.8363 -- so the mean of the per-observer APs is used here and the
published values are reproduced exactly. The averaged *curve* itself is unchanged.

Changes made here, all of them presentational:
  * type sizes;
  * legend labels drop the redundant 'AUC = ' / 'AP = ' prefix so the legend box
    stops covering the curves -- the quantity is named in the caption instead;
  * in the PR figure the chance baseline is now visible. The original drew it with
    axhline(y=0.5) while the y-axis started at exactly 0.5, so the line fell on the
    frame and could not be seen, leaving a legend entry for an invisible line. The
    lower limit is 0.45 here so the baseline sits inside the axes. Its value, 0.5,
    is unchanged and is the correct chance level: the quiz held 100 real and 100
    synthetic images per observer, so the positive-class prevalence is 0.5;
  * that baseline is labelled 'Random Guess', as in the ROC figure, rather than
    'Random'. The original set every element to 20 pt on a
16x8 inch canvas; at the 0.7\\linewidth used in the submission that arrives on
the page at roughly 4.7 pt. Scaling the type by 1.5x and placing the figure at
full \\linewidth puts it at roughly 9-11 pt, matching the pie-chart figure.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             average_precision_score)

QUIZ = sys.argv[1]      # .../PanoDiff-QuizData/quiz_fullres_mix
OUT = sys.argv[2]       # output directory

# ---------------------------------------------------------------- type sizes
# Original values were 20 pt throughout on the same 16x8 canvas.
FS_LABEL = 30           # axis labels
FS_TITLE = 32           # per-panel titles
FS_TICK = 27            # tick labels
FS_LEGEND = 26          # legend entries
FS_SUPTITLE = 34        # figure title

# ------------------------------------------------------------------ the data
# Verbatim from the original script.
predicted_label_mapping = {
    "Definitely Fake": 1,
    "Probably Fake": 0.75,
    "Unsure": 0.50,
    "Probably Real": 0.25,
    "Definitely Real": 0,
}

submissions = pd.read_csv(os.path.join(QUIZ, "submissions.csv"))
image_labels = pd.read_csv(os.path.join(QUIZ, "image_labels.csv"))
submissions.columns = submissions.columns.str.strip()
image_labels.columns = image_labels.columns.str.strip()

all_users_roc_data = {}
all_users_pr_data = {}

for user in submissions["User Name"].unique():
    user_data = submissions[submissions["User Name"] == user]
    merged_data = user_data.merge(image_labels, left_on="File",
                                  right_on="Filename", how="inner")
    final_data = merged_data[["File", "Label", "RealFakeValueSlider"]]
    final_data.columns = ["file_name", "original_label", "predicted_label"]

    final_data.loc[:, "predicted_softmax"] = \
        final_data["predicted_label"].map(predicted_label_mapping)
    final_data["predicted_softmax"] = final_data["predicted_softmax"].fillna(0)

    y_true = final_data["original_label"].map({"Real": 0, "Fake": 1})
    y_scores = final_data["predicted_softmax"]

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    all_users_roc_data[user] = (fpr, tpr, auc(fpr, tpr))

    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
    all_users_pr_data[user] = (precision_curve, recall_curve,
                               average_precision_score(y_true, y_scores))


# ------------------------------------------------------------------- Figure 8
def create_combined_group_roc_plots(all_users_roc_data, base_dir):
    plt.figure(figsize=(16, 8))

    ax1 = plt.subplot(1, 2, 1)
    ec_users = [u for u in all_users_roc_data.keys() if u.startswith("EC")]

    if ec_users:
        all_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(all_fpr)

        for user in sorted(ec_users):
            fpr, tpr, roc_auc = all_users_roc_data[user]
            ax1.plot(fpr, tpr, lw=3, alpha=0.7,
                     label=f'{user} ({roc_auc:.2f})')

            interp_tpr = np.interp(all_fpr, fpr, tpr)
            mean_tpr += interp_tpr

        mean_tpr /= len(ec_users)
        mean_auc = auc(all_fpr, mean_tpr)
        ax1.plot(all_fpr, mean_tpr, 'k-', lw=4, linestyle='--',
                 label=f'Average ({mean_auc:.2f})')

    ax1.plot([0, 1], [0, 1], '--', color='gray', label='Random Guess')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=FS_LABEL)
    ax1.set_ylabel('True Positive Rate', fontsize=FS_LABEL)
    ax1.set_title('Early Career Dentists', fontsize=FS_TITLE)
    ax1.tick_params(axis='both', which='major', labelsize=FS_TICK)
    ax1.legend(loc="lower right", fontsize=FS_LEGEND)
    ax1.grid(False)

    ax2 = plt.subplot(1, 2, 2)
    ep_users = [u for u in all_users_roc_data.keys() if u.startswith("EP")]

    if ep_users:
        all_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(all_fpr)

        for user in sorted(ep_users):
            fpr, tpr, roc_auc = all_users_roc_data[user]
            ax2.plot(fpr, tpr, lw=3, alpha=0.7,
                     label=f'{user} ({roc_auc:.2f})')

            interp_tpr = np.interp(all_fpr, fpr, tpr)
            mean_tpr += interp_tpr

        mean_tpr /= len(ep_users)
        mean_auc = auc(all_fpr, mean_tpr)
        ax2.plot(all_fpr, mean_tpr, 'k-', lw=4, linestyle='--',
                 label=f'Average ({mean_auc:.2f})')

    ax2.plot([0, 1], [0, 1], '--', color='gray', label='Random Guess')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate', fontsize=FS_LABEL)
    ax2.set_ylabel('True Positive Rate', fontsize=FS_LABEL)
    ax2.set_title('Experienced Dentists', fontsize=FS_TITLE)
    ax2.tick_params(axis='both', which='major', labelsize=FS_TICK)
    ax2.legend(loc="lower right", fontsize=FS_LEGEND)
    ax2.grid(False)

    plt.suptitle('ROC Curves by Dentist Experience Level',
                 fontsize=FS_SUPTITLE, y=1.02)
    plt.tight_layout()

    out = os.path.join(base_dir, "roc_curves.pdf")
    plt.savefig(out, format='pdf', bbox_inches='tight')
    plt.savefig(out.replace(".pdf", ".png"), dpi=110, bbox_inches='tight')
    plt.close()
    print("wrote", out)


# ------------------------------------------------------------------- Figure 9
def create_combined_group_pr_plots(all_users_pr_data, base_dir):
    plt.figure(figsize=(16, 8))

    ax1 = plt.subplot(1, 2, 1)
    ec_users = [u for u in all_users_pr_data.keys() if u.startswith("EC")]

    if ec_users:
        all_recall = np.linspace(0, 1, 100)
        mean_precision = np.zeros_like(all_recall)

        for user in sorted(ec_users):
            precision, recall, avg_precision = all_users_pr_data[user]
            ax1.plot(recall, precision, lw=3, alpha=0.7,
                     label=f'{user} ({avg_precision:.2f})')

            interp_precision = np.interp(all_recall, recall[::-1], precision[::-1])
            mean_precision += interp_precision

        mean_precision /= len(ec_users)
        mean_ap = np.mean([all_users_pr_data[u][2] for u in ec_users])

        ax1.plot(all_recall, mean_precision, 'k-', lw=4, linestyle='--',
                 label=f'Average ({mean_ap:.2f})')

    ax1.axhline(y=0.5, color='gray', linestyle='--', label='Random Guess')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.45, 1.05])
    ax1.set_xlabel('Recall', fontsize=FS_LABEL)
    ax1.set_ylabel('Precision', fontsize=FS_LABEL)
    ax1.set_title('Early Career Dentists', fontsize=FS_TITLE)
    ax1.tick_params(axis='both', which='major', labelsize=FS_TICK)
    ax1.legend(loc="lower left", fontsize=FS_LEGEND)
    ax1.grid(False)

    ax2 = plt.subplot(1, 2, 2)
    ep_users = [u for u in all_users_pr_data.keys() if u.startswith("EP")]

    if ep_users:
        all_recall = np.linspace(0, 1, 100)
        mean_precision = np.zeros_like(all_recall)

        for user in sorted(ep_users):
            precision, recall, avg_precision = all_users_pr_data[user]
            ax2.plot(recall, precision, lw=3, alpha=0.7,
                     label=f'{user} ({avg_precision:.2f})')

            interp_precision = np.interp(all_recall, recall[::-1], precision[::-1])
            mean_precision += interp_precision

        mean_precision /= len(ep_users)
        mean_ap = np.mean([all_users_pr_data[u][2] for u in ep_users])

        ax2.plot(all_recall, mean_precision, 'k-', lw=4, linestyle='--',
                 label=f'Average ({mean_ap:.2f})')

    ax2.axhline(y=0.5, color='gray', linestyle='--', label='Random Guess')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.45, 1.05])
    ax2.set_xlabel('Recall', fontsize=FS_LABEL)
    ax2.set_ylabel('Precision', fontsize=FS_LABEL)
    ax2.set_title('Experienced Dentists', fontsize=FS_TITLE)
    ax2.tick_params(axis='both', which='major', labelsize=FS_TICK)
    ax2.legend(loc="lower left", fontsize=FS_LEGEND)
    ax2.grid(False)

    plt.suptitle('Precision-Recall Curves by Dentist Experience Level',
                 fontsize=FS_SUPTITLE, y=1.02)
    plt.tight_layout()

    out = os.path.join(base_dir, "pr_curves.pdf")
    plt.savefig(out, format='pdf', bbox_inches='tight')
    plt.savefig(out.replace(".pdf", ".png"), dpi=110, bbox_inches='tight')
    plt.close()
    print("wrote", out)


os.makedirs(OUT, exist_ok=True)
create_combined_group_roc_plots(all_users_roc_data, OUT)
create_combined_group_pr_plots(all_users_pr_data, OUT)

# Report the numbers so they can be checked against the submitted figure.
print("\nAUC per observer (submitted: EC1 .74 EC2 .79 EC3 .85 | EP1 .89 EP2 .91 EP3 .80)")
for u in sorted(all_users_roc_data):
    print("   %s  AUC=%.2f" % (u, all_users_roc_data[u][2]))
print("AP per observer (submitted: EC1 .69 EC2 .76 EC3 .81 | EP1 .85 EP2 .89 EP3 .77)")
for u in sorted(all_users_pr_data):
    print("   %s  AP=%.2f" % (u, all_users_pr_data[u][2]))

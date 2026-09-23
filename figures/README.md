# Figures

Scripts that draw the paper's figures from the outputs of `evaluation/` and `baselines/`
(paths resolve against `PANODIFF_WORK`, see `docs/REPRODUCE.md`).

| Script | Figure |
|---|---|
| `make_roc_pr.py` | ROC and precision-recall curves per observer (Figures 10-11) |
| `make_piecharts.py` | per-observer decision pie charts (Figure 9) |
| `make_attention_process.py` | how the attention statistics are computed (Figure 14) |
| `make_tsne_arms.py` | t-SNE of every generative arm (Figure 16) |
| `make_pr_scores.py` | per-image fidelity and coverage behind Table 9 (Figure 17) |
| `make_pr_scores_t11.py` | the same for every model of Table 11 (not in the paper) |
| `select_best_examples.py`, `make_mosaics.py` | LR / HAT-SR / SwinIR mosaic (Figure 18) and direct-HR mosaic (Figure 19) |

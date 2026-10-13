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
| `make_pr_scores_t11.py` | the same for every model of the direct high-resolution table, now Table 10 (not in the paper; `t11` in the file name is the earlier table number) |
| `select_random_examples.py`, `make_mosaics.py` | LR / HAT-SR / SwinIR mosaic (Figure 18) and direct-HR mosaic (Figure 19). The rows of Figure 19 are not selected: `select_random_examples.py` takes, for every row, the first three images of the random 1500-image sample of `evaluation/anatomy_memorisation/anatomy.py` (`--check` recomputes their measures to confirm the order) and writes `random_examples.json`, which `make_mosaics.py` reads |

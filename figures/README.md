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
| `make_mosaics.py`, `directhr_examples.json` | LR / HAT-SR / SwinIR mosaic (Figure 18) and direct-HR mosaic (Figure 19). The three examples per row of Figure 19 were chosen by hand; `directhr_examples.json` lists their file names per image set, relative to `$PANODIFF_WORK/experiments/direct-hr-baselines`. `make_mosaics.py --directhr-only` draws Figure 19 alone and `--out DIR` writes it to another folder |

# Evaluation

One folder per question the paper asks. These are the scripts that produced the published numbers,
kept as they ran; paths resolve against `PANODIFF_WORK` (see [`docs/REPRODUCE.md`](../docs/REPRODUCE.md)).

| Folder | What it measures | Paper |
|---|---|---|
| `distribution/` | FID, KID, IS, precision/recall (k = 3) from one Inception pass per image set; per-image fidelity/coverage scores; recall robustness under resampling; high-frequency texture energy | Tables 3, 4, 8, 9, 11; Figures 16, 17 |
| `sr_restoration/` | HAT vs SwinIR, released and fine-tuned, restoring 185 held-out DENTEX radiographs (PSNR, SSIM, LPIPS); SR ablation | Table 10, Section 3.4 |
| `vit/` | ViT real-vs-synthetic classifier, attention rollout and attention statistics; duplicate-aware split; per-arm detectability | Section 4.3, Table 11 |
| `anatomy_memorisation/` | Bilateral symmetry, tooth-row periodicity, enamel crowns, occlusal separation; nearest-neighbour memorisation test | Table 12, Section 4.5.2 |
| `device/` | Per-device FID at matched n = 500 with the corpus crop | Section 4.5.3 |
| `medsam/` | Frozen MedSAM under identical prompts on real and synthetic PRs | Section 4.4, Table 7, Figure 15 |
| `observer_study/` | Reader accuracies against chance and against published GAN values (t-test, Wilcoxon), DeLong AUC tests, EC/EP permutation test; run as `python observer_study/observer_stats.py ../observer_study` | Section 4.1 |
| `efficiency/` | Parameters, time per step, inference time, training cost in accelerator-hours | Table 2, Table 11 cost column |
| `ablation/` | Diffusion ablation (EMA, DDIM steps) | Section 3.4 |

## Typical order for a new set of samples

```bash
python distribution/extract_feats.py        # one Inception pass per set -> feats/<set>.npz
python distribution/metrics_from_feats.py   # FID, KID, IS, precision, recall
python distribution/pr_ratios.py            # per-image fidelity / coverage scores (Figure 17)
python vit/vit_detect.py                    # ViT detectability, one arm per GPU
python anatomy_memorisation/anatomy.py && python anatomy_memorisation/make_anat_table.py
python anatomy_memorisation/memorisation.py
```

Scripts that train a model or run a large pass read `SLURM_PROCID` to split work across GPUs and
run unchanged on a single GPU.

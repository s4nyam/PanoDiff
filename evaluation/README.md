# Evaluation

One folder per question the paper asks. These are the scripts that produced the published numbers,
kept as they ran; paths resolve against `PANODIFF_WORK` (see [`docs/REPRODUCE.md`](../docs/REPRODUCE.md)).

| Folder | What it measures | Paper |
|---|---|---|
| `distribution/` | FID, KID, IS, precision/recall (k = 3) from one Inception pass per image set; per-image fidelity/coverage scores; precision and recall on ten random subsets of 5000 generated images (`pr_subsample.py`); high-frequency texture energy | Tables 3, 4, 8, 9, 10; Figures 16, 17; Sections 4.5.1, 4.5.2 |
| `sr_restoration/` | HAT vs SwinIR, released and fine-tuned, restoring 185 held-out DENTEX radiographs (PSNR, SSIM, LPIPS); SR ablation. The restoration table is no longer in the paper; the paper quotes the LPIPS of HAT with released and with fine-tuned weights | Section 3.4 |
| `vit/` | ViT real-vs-synthetic classifier, attention rollout and attention statistics; duplicate-aware split; per-arm detectability | Section 4.3, Table 10 |
| `anatomy_memorisation/` | Three anatomical measures (bilateral symmetry, tooth-row periodicity, occlusal separation) as mean and standard deviation, and the share of images outside the central 95% range of the real images; nearest-neighbour memorisation test; share of images with a block of horizontal streaks at the image border (`border_streaks.py`). `anatomy.py` also stores a crown count, which the paper does not report | Table 11, Section 4.5.2, caption of Figure 19 |
| `device/` | Per-device FID at matched n = 500 with the corpus crop | Section 4.5.3 |
| `medsam/` | Frozen MedSAM under two identical box prompts (fixed box, dentition box) on real and synthetic PRs; Figure 15 picks in `medsam_figure_picks.json` | Section 4.4, Table 7, Figure 15 |
| `observer_study/` | Reader accuracies against chance and against published GAN values (t-test, Wilcoxon), DeLong AUC tests, EC/EP permutation test; run as `python observer_study/observer_stats.py ../observer_study` | Section 4.1 |
| `efficiency/` | Parameters, time per step, inference time, training cost in GPU-hours | Table 2, Table 10 cost column |
| `ablation/` | Diffusion ablation (EMA, DDIM steps) | Section 3.4 |

`build_anatomical_prior.py` (top level of this folder) builds the fixed tooth and jaw regions from the
TUFTS expert masks that `vit/vit_attention.py`, `vit/vit_detect.py`, `anatomy_memorisation/anatomy.py`
and `figures/make_attention_process.py` load; run it once first.

File names that contain `t11` (`distribution/extract_feats_t11.py`, `distribution/pr_ratios_t11.py`)
belong to the direct high-resolution table. It is Table 10 of the paper; it was Table 11 in an
earlier version, and the file names were kept.

## Typical order for a new set of samples

```bash
python build_anatomical_prior.py            # once: tooth and jaw regions (Section 4.3, Table 11)
python distribution/extract_feats.py        # one Inception pass per set -> feats/<set>.npz
python distribution/metrics_from_feats.py   # FID, KID, IS, precision, recall
python distribution/pr_ratios.py            # per-image fidelity / coverage scores (Figure 17)
python vit/vit_detect.py                    # ViT detectability, one arm per GPU
python anatomy_memorisation/anatomy.py      # anatomical measures per image -> out/anat_<set>.json
(cd $PANODIFF_WORK/experiments/direct-hr-baselines/analysis_r25_r31 && python $OLDPWD/anatomy_memorisation/make_anat_table.py)   # Table 11; reads out/anat_*.json there
python anatomy_memorisation/memorisation.py
python anatomy_memorisation/border_streaks.py   # streak block at the image border, all images of each set -> out/border_streaks.json
```

Scripts that train a model or run a large pass read `SLURM_PROCID` to split work across GPUs and
run unchanged on a single GPU.

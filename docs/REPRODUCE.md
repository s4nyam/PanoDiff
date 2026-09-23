# Reproducing the paper

## Work directory

The evaluation, baseline and figure scripts are the code that produced the published numbers.
They read and write inside one work directory, given by the environment variable

```bash
export PANODIFF_WORK=/path/to/work        # default: ./work
```

and expect the layout the experiments used (only the parts a script touches need to exist):

```
$PANODIFF_WORK/
├── project-files/PanoDiff/
│   ├── GT-PanoDiff-Training/Original_Datasets/    the five source datasets as downloaded
│   ├── Syn-Calc-FID&IS/                           the image sets of Tables 8-9, one folder each:
│   │     1-TrainingDataInLR  2-DiffLR  3-GANsLR  4-DiffHATSR  5-GANsHATSR
│   │     8-TrainHATSR  10-TrainDataInHR           (10 = the 7243 real PRs at 1024 x 512)
│   └── PanoDiff-QuizData/quiz_fullres_mix/        observer-study images, labels, responses
├── new-baselines-generation/
│   ├── data/train_hr.txt                          list of the 7243 HR files (baseline training)
│   ├── runs/<arm>/latest.pth                      baseline checkpoints
│   ├── samples/<arm>/                             7243 samples per baseline (baselines/sample.py)
│   ├── refs/{real,PanoDiffSR_existing,FastGAN_HATSR_existing}   links to sets 10, 4 and 5
│   └── analysis_r25_r31/out/                      ViT, anatomy and memorisation results
├── swinir-ft-eval/out/                            Inception features and metrics (Tables 9, 11)
├── sr-restore-7243/                               held-out restoration test (Table 10)
├── medsam-probe/                                  MedSAM probe data and results (Table 7)
├── temp-codes/                                    ViT attention (Section 4.3) and device FIDs (4.5.3)
└── new-files/overleaf/figures/                    figure outputs
```

## Paper result → script

| Paper | Script(s) |
|---|---|
| Table 1, corpus | `data/prepare_corpus.py` |
| Table 2, efficiency | `evaluation/efficiency/bench_efficiency.py` |
| Section 3.4, ablations | `evaluation/ablation/diff_ablation.py`, `evaluation/sr_restoration/sr_ablation.py` |
| Tables 3, 4, 8, 9 (FID, KID, IS, precision, recall) | `evaluation/distribution/extract_feats.py` → `metrics_from_feats.py` |
| Figure 16 (t-SNE of every arm) | `figures/make_tsne_arms.py` |
| Figure 17 (per-image precision/recall) | `evaluation/distribution/pr_ratios.py` → `figures/make_pr_scores.py` |
| Figure 18 (LR → HAT-SR / SwinIR mosaic), Figure 19 (direct-HR mosaic) | `figures/make_mosaics.py`, `figures/select_best_examples.py` |
| Table 10 (held-out restoration) | `evaluation/sr_restoration/prep_heldout_crop.py` → `restore_heldout.py` → `merge.py` |
| Texture energy (Section 4.5.1) | `evaluation/distribution/texture_energy.py` |
| Section 4.1 (observer statistics) | `evaluation/observer_study/observer_stats.py` |
| Section 4.3 (ViT, attention statistics) | `evaluation/vit/vit_attention.py`; duplicate-aware split `evaluation/vit/leakage/` |
| Section 4.4, Table 7, Figure 15 (MedSAM) | `evaluation/medsam/prepare_data_crop.py` → `medsam_probe.py` → `analyze_probe.py`, `make_medsam_figure.py` |
| Table 11: training | `baselines/*/train*.py` (commands in `baselines/README.md`) |
| Table 11: samples | `baselines/sample.py` |
| Table 11: FID, IS | `evaluation/distribution/eval_fid_is.py` |
| Table 11: precision, recall | `evaluation/distribution/extract_feats_t11.py` → `pr_ratios_t11.py`; robustness `recall_subsample.py` |
| Table 11: ViT detectability | `evaluation/vit/vit_detect.py` |
| Table 11: cost of rows 1-2 | `evaluation/efficiency/bench_train_cost.py`, `bench_fastgan_cost.py` |
| Table 12, memorisation | `evaluation/anatomy_memorisation/anatomy.py` → `make_anat_table.py`; `memorisation.py` |
| Section 4.5.3 (device FIDs) | `evaluation/device/metrics_stage1.py --crop 64 127 90 127 --resize 1024 512 --save-features` → `analyse_devices.py` |

## Conventions

- Images are width × height (1024 × 512); tensors are height × width (512 × 1024).
- FID, KID and IS use torchmetrics' Inception network with `normalize=True`, on the full 7243-image
  sets unless stated; KID is reported × 10³.
- Precision and recall are those of Kynkäänniemi et al. (2019) with k = 3 on the same features.

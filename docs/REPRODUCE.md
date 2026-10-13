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
├── data/original-corpus/PanoDiff/
│   ├── GT-PanoDiff-Training/Original_Datasets/    the five source datasets as downloaded
│   ├── Syn-Calc-FID&IS/                           the image sets of Tables 8-9, one folder each:
│   │     1-TrainingDataInLR  2-DiffLR  3-GANsLR  4-DiffHATSR  5-GANsHATSR
│   │     6-DiffSwinIR  7-GANsSwinIR  9-TrainSwinIR  (SwinIR with the released weights)
│   │     8-TrainHATSR  10-TrainDataInHR           (10 = the 7243 real PRs at 1024 x 512)
│   ├── PanoDiff-QuizData/quiz_fullres_mix/        observer-study images, labels, responses
│   └── Project/                                   original training code and weights: sd/ (PanoDiff),
│                                                  sr/training/ (HAT code and released HAT weights),
│                                                  sr/experiments/ (fine-tuned HAT-SR checkpoints)
├── experiments/
│   ├── direct-hr-baselines/
│   │   ├── data/train_hr.txt                      list of the 7243 HR files (baseline training)
│   │   ├── runs/<arm>/latest.pth                  baseline checkpoints
│   │   ├── samples/<arm>/                         7243 samples per baseline (baselines/sample.py)
│   │   ├── refs/{real,PanoDiffSR_existing,FastGAN_HATSR_existing}   links to sets 10, 4 and 5
│   │   ├── repos/{stylegan3,FastGAN-pytorch,guided-diffusion,medfusion}   clones of the reference implementations
│   │   └── analysis_r25_r31/out/                  ViT, anatomy and memorisation results
│   ├── sr-distribution-metrics/out/               Inception features, metrics and the fine-tuned SwinIR image sets
│   │                                              6ft-DiffSwinIRft, 7ft-GANsSwinIRft, 9ft-TrainSwinIRft (Tables 8, 9, 10)
│   ├── sr-restoration/{heldout,heldout_crop,out}/ held-out restoration test (Section 3.4); heldout/ holds dl/ (DENTEX
│   │                                              validation and test images as downloaded), dup_check.json and
│   │                                              heldout_list.json
│   ├── medsam-probe/                              MedSAM probe data and results (Table 7)
│   ├── downstream-segmentation/data/              tooth-segmentation data (evaluation/medsam/prepare_data.py), read by the MedSAM probe
│   ├── vit-attention/                             ViT attention (Section 4.3); priors/ holds the tooth and jaw regions
│   ├── vit-duplicate-leakage/out/                 duplicate-aware split of the ViT (Section 4.3)
│   ├── fid-is-recomputation/                      stats/: Inception features of sets 4, 5 and 10 (evaluation/device/metrics_stage1.py)
│   └── device-stratified-fid/                     device FIDs (Section 4.5.3): features per device and results
├── manuscript/
│   ├── overleaf/figures/                          figure outputs
│   └── analysis/, figure-sources/                 intermediate files of the figure scripts; analysis/swinir_r4maj2/network_swinir_stock.py
│                                                  is the SwinIR network loaded by super_resolution/swinir/infer_swinir_ft.py
├── archive/recovered-workstation/swinir/          SwinIR weights, released and fine-tuned
├── weights/medsam_vit_b_official.pth              MedSAM checkpoint
└── quiz/dataset/                                  source radiographs of the observer-study image set
                                                   (observer_study/app/create_quiz_data/run1.py)
```

These folder names are new in this version of the repository. Earlier versions used other
names for the same folders.

## Paper result → script

| Paper | Script(s) |
|---|---|
| Table 1, corpus | `data/prepare_corpus.py` |
| Table 2, efficiency | `evaluation/efficiency/bench_efficiency.py` |
| Section 3.4, diffusion ablation (weight averaging, DDIM steps) | `evaluation/ablation/diff_ablation.py` |
| Section 3.4, SR fine-tuning on 185 held-out PRs (LPIPS, released and fine-tuned HAT weights) | `evaluation/sr_restoration/prep_heldout.py` → `prep_heldout_crop.py` → `restore_heldout.py` → `merge.py heldout`, with `HELD_DIR=$PANODIFF_WORK/experiments/sr-restoration/heldout_crop` (rows `hat_pre` and `hat_ft`); loader and metric helpers in `sr_ablation.py`. `prep_heldout.py` needs `heldout/dl/` (the DENTEX validation and test images as downloaded) and `heldout/dup_check.json` (the highest thumbnail similarity of each of these images to a DENTEX training image); no script in this repository writes `dup_check.json` |
| Tables 3, 4, 8, 9 (FID, KID, IS, precision, recall) | `evaluation/distribution/extract_feats.py` → `metrics_from_feats.py` |
| Figure 16 (t-SNE of every arm) | `figures/make_tsne_arms.py` |
| Figure 17 (per-image precision/recall) | `evaluation/distribution/pr_ratios.py` → `figures/make_pr_scores.py` |
| Figure 18 (LR → HAT-SR / SwinIR mosaic), Figure 19 (direct-HR mosaic, three hand-picked examples per row) | `figures/make_mosaics.py`; the examples of Figure 19 are listed in `figures/directhr_examples.json` |
| Caption of Figure 19: share of images with a block of horizontal streaks at the border (real, PanoDiff-SR and LDM) | `evaluation/anatomy_memorisation/border_streaks.py` |
| Texture energy (Section 4.5.1) | `evaluation/distribution/texture_energy.py` |
| Section 4.1 (observer statistics) | `evaluation/observer_study/observer_stats.py` |
| Section 4.3 (ViT, attention statistics) | `evaluation/build_anatomical_prior.py` → `evaluation/vit/vit_attention.py`; duplicate-aware split `evaluation/vit/leakage/` |
| Section 4.4, Table 7, Figure 15 (MedSAM) | `evaluation/medsam/prepare_data_crop.py` → `medsam_probe.py` → `analyze_probe.py`, `make_medsam_figure.py` |
| Table 10: training | `baselines/*/train*.py` (commands in `baselines/README.md`) |
| Table 10: samples | `baselines/sample.py` |
| Table 10: FID, IS | `evaluation/distribution/eval_fid_is.py` |
| Table 10: precision, recall | `evaluation/distribution/extract_feats_t11.py` → `pr_ratios_t11.py`; the McNemar tests of Section 4.5.2 are computed from the per-image coverage scores in `pr_ratios_t11.npz` (a real image is covered when its score is at most 1); no script in this repository runs them |
| Section 4.5.2: precision and recall on ten random subsets of 5000 generated images | `evaluation/distribution/pr_subsample.py`; the Welch t-tests of the text are computed from the `runs` lists in `pr_subsample.json` (`scipy.stats.ttest_ind(a, b, equal_var=False)`) |
| Table 10: ViT detectability | `evaluation/vit/vit_detect.py` |
| Table 10: cost of rows 1-2 | `evaluation/efficiency/bench_train_cost.py`, `bench_fastgan_cost.py` |
| Table 11 (three anatomical measures, outlier rate) | `evaluation/build_anatomical_prior.py` → `evaluation/anatomy_memorisation/anatomy.py` → `make_anat_table.py`; rate per measure `outlier.py` |
| Section 4.5.2: nearest-neighbour memorisation check | `evaluation/anatomy_memorisation/memorisation.py` |
| Section 4.5.3 (device FIDs) | `evaluation/device/metrics_stage1.py --crop 64 127 90 127 --resize 1024 512 --save-features` → `analyse_devices.py` |

The restoration table (PSNR, SSIM and LPIPS of HAT and SwinIR on the 185 held-out PRs) is no longer in
the paper. `evaluation/sr_restoration/` still produces all its rows; the paper quotes the LPIPS of HAT
with the released and with the fine-tuned weights in Section 3.4.

File names that contain `t11` belong to the direct high-resolution table. It is Table 10 of the
paper; it was Table 11 in an earlier version, and the file names were kept.

## Conventions

- Images are width × height (1024 × 512); tensors are height × width (512 × 1024).
- FID, KID and IS use torchmetrics' Inception network with `normalize=True`, on the full 7243-image
  sets unless stated; KID is reported × 10³.
- Precision and recall are those of Kynkäänniemi et al. (2019) with k = 3 on the same features.

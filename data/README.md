# Data

The corpus pools five public panoramic-radiograph datasets (paper Table 1). They are not
redistributed here; download them from their sources.

| Dataset | Files used | Source |
|---|---|---|
| ADLD (A dual-labeled dataset) | 500 | [Kaggle](https://www.kaggle.com/datasets/zwbzwb12341234/a-dual-labeled-dataset) |
| DENTEX (training images only) | 3603 | [Zenodo](https://zenodo.org/records/7812323) |
| TSXK (Teeth Segmentation on dental X-ray images) | 598, used twice (1196 files) | [Kaggle](https://www.kaggle.com/datasets/humansintheloop/teeth-segmentation-on-dental-x-ray-images) |
| TUFTS (Tufts Dental Database) | 1000 | [on request](https://tdd.ece.tufts.edu/) |
| USPFORP (São Paulo dataset) | 945 | on request |

The 7243 files contain 5653 distinct radiographs (DENTEX has 2686 unique images among its 3603
files, USPFORP repeats 72 of its 945, and TSXK entered twice). Every real-set quantity in the paper
was computed on the 7243 files.

## Preparation

```bash
python data/prepare_corpus.py --src <folder with the originals> --out work/corpus
```

Every radiograph is cropped by a fixed margin of 64, 127, 90 and 127 pixels (top, left, bottom,
right), resized to 1024 × 512 with LANCZOS (`HR/`), and downsampled fourfold by area averaging to
256 × 128 (`LR/`). The same preparation is applied to the held-out DENTEX radiographs of the
restoration test and to the per-device sets of Section 4.5.3.

Images in the paper are given as width × height (1024 × 512); tensor shapes in the code are
height × width (512 × 1024).

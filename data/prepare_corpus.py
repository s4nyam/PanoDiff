"""Build the PanoDiff-SR training corpus from the original radiographs.

Every source radiograph is prepared identically (paper Section 3.1):
  1. a fixed margin of 64, 127, 90 and 127 pixels (top, left, bottom, right) is cropped away,
     removing burnt-in overlays and the empty image border (capped at a quarter of the side for
     small images);
  2. the result is resized to 1024 x 512 (width x height) with PIL LANCZOS          -> <out>/HR
  3. a low-resolution copy is made by 4x area averaging to 256 x 128                -> <out>/LR

HR is the super-resolution target and the real reference for FID/IS; LR is what PanoDiff is trained
on and the input of the fixed-degradation SR pairs.

    python data/prepare_corpus.py --src /path/to/originals --out work/corpus
"""
import argparse, os
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

CROP = (64, 127, 90, 127)              # top, left, bottom, right, in source pixels
HR_SIZE, LR_SIZE = (1024, 512), (256, 128)
EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def prepare(img):
    img = img.convert("RGB")
    w, h = img.size
    t, l, b, r = CROP
    t, b = min(t, h // 4), min(b, h // 4)
    l, r = min(l, w // 4), min(r, w // 4)
    hr = img.crop((l, t, w - r, h - b)).resize(HR_SIZE, Image.LANCZOS)
    lr = hr.resize(LR_SIZE, Image.BOX)                 # box filter = area averaging for 4x
    return hr, lr


def work(job):
    src, name, out = job
    try:
        with Image.open(src) as im:
            hr, lr = prepare(im)
        hr.save(os.path.join(out, "HR", name))
        lr.save(os.path.join(out, "LR", name))
        return None
    except Exception as e:                            # noqa: BLE001
        return f"{src}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder (searched recursively) with the original radiographs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "HR"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "LR"), exist_ok=True)
    jobs, k = [], 0
    for root, _, files in os.walk(a.src):
        for f in sorted(files):
            if f.lower().endswith(EXT):
                jobs.append((os.path.join(root, f), f"train_{k:08d}.png", a.out)); k += 1
    with ProcessPoolExecutor(a.workers) as ex:
        errors = [e for e in ex.map(work, jobs, chunksize=16) if e]
    print(f"{len(jobs) - len(errors)} radiographs written to {a.out}/HR and {a.out}/LR")
    for e in errors:
        print("skipped", e)


if __name__ == "__main__":
    main()

"""High-frequency energy of each HR set: mean squared 3x3 Laplacian of the grey image (0-255).

Over every image of each set by default; TEXTURE_N=300 restricts it to a random subset of that
size, which is what the first version of this measurement used. Tests the section's claim that
SwinIR adds a grainy high-frequency texture; the real HR radiographs (10) are the reference."""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import os, json, numpy as np
from PIL import Image
from scipy.ndimage import laplace
from multiprocessing import Pool
P = WORK
LEG = f"{P}/project-files/PanoDiff/Syn-Calc-FID&IS"; NEW = f"{P}/swinir-ft-eval/out"
SETS = {"10 real HR": f"{LEG}/10-TrainDataInHR", "4 PanoDiff+HAT": f"{LEG}/4-DiffHATSR",
        "6 PanoDiff+SwinIR (released)": f"{LEG}/6-DiffSwinIR", "6ft PanoDiff+SwinIR (PR-ft)": f"{NEW}/6ft-DiffSwinIRft",
        "5 FastGAN+HAT": f"{LEG}/5-GANsHATSR", "7 FastGAN+SwinIR (released)": f"{LEG}/7-GANsSwinIR",
        "7ft FastGAN+SwinIR (PR-ft)": f"{NEW}/7ft-GANsSwinIRft", "8 realLR+HAT": f"{LEG}/8-TrainHATSR",
        "9 realLR+SwinIR (released)": f"{LEG}/9-TrainSwinIR", "9ft realLR+SwinIR (PR-ft)": f"{NEW}/9ft-TrainSwinIRft"}
def energy(path):
    a = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    return float((laplace(a) ** 2).mean())
if __name__ == "__main__":
    rng = np.random.default_rng(0); out = {}
    with Pool(32) as pool:
        for name, d in SETS.items():
            fs = sorted(f for f in os.listdir(d) if f.endswith(".png"))[:7243]
            n = int(os.environ.get("TEXTURE_N", 0))
            idx = np.random.default_rng(0).choice(len(fs), n, replace=False) if 0 < n < len(fs) else range(len(fs))
            pick = [os.path.join(d, fs[i]) for i in idx]
            e = pool.map(energy, pick); out[name] = (float(np.mean(e)), float(np.median(e)), len(e))
            print(f"{name:32s} n {out[name][2]:5d}  mean {out[name][0]:8.1f}  median {out[name][1]:8.1f}", flush=True)
    json.dump(out, open(f"{NEW}/texture_energy_{len(pick)}.json", "w"), indent=1)

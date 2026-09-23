"""Step A: the existing classifier (temp-codes/vit_realfake.pt, 97.41% on its held-out set),
re-scored with the held-out real radiographs that also sit in its training half removed."""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import json
import numpy as np
import torch
from common import OUT, ViT, load, random_split, leaked, predict

real, syn, groups, g = load()
n = len(groups)
tr_r, va_r, tr_s, va_s = random_split(n, 0)
leak = leaked(groups, tr_r, va_r)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViT().to(dev).eval()
model.load_state_dict(torch.load(WORK + "/temp-codes/vit_realfake.pt", map_location=dev))
pr = predict(model, real, va_r, dev); ps = predict(model, syn, va_s, dev)
ok_r, ok_s = pr == 0, ps == 1
res = {
 "held_out_per_class": len(va_r),
 "all": float(np.concatenate([ok_r, ok_s]).mean()),
 "real_acc_all": float(ok_r.mean()), "syn_acc": float(ok_s.mean()),
 "n_real_leaked": int(leak.sum()),
 "real_acc_leaked": float(ok_r[leak].mean()) if leak.any() else None,
 "real_acc_clean": float(ok_r[~leak].mean()),
 # drop each leaked real image and the synthetic image at the same index, as the split pairs them
 "clean_paired": float(np.concatenate([ok_r[~leak], ok_s[~leak]]).mean()),
 "n_clean_pairs": int((~leak).sum()),
 # or keep every synthetic image and weight the two classes equally
 "clean_balanced": float((ok_r[~leak].mean() + ok_s.mean()) / 2),
}
json.dump(res, open(f"{OUT}/step_a.json", "w"), indent=1)
print(json.dumps(res, indent=1))

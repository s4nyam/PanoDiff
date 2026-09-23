"""Step B: retrain the classifier from scratch with the recipe of vit_attention.py (8 epochs, batch 32,
AdamW 3e-4, weight decay 0.05, one-cycle), on either the original random split or a duplicate-aware
split. One run per SLURM task; RUNS below maps the task to (split, seed)."""
import json, os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from common import OUT, ViT, Arr, load, random_split, grouped_split, leaked, predict

RUNS = [("random", s) for s in range(4)] + [("grouped", s) for s in range(4)]
rank = int(os.environ.get("SLURM_PROCID", 0))
split, seed = RUNS[rank]
torch.manual_seed(seed); np.random.seed(seed)
real, syn, groups, _ = load()
n = len(groups)
tr_r, va_r, tr_s, va_s = (random_split(n, seed) if split == "random" else grouped_split(groups, seed))
leak = leaked(groups, tr_r, va_r)
dev = torch.device("cuda")
dtr = DataLoader(Arr(real, syn, tr_r, tr_s), batch_size=32, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
model = ViT().to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, epochs=8, steps_per_epoch=len(dtr))
for ep in range(8):
    model.train()
    for x, y in dtr:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        logits, _ = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    model.eval()
    pr = predict(model, real, va_r, dev); ps = predict(model, syn, va_s, dev)
    acc = float(np.concatenate([pr == 0, ps == 1]).mean())
    print(f"{split} seed {seed} epoch {ep + 1}/8 held-out acc {acc:.4f}", flush=True)
ok_r, ok_s = pr == 0, ps == 1
res = {"split": split, "seed": seed, "n_train_per_class": [len(tr_r), len(tr_s)], "n_val_per_class": [len(va_r), len(va_s)],
       "acc": acc, "real_acc": float(ok_r.mean()), "syn_acc": float(ok_s.mean()),
       "n_real_leaked": int(leak.sum()),
       "real_acc_leaked": float(ok_r[leak].mean()) if leak.any() else None,
       "real_acc_clean": float(ok_r[~leak].mean())}
json.dump(res, open(f"{OUT}/step_b_{split}_{seed}.json", "w"), indent=1)
print(json.dumps(res), flush=True)

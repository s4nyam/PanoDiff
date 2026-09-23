"""Make NVlabs/stylegan3 usable on LUMI (AMD MI250X / ROCm).

The repo's custom HIP kernels cannot build in the LUMI PyTorch container: the
container has no host `c++`, and the NVCC-only flags `--use_fast_math` /
`--allow-unsupported-compiler` are rejected by hipcc's clang++. stylegan2-ada
tolerated this (it caught the failure and used the reference path); stylegan3
dropped the try/except, so _init() raises and training dies.

Both ops ship pure-PyTorch `_ref` implementations that the call sites already
fall back to when _init() is falsy -- these are the same paths used for CPU, so
they are numerically correct, only slower. This patch restores the fallback.
Idempotent.
"""
import re, sys, os
R = os.path.join(os.environ["STYLEGAN3_REPO"], "torch_utils", "ops")
TARGETS = ["bias_act.py", "upfirdn2d.py", "filtered_lrelu.py"]
MARK = "# [LUMI-ROCm patch]"

for fn in TARGETS:
    p = os.path.join(R, fn)
    s = open(p).read()
    if MARK in s:
        print(f"  {fn}: already patched"); continue
    m = re.search(r"def _init\(\):\n(.*?)\n    return True\n", s, re.S)
    if not m:
        print(f"  {fn}: _init() pattern not found -- SKIPPED"); continue
    body = m.group(1)
    indented = "\n".join("    " + l if l.strip() else l for l in body.split("\n"))
    new = (f"def _init():\n"
           f"    {MARK} kernels cannot build on ROCm; fall back to the\n"
           f"    # repo's own pure-PyTorch reference path instead of raising.\n"
           f"    global _plugin\n"
           f"    try:\n"
           f"{indented}\n"
           f"    except Exception as e:\n"
           f"        import warnings\n"
           f"        if not getattr(_init, '_warned', False):\n"
           f"            warnings.warn(f'[LUMI] {fn} CUDA kernels unavailable ({{type(e).__name__}}); "
           f"using reference implementation.')\n"
           f"            _init._warned = True\n"
           f"        return False\n"
           f"    return True\n")
    s = s[:m.start()] + new + s[m.end():]
    open(p, "w").write(s)
    print(f"  {fn}: patched")
print("done")

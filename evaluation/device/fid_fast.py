"""Fast exact Frechet distance.

scipy.linalg.sqrtm on a 2048x2048 product costs roughly a minute, which makes
any bootstrap over subsamples infeasible. The trace term can instead be written
symmetrically,

    Tr((S1 S2)^{1/2}) = Tr((S1^{1/2} S2 S1^{1/2})^{1/2}),

where the inner matrix is symmetric positive semi-definite, so its eigenvalues
come from eigvalsh in a couple of seconds. The value is identical up to
floating point; only the route differs. The __main__ block checks that against
scipy on random data.
"""
import numpy as np


def _sqrt_psd(S):
    w, V = np.linalg.eigh(S)
    w = np.clip(w, 0, None)
    return (V * np.sqrt(w)) @ V.T


def frechet_fast(mu1, S1, mu2, S2):
    d = mu1 - mu2
    A = _sqrt_psd(S1)
    M = A @ S2 @ A
    M = (M + M.T) / 2
    ev = np.clip(np.linalg.eigvalsh(M), 0, None)
    return float(d.dot(d) + np.trace(S1) + np.trace(S2) - 2 * np.sqrt(ev).sum())


def stats_from(F):
    mu = F.mean(0)
    D = F - mu
    return mu, D.T @ D / (len(F) - 1)


if __name__ == "__main__":
    import time
    from scipy import linalg

    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 2048))
    Y = rng.normal(size=(600, 2048)) * 1.1 + 0.3
    m1, s1 = stats_from(X)
    m2, s2 = stats_from(Y)

    t0 = time.time()
    fast = frechet_fast(m1, s1, m2, s2)
    t1 = time.time()
    cm, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if np.iscomplexobj(cm):
        cm = cm.real
    slow = float((m1 - m2).dot(m1 - m2) + np.trace(s1) + np.trace(s2)
                 - 2 * np.trace(cm))
    t2 = time.time()
    print(f"fast  {fast:.6f}  in {t1-t0:.2f}s")
    print(f"sqrtm {slow:.6f}  in {t2-t1:.2f}s")
    print(f"relative difference {abs(fast-slow)/abs(slow):.3e}   "
          f"speedup {(t2-t1)/max(t1-t0,1e-9):.1f}x")

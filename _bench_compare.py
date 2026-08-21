"""A/B comparison + benchmark: original fabada (git HEAD) vs optimized."""
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

# ---- load ORIGINAL implementation from _fabada_orig.py (git HEAD export) ----
with open(os.path.join(ROOT, "_fabada_orig.py")) as f:
    orig_src = f.read()

_ns = {"np": np}
exec(compile(orig_src, "fabada_orig.py", "exec"), _ns)
fabada_old = _ns["fabada"]

# ---- load OPTIMIZED implementation from the working tree ----
import sys
sys.path.insert(0, ROOT)
import fabada as mod
fabada_new = mod.fabada


def compare_1d(label, data, variance):
    old = fabada_old(data, variance)
    new = fabada_new(data, variance)
    d = np.abs(old - new)
    print(f"[1D] {label:22s} shape={data.shape} max|old-new|={d.max():.3e} "
          f"allclose(rtol=1e-8)={np.allclose(old, new, rtol=1e-8, atol=1e-8)}")


def compare_2d(label, data, variance):
    old = fabada_old(data, variance)
    new = fabada_new(data, variance)
    d = np.abs(old - new)
    print(f"[2D] {label:22s} shape={data.shape} max|old-new|={d.max():.3e} "
          f"allclose(rtol=1e-8)={np.allclose(old, new, rtol=1e-8, atol=1e-8)}")


rng = np.random.default_rng(12431)

# 1D spectrum-like
x = np.linspace(0, 4 * np.pi, 1430)
sig = np.sin(x) + 0.5 * np.sin(3 * x)
sig = (sig - sig.min()) / (sig.max() - sig.min()) * 255
for nstd in (5, 10, 25):
    noisy = sig + rng.normal(0, nstd, sig.shape)
    compare_1d(f"noise={nstd}, scalar var", noisy, float(nstd ** 2))
    compare_1d(f"noise={nstd}, array var", noisy, np.full_like(noisy, nstd ** 2))

# 2D image-like
img = np.tile(sig[:, None], (1, 256))
for nstd in (10, 20):
    noisy = img + rng.normal(0, nstd, img.shape)
    compare_2d(f"noise={nstd}, scalar var", noisy, float(nstd ** 2))

# ---- benchmark ----
def bench(fn, data, variance, reps=5):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(data, variance)
        best = min(best, time.perf_counter() - t0)
    return best


print("\n--- benchmark (best of 5) ---")
spec = sig + rng.normal(0, 10, sig.shape)
img2 = img + rng.normal(0, 15, img.shape)

t_old1 = bench(fabada_old, spec, 100.0)
t_new1 = bench(fabada_new, spec, 100.0)
print(f"1D 1430 pts : old={t_old1*1e3:8.2f} ms  new={t_new1*1e3:8.2f} ms  "
      f"speedup={t_old1/t_new1:.2f}x")

t_old2 = bench(fabada_old, img2, 225.0)
t_new2 = bench(fabada_new, img2, 225.0)
print(f"2D 1430x256 : old={t_old2*1e3:8.2f} ms  new={t_new2*1e3:8.2f} ms  "
      f"speedup={t_old2/t_new2:.2f}x")

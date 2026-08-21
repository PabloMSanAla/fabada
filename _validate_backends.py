"""Validate the multi-backend fabada: numpy vs numba (vs cupy if available)."""
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

import sys
sys.path.insert(0, ROOT)
import fabada as fabada_mod
from fabada import fabada as fabada_fn
fabada = fabada_fn

print("HAS_NUMBA =", fabada_mod._HAS_NUMBA, "| cupy =", fabada_mod._load_cupy())

rng = np.random.default_rng(7)

# ---------- 1D ----------
x = np.linspace(0, 4 * np.pi, 1430)
sig = np.sin(x) + 0.5 * np.sin(3 * x)
sig = (sig - sig.min()) / (sig.max() - sig.min()) * 255
noisy1 = sig + rng.normal(0, 10, sig.shape)

r_np = fabada(noisy1, 100.0, backend="numpy")
r_nb = fabada(noisy1, 100.0, backend="numba")
r_au = fabada(noisy1, 100.0, backend="auto")

print("\n[1D 1430] numpy vs numba max diff:", np.abs(r_np - r_nb).max())
print("[1D 1430] auto is numba?       :", np.array_equal(r_au, r_nb))

# ---------- 2D ----------
img = np.tile(sig[:, None], (1, 256))
noisy2 = img + rng.normal(0, 15, img.shape)

r2_np = fabada(noisy2, 225.0, backend="numpy")
r2_nb = fabada(noisy2, 225.0, backend="numba")
print("[2D 1430x256] numpy vs numba max diff:", np.abs(r2_np - r2_nb).max())

# input is not mutated
inp = noisy1.copy()
fabada(inp, 100.0, backend="numba")
print("[1D] input unchanged by numba:", np.array_equal(inp, noisy1))

# variance as array
var_arr = np.full_like(noisy1, 100.0)
print("[1D] array-variance numba ok :",
      np.allclose(fabada(noisy1, var_arr, backend="numba"),
                  fabada(noisy1, var_arr, backend="numpy"), atol=1e-6))

# ---------- benchmarks ----------
def bench(fn, *a, reps=5, **kw):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*a, **kw)
        best = min(best, time.perf_counter() - t0)
    return best

print("\n--- 1D 1430 pts (best of 5) ---")
t1 = bench(fabada, noisy1, 100.0, backend="numpy")
t2 = bench(fabada, noisy1, 100.0, backend="numba")
print(f"numpy: {t1*1e3:8.2f} ms   numba: {t2*1e3:8.2f} ms   speedup={t1/t2:.2f}x")

print("--- 2D 1430x256 (best of 5) ---")
t1 = bench(fabada, noisy2, 225.0, backend="numpy")
t2 = bench(fabada, noisy2, 225.0, backend="numba")
print(f"numpy: {t1*1e3:8.2f} ms   numba: {t2*1e3:8.2f} ms   speedup={t1/t2:.2f}x")

# ---------- backend selection / errors ----------
try:
    fabada(noisy1, 100.0, backend="cupy")
    print("cupy: unexpectedly available")
except ImportError as e:
    print("cupy request -> ImportError (expected):", str(e)[:60])

try:
    fabada(noisy1, 100.0, backend="bad")
except ValueError as e:
    print("bad backend -> ValueError (expected):", str(e)[:50])

# verbose forces numpy backend
r_v = fabada(noisy1[:200], 100.0, backend="numba", verbose=True)
print("verbose forced numpy:",
      np.allclose(r_v, fabada(noisy1[:200], 100.0, backend="numpy")))

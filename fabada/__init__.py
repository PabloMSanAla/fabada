"""
FABADA is a non-parametric noise reduction technique based on Bayesian
inference that iteratively evaluates possibles moothed  models  of  
the  data introduced,  obtaining  an  estimation  of the  underlying  
signal that is statistically  compatible  with the  noisy  measurements.

based on P.M. Sanchez-Alarcon, Y. Ascasibar, 2022
"Fully Adaptive Bayesian Algorithm for Data Analysis. FABADA"

Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
Everyone is permitted to copy and distribute verbatim copies
of this license document, but changing it is not allowed.
"""

from __future__ import print_function, division
import math
import numpy as np
from typing import Union
from time import time as time
from scipy import ndimage
import scipy.stats as stats
import sys
from math import exp as _exp, log as _log, lgamma as _lgamma

# ---------------------------------------------------------------------------
# Optional accelerators.  Everything is optional: if Numba or CuPy are missing
# the package transparently falls back to the pure-NumPy implementation.
# ---------------------------------------------------------------------------
try:
    from numba import njit, prange
    _HAS_NUMBA = True
except Exception:  # pragma: no cover - numba not installed
    njit = None
    prange = None
    _HAS_NUMBA = False

_cupy_mod = None  # lazily imported (importing cupy is slow)


def _load_cupy():
    """Lazily import CuPy and return it, or ``False`` when unavailable."""
    global _cupy_mod
    if _cupy_mod is None:
        try:
            import cupy as _cp
            _cupy_mod = _cp if _cp.cuda.runtime.getDeviceCount() > 0 else False
        except Exception:  # pragma: no cover - cupy / CUDA not available
            _cupy_mod = False
    return _cupy_mod


def _is_cupy(x):
    """Detect a CuPy array without forcing a CuPy import."""
    return x.__class__.__module__.split(".")[0] == "cupy"


def fabada(
    data: Union[np.array, list],
    data_variance: Union[np.array, list, float],
    max_iter: int = 3000,
    verbose: bool = False,
    backend: str = "auto",
    **kwargs
) -> np.array:

    """
        FABADA for any kind of data (1D or 2D). Performs noise reduction in input.

        FABADA is a non-parametric noise reduction technique based on Bayesian
        inference that iteratively evaluates possibles smoothed models of
        the data introduced, obtaining an estimation of the underlying
        signal that is statistically compatible with the noisy measurements.

        based on Sanchez-Alarcon, P.M. & Ascasibar, Y. 2022
        "Fully Adaptive Bayesian Algorithm for Data Analysis. FABADA"
        arXiv:2201.05145

        Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
        Everyone is permitted to copy and distribute verbatim copies
        of this license document, but changing it is not allowed.

    :param data: Noisy measurements, either 1 dimension (M) or 2 dimensions (MxN)

    :param data_variance: Estimated variance of the input, either MxN array, list
                          or float assuming all point have same variance.

    :param max_iter: 3000 (default). Maximum of iterations to converge in solution.

    :param verbose: False (default) or True. Spits some informations about process.
                    Note: the per-iteration verbose output is only implemented in
                    the 'numpy' backend (requesting verbose forces numpy backend).

    :param backend: Execution backend, one of:
                    - 'auto'   (default) picks the best available one: CuPy if the
                               input is already a CuPy array, else Numba if it is
                               installed, else the pure NumPy implementation.
                    - 'numpy'  pure NumPy reference implementation (always works).
                    - 'numba'  JIT-compiled fused CPU loops (fastest on CPU).
                    - 'cupy'   GPU implementation via CuPy (best for large arrays).
                    An ImportError is raised if a requested backend is unavailable.

    :param **kwargs: Future Work.

    :return bayes: denoised estimation of the data with same size as input.
                   Always returned as a NumPy array.
    """

    # ----- resolve the execution backend -----
    if backend == "auto":
        if (_is_cupy(data) or _is_cupy(data_variance)) and _load_cupy():
            backend = "cupy"
        elif _HAS_NUMBA:
            backend = "numba"
        else:
            backend = "numpy"
    elif backend not in ("numpy", "numba", "cupy"):
        raise ValueError(
            "backend must be one of 'auto', 'numpy', 'numba', 'cupy' (got %r)"
            % (backend,)
        )
    elif backend == "numba" and not _HAS_NUMBA:
        raise ImportError(
            "backend='numba' requires the 'numba' package (pip install numba)."
        )
    elif backend == "cupy" and not _load_cupy():
        raise ImportError(
            "backend='cupy' requires CuPy and an available CUDA device "
            "(e.g. pip install cupy-cuda12x)."
        )

    # The per-iteration verbose output is only implemented in the NumPy backend.
    if verbose:
        backend = "numpy"

    # ----- dispatch -----
    if backend == "cupy":
        cp = _load_cupy()
        if not _is_cupy(data):
            data = cp.asarray(data)
        if not _is_cupy(data_variance):
            data_variance = cp.asarray(data_variance)
        return _fabada_cupy(data, data_variance, max_iter)

    if _is_cupy(data):
        data = _load_cupy().asnumpy(data)
    if _is_cupy(data_variance):
        data_variance = _load_cupy().asnumpy(data_variance)

    if backend == "numba":
        return _fabada_numba(data, data_variance, max_iter)

    return _fabada_numpy(data, data_variance, max_iter, verbose)


def _fabada_numpy(data, data_variance, max_iter=3000, verbose=False):

    """
        FABADA for any kind of data (1D or 2D). Performs noise reduction in input.

        FABADA is a non-parametric noise reduction technique based on Bayesian
        inference that iteratively evaluates possibles smoothed  models  of  
        the  data introduced,  obtaining  an  estimation  of the  underlying  
        signal that is statistically  compatible  with the  noisy  measurements.

        based on Sanchez-Alarcon, P.M. & Ascasibar, Y.  2022
        "Fully Adaptive Bayesian Algorithm for Data Analysis. FABADA"
        arXiv:2201.05145

        Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
        Everyone is permitted to copy and distribute verbatim copies
        of this license document, but changing it is not allowed.

    :param data: Noisy measurements, either 1 dimension (M) or 2 dimensions (MxN)

    :param data_variance: Estimated variance of the input, either MxN array, list
                          or float assuming all point have same variance.

    :param max_iter: 3000 (default). Maximum of iterations to converge in solution.

    :param verbose: False (default) or True. Spits some informations about process.

    :param **kwargs: Future Work.

    :return bayes: denoised estimation of the data with same size as input.
    """
    

    data = np.array(data, dtype=float)
    data_variance = np.array(data_variance, dtype=float)

    nan_mask = np.isnan(data)
    data[nan_mask] = 0.0

    if verbose:
        if len(data.shape) == 1:
            print("FABADA 1-D initialize")
        elif len(data.shape) == 2:
            print("FABADA 2-D initialize")
        else:
            print("Warning: Size of array not supported")

    if data_variance.size != data.size:
        data_variance = data_variance * np.ones_like(data)
        data_variance[nan_mask] = 1e-15

    # ---- PRE-COMPUTED CONSTANTS (HOISTED OUT OF THE ITERATION LOOP) ----
    data_size = data.size
    inv_data_variance = 1.0 / data_variance          # 1 / sigma^2 (constant)
    data_over_variance = data * inv_data_variance    # data / sigma^2 (constant)
    two_pi = 2.0 * np.pi
    # chi2 pdf: f(x; k) = x^(k/2-1) * e^(-x/2) / (2^(k/2) * Gamma(k/2))
    half_df_minus_1 = 0.5 * data_size - 1.0
    chi2_log_norm = -0.5 * data_size * _log(2.0) - _lgamma(0.5 * data_size)

    # INITIALIZING ALGORITMH ITERATION ZERO
    t = time()
    posterior_mean = data
    posterior_variance = data_variance
    # Evidence(0, sqrt(var), 0, var) == exp(-0.5) / sqrt(2*pi*var)
    initial_evidence = np.exp(-0.5) / np.sqrt(two_pi * data_variance)
    evidence = initial_evidence
    chi2_pdf, chi2_data, iteration = 0.0, data_size, 0
    chi2_pdf_derivative, chi2_data_min = 0.0, data_size
    bayesian_weight = np.zeros_like(data)
    bayesian_model = np.zeros_like(data)
    evidence_previous = np.mean(evidence)

    converged = False

    try:
        while not converged:

            if verbose:
                print('\rIteration = %5d ;' % iteration +
                    '<E> = %4.2f ; ' % evidence_previous +
                    'Chi^2 = %3.4e/%3.3e ' % (chi2_data, data_size), end='')

            chi2_pdf_previous = chi2_pdf
            chi2_pdf_derivative_previous = chi2_pdf_derivative
            evidence_previous = np.mean(evidence)

            iteration += 1  # Check number of iterartions done

            # GENERATES PRIORS
            prior_mean = running_mean(posterior_mean)
            prior_variance = posterior_variance

            # APPLY BAYES' THEOREM
            inv_prior_variance = 1.0 / prior_variance
            posterior_variance = 1.0 / (inv_prior_variance + inv_data_variance)
            posterior_mean = (
                prior_mean * inv_prior_variance + data_over_variance
            ) * posterior_variance

            # EVALUATE EVIDENCE (inlined; var1 + var2 computed once as s)
            s = prior_variance + data_variance
            diff = prior_mean - data
            evidence = np.exp(-(diff * diff) / (2.0 * s)) / np.sqrt(two_pi * s)
            evidence_derivative = np.mean(evidence) - evidence_previous

            # EVALUATE CHI2
            chi2_data = float(
                np.sum((data - posterior_mean) ** 2 * inv_data_variance)
            )
            chi2_pdf = _exp(
                half_df_minus_1 * _log(chi2_data) - 0.5 * chi2_data + chi2_log_norm
            )
            chi2_pdf_derivative = chi2_pdf - chi2_pdf_previous
            chi2_pdf_snd_derivative = chi2_pdf_derivative - chi2_pdf_derivative_previous

            # COMBINE MODELS FOR THE ESTIMATION
            model_weight = evidence * chi2_data
            np.add(bayesian_weight, model_weight, out=bayesian_weight)
            np.add(bayesian_model, model_weight * posterior_mean, out=bayesian_model)

            if iteration == 1:
                chi2_data_min = chi2_data

            # CHECK CONVERGENCE
            if (
                (chi2_data > data_size and chi2_pdf_snd_derivative >= 0)
                and (evidence_derivative < 0)
                or (iteration > max_iter)
            ):

                converged = True

                # COMBINE ITERATION ZERO
                model_weight = initial_evidence * chi2_data_min
                np.add(bayesian_weight, model_weight, out=bayesian_weight)
                np.add(bayesian_model, model_weight * data, out=bayesian_model)

    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise

    bayes = bayesian_model / bayesian_weight

    if verbose:
        print('\rIteration = %5d ; ' % iteration +
                    '<E> = %4.2f ; ' % np.mean(evidence) +
                    'Chi^2 = %3.4e/%3.3e ' % (chi2_data, data_size), end='')
        print(
            "\nFinish at {} iterations".format(iteration),
            " and with an execute time of {:3.2f} seconds.".format(time() - t),
        )

    return bayes


def running_mean(dat):

    mean = np.asarray(dat).copy()
    dim = len(mean.shape)

    if dim == 1:
        mean[:-1] += dat[1:]
        mean[1:] += dat[:-1]
        mean[1:-1] /= 3
        mean[0] /= 2
        mean[-1] /= 2
    elif dim == 2:
        mean[:-1, :] += dat[1:, :]
        mean[1:, :] += dat[:-1, :]
        mean[:, :-1] += dat[:, 1:]
        mean[:, 1:] += dat[:, :-1]
        mean[1:-1, 1:-1] /= 5
        mean[0, 1:-1] /= 4
        mean[-1, 1:-1] /= 4
        mean[1:-1, 0] /= 4
        mean[1:-1, -1] /= 4
        mean[0, 0] /= 3
        mean[-1, -1] /= 3
        mean[0, -1] /= 3
        mean[-1, 0] /= 3
    else:
        print("Warning: Size of array not supported")
    return mean


def Evidence(mu1, mu2, var1, var2):
    return np.exp(-((mu1 - mu2) ** 2) / (2 * (var1 + var2))) / np.sqrt(
        2 * np.pi * (var1 + var2)
    )


def PSNR(recover, signal, L=255):
    MSE = np.sum((recover - signal) ** 2) / (recover.size)
    return 10 * np.log10((L) ** 2 / MSE)


# ---------------------------------------------------------------------------
# Numba backend (CPU, JIT-compiled, fused loops).
# Only defined when Numba is importable; otherwise `_numba_fabada` raises and
# `_fabada_numba` falls back to the pure-NumPy implementation.
# ---------------------------------------------------------------------------
if _HAS_NUMBA:

    @njit(cache=True, fastmath=True)
    def _numba_running_mean_1d(dat, out):
        n = dat.shape[0]
        if n == 1:
            out[0] = dat[0] * 0.5
            return
        out[0] = (dat[0] + dat[1]) * 0.5
        out[n - 1] = (dat[n - 1] + dat[n - 2]) * 0.5
        for i in range(1, n - 1):
            out[i] = (dat[i - 1] + dat[i] + dat[i + 1]) / 3.0

    @njit(cache=True, fastmath=True)
    def _numba_running_mean_2d(dat, out):
        rows = dat.shape[0]
        cols = dat.shape[1]
        for i in range(rows):
            for j in range(cols):
                if 0 < i < rows - 1 and 0 < j < cols - 1:
                    out[i, j] = (
                        dat[i - 1, j] + dat[i + 1, j]
                        + dat[i, j - 1] + dat[i, j + 1] + dat[i, j]
                    ) / 5.0
                elif 0 < i < rows - 1:  # left / right column
                    if j == 0:
                        out[i, j] = (
                            dat[i - 1, j] + dat[i + 1, j] + dat[i, j + 1] + dat[i, j]
                        ) / 4.0
                    else:
                        out[i, j] = (
                            dat[i - 1, j] + dat[i + 1, j] + dat[i, j - 1] + dat[i, j]
                        ) / 4.0
                elif 0 < j < cols - 1:  # top / bottom row
                    if i == 0:
                        out[i, j] = (
                            dat[i, j] + dat[i + 1, j] + dat[i, j - 1] + dat[i, j + 1]
                        ) / 4.0
                    else:
                        out[i, j] = (
                            dat[i, j] + dat[i - 1, j] + dat[i, j - 1] + dat[i, j + 1]
                        ) / 4.0
                else:  # corners (3-point mean)
                    s = dat[i, j]
                    if i > 0:
                        s += dat[i - 1, j]
                    if i < rows - 1:
                        s += dat[i + 1, j]
                    if j > 0:
                        s += dat[i, j - 1]
                    if j < cols - 1:
                        s += dat[i, j + 1]
                    out[i, j] = s / 3.0

    @njit(cache=True, fastmath=True)
    def _numba_fabada(data, data_variance, max_iter):
        n = data.size
        two_pi = 2.0 * math.pi

        inv_var = 1.0 / data_variance
        data_over_var = data * inv_var
        initial_evidence = math.exp(-0.5) / np.sqrt(two_pi * data_variance)

        # chi2 pdf: f(x; k) = x^(k/2-1) e^(-x/2) / (2^(k/2) Gamma(k/2))
        half_df_minus_1 = 0.5 * n - 1.0
        chi2_log_norm = -0.5 * n * math.log(2.0) - math.lgamma(0.5 * n)

        posterior_mean = data.copy()
        posterior_var = data_variance.copy()
        prior_mean = np.empty_like(data)
        evidence = np.empty_like(data)
        bayes_w = np.zeros_like(data)
        bayes_m = np.zeros_like(data)

        # flat (contiguous) views used by the fused passes
        pm = posterior_mean.ravel()
        pv = posterior_var.ravel()
        pr = prior_mean.ravel()
        ev = evidence.ravel()
        bw = bayes_w.ravel()
        bm = bayes_m.ravel()
        dd = data.ravel()
        vv = data_variance.ravel()
        iv = inv_var.ravel()
        dov = data_over_var.ravel()
        iev = initial_evidence.ravel()

        ev_prev = 0.0
        for i in range(n):
            ev_prev += iev[i]
        ev_prev /= n

        chi2_data = float(n)
        chi2_pdf = 0.0
        chi2_pdf_deriv = 0.0
        chi2_min = float(n)
        ndim = data.ndim

        for it in range(1, max_iter + 1):
            if ndim == 1:
                _numba_running_mean_1d(posterior_mean, prior_mean)
            else:
                _numba_running_mean_2d(posterior_mean, prior_mean)

            # pass A: Bayes' theorem + evidence + reductions (this iteration)
            s_ev = 0.0
            chi2_s = 0.0
            for i in range(n):
                inv_prior = 1.0 / pv[i]
                pv_new = 1.0 / (inv_prior + iv[i])
                pm_new = (pr[i] * inv_prior + dov[i]) * pv_new

                s = pv[i] + vv[i]
                d = pr[i] - dd[i]
                e = math.exp(-(d * d) / (2.0 * s)) / math.sqrt(two_pi * s)
                ev[i] = e
                s_ev += e

                r = dd[i] - pm_new
                chi2_s += r * r * iv[i]

                pv[i] = pv_new
                pm[i] = pm_new

            # scalar bookkeeping (identical to the numpy backend)
            ev_deriv = s_ev / n - ev_prev
            ev_prev = s_ev / n
            chi2_data = chi2_s
            chi2_pdf_new = math.exp(
                half_df_minus_1 * math.log(chi2_data) - 0.5 * chi2_data + chi2_log_norm
            )
            chi2_pdf_deriv_new = chi2_pdf_new - chi2_pdf
            chi2_snd = chi2_pdf_deriv_new - chi2_pdf_deriv
            chi2_pdf = chi2_pdf_new
            chi2_pdf_deriv = chi2_pdf_deriv_new
            if it == 1:
                chi2_min = chi2_data

            # pass B: combine models, weight = evidence * chi2_data (this iter)
            for i in range(n):
                w = ev[i] * chi2_data
                bw[i] += w
                bm[i] += w * pm[i]

            # CHECK CONVERGENCE
            if (chi2_data > n and chi2_snd >= 0.0) and ev_deriv < 0.0:
                break

        # COMBINE ITERATION ZERO
        mw = initial_evidence * chi2_min
        return (bayes_m + mw * data) / (bayes_w + mw)

    # Parallel variant: identical math, but the elementwise passes use `prange`.
    # Only worth it for large arrays (thread/barrier overhead dominates on small
    # inputs), so `_fabada_numba` picks it based on a size threshold.
    @njit(cache=True, parallel=True, fastmath=True)
    def _numba_fabada_par(data, data_variance, max_iter):
        n = data.size
        two_pi = 2.0 * math.pi

        inv_var = 1.0 / data_variance
        data_over_var = data * inv_var
        initial_evidence = math.exp(-0.5) / np.sqrt(two_pi * data_variance)

        half_df_minus_1 = 0.5 * n - 1.0
        chi2_log_norm = -0.5 * n * math.log(2.0) - math.lgamma(0.5 * n)

        posterior_mean = data.copy()
        posterior_var = data_variance.copy()
        prior_mean = np.empty_like(data)
        evidence = np.empty_like(data)
        bayes_w = np.zeros_like(data)
        bayes_m = np.zeros_like(data)

        pm = posterior_mean.ravel()
        pv = posterior_var.ravel()
        pr = prior_mean.ravel()
        ev = evidence.ravel()
        bw = bayes_w.ravel()
        bm = bayes_m.ravel()
        dd = data.ravel()
        vv = data_variance.ravel()
        iv = inv_var.ravel()
        dov = data_over_var.ravel()
        iev = initial_evidence.ravel()

        ev_prev = 0.0
        for i in range(n):
            ev_prev += iev[i]
        ev_prev /= n

        chi2_data = float(n)
        chi2_pdf = 0.0
        chi2_pdf_deriv = 0.0
        chi2_min = float(n)
        ndim = data.ndim

        for it in range(1, max_iter + 1):
            if ndim == 1:
                _numba_running_mean_1d(posterior_mean, prior_mean)
            else:
                _numba_running_mean_2d(posterior_mean, prior_mean)

            s_ev = 0.0
            chi2_s = 0.0
            for i in prange(n):
                inv_prior = 1.0 / pv[i]
                pv_new = 1.0 / (inv_prior + iv[i])
                pm_new = (pr[i] * inv_prior + dov[i]) * pv_new

                s = pv[i] + vv[i]
                d = pr[i] - dd[i]
                e = math.exp(-(d * d) / (2.0 * s)) / math.sqrt(two_pi * s)
                ev[i] = e
                s_ev += e

                r = dd[i] - pm_new
                chi2_s += r * r * iv[i]

                pv[i] = pv_new
                pm[i] = pm_new

            ev_deriv = s_ev / n - ev_prev
            ev_prev = s_ev / n
            chi2_data = chi2_s
            chi2_pdf_new = math.exp(
                half_df_minus_1 * math.log(chi2_data) - 0.5 * chi2_data + chi2_log_norm
            )
            chi2_pdf_deriv_new = chi2_pdf_new - chi2_pdf
            chi2_snd = chi2_pdf_deriv_new - chi2_pdf_deriv
            chi2_pdf = chi2_pdf_new
            chi2_pdf_deriv = chi2_pdf_deriv_new
            if it == 1:
                chi2_min = chi2_data

            for i in prange(n):
                w = ev[i] * chi2_data
                bw[i] += w
                bm[i] += w * pm[i]

            if (chi2_data > n and chi2_snd >= 0.0) and ev_deriv < 0.0:
                break

        mw = initial_evidence * chi2_min
        return (bayes_m + mw * data) / (bayes_w + mw)

else:  # pragma: no cover - numba not installed
    def _numba_fabada(*args, **kwargs):
        raise ImportError("The 'numba' backend requires the 'numba' package.")

    def _numba_fabada_par(*args, **kwargs):
        raise ImportError("The 'numba' backend requires the 'numba' package.")


# Numba parallel execution pays off only for large arrays (thread/barrier
# overhead dominates on small inputs). Arrays with >= this many elements use
# the parallel kernel (measured crossover is between ~10k and ~50k elements).
_NUMBA_PAR_THRESHOLD = 50000


def _fabada_numba(data, data_variance, max_iter=3000):
    """FABADA with Numba (fastest on CPU). Falls back to NumPy on any problem."""
    data = np.array(data, dtype=float)
    data_variance = np.array(data_variance, dtype=float)

    nan_mask = np.isnan(data)
    data[nan_mask] = 0.0
    if data_variance.size != data.size:
        data_variance = data_variance * np.ones_like(data)
        data_variance[nan_mask] = 1e-15

    # The Numba stencils cover standard 1D/2D cases; fall back otherwise.
    supported = (data.ndim == 1 and data.size >= 2) or (
        data.ndim == 2 and data.shape[0] >= 3 and data.shape[1] >= 3
    )
    if not supported:
        return _fabada_numpy(data, data_variance, max_iter)

    try:
        kernel = _numba_fabada_par if data.size >= _NUMBA_PAR_THRESHOLD else _numba_fabada
        return kernel(data, data_variance, int(max_iter))
    except Exception:  # pragma: no cover - compilation/runtime failure
        return _fabada_numpy(data, data_variance, max_iter)


# ---------------------------------------------------------------------------
# CuPy backend (GPU). A drop-in mirror of the NumPy algorithm that keeps every
# array on the device for the whole run (single H2D/D2H transfer).
# ---------------------------------------------------------------------------
def _running_mean_cupy(dat):
    mean = dat.copy()
    dim = len(mean.shape)

    if dim == 1:
        mean[:-1] += dat[1:]
        mean[1:] += dat[:-1]
        mean[1:-1] /= 3
        mean[0] /= 2
        mean[-1] /= 2
    elif dim == 2:
        mean[:-1, :] += dat[1:, :]
        mean[1:, :] += dat[:-1, :]
        mean[:, :-1] += dat[:, 1:]
        mean[:, 1:] += dat[:, :-1]
        mean[1:-1, 1:-1] /= 5
        mean[0, 1:-1] /= 4
        mean[-1, 1:-1] /= 4
        mean[1:-1, 0] /= 4
        mean[1:-1, -1] /= 4
        mean[0, 0] /= 3
        mean[-1, -1] /= 3
        mean[0, -1] /= 3
        mean[-1, 0] /= 3
    else:
        raise ValueError("Warning: Size of array not supported")
    return mean


def _fabada_cupy(data, data_variance, max_iter=3000):
    """FABADA on GPU via CuPy (``data`` and ``data_variance`` are CuPy arrays)."""
    cp = _load_cupy()
    if not cp:
        raise ImportError("The 'cupy' backend requires CuPy with CUDA.")

    data = cp.array(data, dtype=cp.float64)  # copy: never mutate the caller's data
    data_variance = cp.array(data_variance, dtype=cp.float64)

    nan_mask = cp.isnan(data)
    data[nan_mask] = 0.0
    if data_variance.size != data.size:
        data_variance = data_variance * cp.ones_like(data)
        data_variance[nan_mask] = 1e-15

    data_size = int(data.size)
    inv_data_variance = 1.0 / data_variance
    data_over_variance = data * inv_data_variance
    two_pi = 2.0 * np.pi
    # chi2 pdf constants (scalars, computed on the host)
    half_df_minus_1 = 0.5 * data_size - 1.0
    chi2_log_norm = -0.5 * data_size * _log(2.0) - _lgamma(0.5 * data_size)

    posterior_mean = data
    posterior_variance = data_variance
    # Evidence(0, sqrt(var), 0, var) == exp(-0.5) / sqrt(2*pi*var)
    initial_evidence = np.exp(-0.5) / cp.sqrt(two_pi * data_variance)
    evidence = initial_evidence
    chi2_pdf, chi2_data, iteration = 0.0, float(data_size), 0
    chi2_pdf_derivative, chi2_data_min = 0.0, float(data_size)
    bayesian_weight = cp.zeros_like(data)
    bayesian_model = cp.zeros_like(data)
    evidence_previous = float(cp.mean(evidence))

    converged = False
    while not converged:
        chi2_pdf_previous = chi2_pdf
        chi2_pdf_derivative_previous = chi2_pdf_derivative
        evidence_previous = float(cp.mean(evidence))

        iteration += 1

        # GENERATES PRIORS
        prior_mean = _running_mean_cupy(posterior_mean)
        prior_variance = posterior_variance

        # APPLY BAYES' THEOREM
        inv_prior_variance = 1.0 / prior_variance
        posterior_variance = 1.0 / (inv_prior_variance + inv_data_variance)
        posterior_mean = (
            prior_mean * inv_prior_variance + data_over_variance
        ) * posterior_variance

        # EVALUATE EVIDENCE (inlined; var1 + var2 computed once as s)
        s = prior_variance + data_variance
        diff = prior_mean - data
        evidence = cp.exp(-(diff * diff) / (2.0 * s)) / cp.sqrt(two_pi * s)
        evidence_derivative = float(cp.mean(evidence)) - evidence_previous

        # EVALUATE CHI2
        chi2_data = float(cp.sum((data - posterior_mean) ** 2 * inv_data_variance))
        chi2_pdf = _exp(
            half_df_minus_1 * _log(chi2_data) - 0.5 * chi2_data + chi2_log_norm
        )
        chi2_pdf_derivative = chi2_pdf - chi2_pdf_previous
        chi2_pdf_snd_derivative = chi2_pdf_derivative - chi2_pdf_derivative_previous

        # COMBINE MODELS FOR THE ESTIMATION
        model_weight = evidence * chi2_data
        cp.add(bayesian_weight, model_weight, out=bayesian_weight)
        cp.add(bayesian_model, model_weight * posterior_mean, out=bayesian_model)

        if iteration == 1:
            chi2_data_min = chi2_data

        # CHECK CONVERGENCE
        if (
            (chi2_data > data_size and chi2_pdf_snd_derivative >= 0)
            and (evidence_derivative < 0)
            or (iteration > max_iter)
        ):
            converged = True

            # COMBINE ITERATION ZERO
            model_weight = initial_evidence * chi2_data_min
            cp.add(bayesian_weight, model_weight, out=bayesian_weight)
            cp.add(bayesian_model, model_weight * data, out=bayesian_model)

    return cp.asnumpy(bayesian_model / bayesian_weight)

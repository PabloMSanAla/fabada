#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test FABADA smoothing scheme"""

from __future__ import print_function, division
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# from time import time as time
import scipy.stats as stats
import scipy.signal


def Evidence(mu1, mu2, var1, var2):
    return np.exp(-(mu1-mu2)**2/(2*(var1+var2))) / np.sqrt(2*np.pi*(var1+var2))


# %%  User-defined parameters


# Data

# name = "arp256"
name = "kurucz"
# name = "SN132D"

spectrum = pd.read_csv(os.path.join('test_spectra', name+'.csv'))
signal = spectrum.flux.to_numpy()
signal = 255 * signal/np.max(signal)
N = len(signal)


# Noise

sigma_noise = 15
data_variance = sigma_noise**2 * np.ones_like(signal)
np.random.seed(12431)


# FABADA

max_iter = 1000
fraction_of_Pmax = .1

sg_max_deg = 5

# Plots

show_intermediate_plots = False
zoom_center = 600
zoom_radius = 90


# %%  Main loop

noise = np.random.normal(0, sigma_noise, signal.shape)
data = signal + noise

posterior_mean = data
posterior_variance = data_variance
initial_evidence = Evidence(data, data, data_variance, data_variance)
max_chi2_pdf = stats.chi2.pdf(N, df=N)

# bayesian_weight = np.array(initial_evidence)
# bayesian_weight = np.array(initial_evidence) * max_chi2_pdf
# bayesian_model = bayesian_weight * data
bayesian_weight = np.zeros_like(signal)
bayesian_model = np.zeros_like(signal)


total_smooth_weight = np.zeros_like(signal)
smooth_model = np.zeros_like(signal)
smooth_variance = np.zeros_like(signal)

sg_window = 1
while sg_window < np.sqrt(N):
    sg_window += 2

    for sg_deg in range(0, np.min([sg_window, sg_max_deg+1])):

        smooth_candidate = scipy.signal.savgol_filter(data, sg_window, sg_deg)
        chi2_candidate = np.sum((data-smooth_candidate)**2/data_variance)
        candidate_weight = stats.chi2.pdf(chi2_candidate, df=N)
        total_smooth_weight += candidate_weight
        print(candidate_weight)
        smooth_model += candidate_weight * smooth_candidate
        smooth_variance += candidate_weight * smooth_candidate**2

smooth_model /= total_smooth_weight
smooth_variance /= total_smooth_weight
smooth_variance -= smooth_model**2
smooth_2sig = 2*np.sqrt(smooth_variance)
mse_smooth = np.mean((signal-smooth_model)**2)
psnr_smooth = 10*np.log10(255**2/mse_smooth)

prior_mean = smooth_model
prior_variance = smooth_variance
posterior_variance = 1/(1/prior_variance + 1/data_variance)
posterior_mean = (prior_mean/prior_variance + data/data_variance
                  )*posterior_variance
smooth_evidence = Evidence(prior_mean, data, prior_variance, data_variance)
mse_posterior = np.mean((signal-posterior_mean)**2)
psnr_posterior = 10*np.log10(255**2/mse_posterior)

initial_evidence = Evidence(data, data, 300*data_variance, data_variance)
bayes = (initial_evidence*data + smooth_evidence*smooth_model
         ) / (initial_evidence + smooth_evidence)
mse_bayes = np.mean((signal-bayes)**2)
psnr_bayes = 10*np.log10(255**2/mse_bayes)


fig, (ax) = plt.subplots(1, 1)
ax.set_title('{:.2f} {:.2f} {:.2f}'.format(psnr_smooth, psnr_posterior, psnr_bayes))
ax.plot(signal, 'r-', alpha=.3)
ax.plot(data, 'k-', alpha=.1)
ax.plot(posterior_mean, 'y-', alpha=1)
ax.plot(bayes, 'k-', alpha=1)
ax.plot(smooth_model+smooth_2sig, 'y-', alpha=.2)
ax.plot(smooth_model-smooth_2sig, 'y-', alpha=.2)
ax.set_xlim(zoom_center-zoom_radius, zoom_center+zoom_radius)
ax.set_ylim(-25, 255)
plt.show()

# %% Bye
print("... Paranoy@ Rulz!")

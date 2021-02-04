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


def running_mean(dat, w=0.):

    mean = np.array(dat)*w
    dim = len(mean.shape)

    if dim == 1:
        mean[:-1] += dat[1:]
        mean[1:] += dat[:-1]
        mean[1:-1] /= (w+2.)
        mean[0] /= (w+1.)
        mean[-1] /= (w+1.)
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
fraction_of_Pmax = .5


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
max_evidence = np.array(initial_evidence)
max_chi2_pdf = stats.chi2.pdf(N, df=N)

bayesian_weight = np.array(initial_evidence) * max_chi2_pdf
bayesian_model = bayesian_weight * data

converged = False
iteration = 0
while not converged:
    iteration += 1

    prior_mean = running_mean(posterior_mean)
    prior_variance = running_mean(posterior_mean**2) - prior_mean**2
    prior_variance = np.max([prior_variance, posterior_variance], axis=0)

    posterior_variance = 1/(1/prior_variance + 1/data_variance)
    posterior_mean = (prior_mean/prior_variance + data/data_variance
                      )*posterior_variance
    evidence = Evidence(prior_mean, data, prior_variance, data_variance)
    chi2_data = np.sum((data-posterior_mean)**2/data_variance)
    chi2_pdf = stats.chi2.pdf(chi2_data, df=N)

    model_weight = evidence * chi2_pdf
    bayesian_weight += model_weight
    bayesian_model += model_weight*posterior_mean
    bayes = bayesian_model/bayesian_weight
    chi2_bayes = np.sum((data-bayes)**2/data_variance)

    if (
            chi2_data > N and chi2_pdf < fraction_of_Pmax*max_chi2_pdf
            ) or (iteration == max_iter):
        converged = True

    if show_intermediate_plots or converged:

        mse = np.mean((signal-posterior_mean)**2)
        psnr = 10*np.log10(255**2/mse)
        mse = np.mean((signal-bayes)**2)
        psnr_bayes = 10*np.log10(255**2/mse)

        fig, (ax) = plt.subplots(1, 1)
        ax.set_title(
            '{}({:.3f}): {:.2f}/{:.4f};{:.2f}/{:.4f}'.format(
                iteration, np.mean(evidence/initial_evidence-1),
                chi2_data/N, psnr,
                chi2_bayes/N, psnr_bayes,
                ))
        # ax.plot(signal, 'r-', alpha=.3)
        ax.plot(data, 'k-', alpha=.1)
        ax.plot(bayes, 'k-', alpha=1)
        ax.plot(posterior_mean, 'y-', alpha=.5)
        ax.set_xlim(zoom_center-zoom_radius, zoom_center+zoom_radius)
        ax.set_ylim(-25, 255)
        plt.show()


# %% Bye
print("... Paranoy@ Rulz!")

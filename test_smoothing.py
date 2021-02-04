#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test FABADA smoothing scheme"""

from __future__ import print_function, division
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# from time import time as time


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


# %%  Read data

# name = "arp256"
name = "SN132D"
# name = "kurucz"
sigma_noise = 25

spectrum = pd.read_csv(os.path.join('test_spectra', name+'.csv'))
signal = spectrum.flux.to_numpy()
signal = 255 * signal/np.max(signal)

# Add noise

# np.random.seed(124311)
noise = np.random.normal(0, sigma_noise, signal.shape)


# %%  Smooth

plots = False
zoom_center = 400
zoom_radius = 400

# extra_noise = .2*np.random.normal(0, sigma_noise, signal.shape)

data = signal + noise
errors = sigma_noise * np.ones_like(data)
data_variance = errors**2
N = len(data)
index = np.arange(N)

smooth_data = running_mean(data)
smooth_variance = np.max([
    running_mean(data**2) - smooth_data**2,
    data_variance], axis=0)

posterior_mean = data
posterior_variance = data_variance
initial_evidence = Evidence(data, data, data_variance, data_variance)
max_evidence = np.array(initial_evidence)
bayesian_weight = np.array(initial_evidence)
bayesian_model = initial_evidence*data
bayes = data

# if plots:
#     plt.figure()

#     ax = plt.subplot(2, 1, 1)
#     ax.plot(initial_evidence, 'k-')
#     ax.set_xlim(zoom_center-zoom_radius, zoom_center+zoom_radius)

#     ax = plt.subplot(2, 1, 2)
#     ax.plot(data, 'k-')
#     band = 2*np.sqrt(data_variance)
#     # ax.fill_between(index, data-band, data+band, 'k', alpha=.1)
#     ax.plot(data, 'k', alpha=.1)
#     ax.plot(posterior_mean, 'r-')
#     band = 2*np.sqrt(posterior_variance)
#     ax.fill_between(index, posterior_mean-band, posterior_mean+band,
#                     'r', alpha=.1)
#     ax.set_xlim(zoom_center-zoom_radius, zoom_center+zoom_radius)

#     plt.show()

converged = False
# while not converged:
max_iter = 2000
for iteration in range(max_iter):
    prior_mean = running_mean(posterior_mean)
    # prior_variance = posterior_variance
    prior_variance = running_mean(posterior_mean**2) - prior_mean**2
    prior_variance = np.max([prior_variance, posterior_variance], axis=0)

    # prior_mean = running_mean(bayes)
    # prior_variance = running_mean(bayes**2) - prior_mean**2
    # prior_variance = np.max([prior_variance, posterior_variance], axis=0)
    # prior_variance = (running_mean(bayes**2)
    #                   - prior_mean**2)
    # mean1 = posterior_mean
    # var1 = posterior_variance
    # mean2 = running_mean(bayes)
    # var2 = running_mean(bayes**2) - mean2**2
    # prior_mean = (mean1+mean2)/2
    # prior_variance = (var1+var2)/2

    posterior_variance = 1/(1/prior_variance + 1/data_variance)
    posterior_mean = (prior_mean/prior_variance + data/data_variance
                      )*posterior_variance
    evidence = Evidence(prior_mean, data, prior_variance, data_variance)
    # evidence *= Evidence(prior_mean, smooth_data, prior_variance, smooth_variance)

    chi2_data = np.mean((data-posterior_mean)**2/data_variance)
    chi2_signal = np.mean((signal-posterior_mean)**2/data_variance)

    model_weight = evidence * np.sqrt(chi2_data)
    bayesian_model += model_weight*posterior_mean
    bayesian_weight += model_weight
    bayes = bayesian_model/bayesian_weight

    # improved = np.where(evidence > max_evidence)
    # impovement = np.sum(
    #     (evidence[improved]-max_evidence[improved])
    #     / initial_evidence[improved])
    # max_evidence[improved] = evidence[improved]

    chi2_data_bayes = np.mean((data-bayes)**2/data_variance)
    chi2_signal_bayes = np.mean((signal-bayes)**2/data_variance)
    chi2_cross = np.mean((posterior_mean-bayes)**2/data_variance)

    # if impovement < np.mean(evidence[-1]/evidence[0]-1):
    if chi2_data > 1 + chi2_cross:
        # plots = True
        converged = True

    if plots or (iteration == max_iter-1):
        f, (ax2) = plt.subplots(1, 1, sharex=True)

        ax2.set_title(
            '{} : {:.3f} {:.2f}/{:.4f};{:.2f}/{:.4f};{:.4f}'.format(
                iteration,
                np.mean(evidence/initial_evidence-1),
                # len(improved[0]), impovement,
                chi2_data, chi2_signal,
                chi2_data_bayes, chi2_signal_bayes, chi2_cross,
                ))
        # ax1.plot(evidence/initial_evidence, 'b-')
        # ax1.plot(np.max(evidence, axis=0)/evidence[0], 'k-')
        # ax1.plot(posterior_variance/data_variance, 'k-')
        # ax1.plot((signal-posterior_mean)**2/data_variance, 'r-')
        # ax1.grid('both')
        # ax1.set_ylim(-.1, 2.1)

        ax2.plot(signal, 'r-', alpha=.5)
        ax2.plot(data, 'k-', alpha=.1)
        band = np.sqrt(data_variance)
        # ax2.fill_between(index, data-band, data+band, 'k', alpha=.1)
        # ax2.plot(posterior_mean, 'r-')
        ax2.plot(bayes, 'k-', alpha=1)
        ax2.plot(posterior_mean, 'y-', alpha=.5)
        band = 2*np.sqrt(posterior_variance)
        # ax2.fill_between(index, posterior_mean-band, posterior_mean+band,
                         # 'r', alpha=.1)
        ax2.set_xlim(zoom_center-zoom_radius, zoom_center+zoom_radius)
        # ax2.grid('both')
        ax2.set_ylim(np.percentile(data, 1), np.percentile(data, 99.5))

        plt.show()

    if converged:
        break


bayes = bayesian_model/bayesian_weight
chi2_data_bayes = np.mean((data-bayes)**2/data_variance)

median_filtered = np.concatenate([[bayes[0]],
                                  np.median([bayes[:-2],
                                             bayes[1:-1],
                                             bayes[2:]], axis=0),
                                  [bayes[-1]]])
chi2_data_median = np.mean((data-median_filtered)**2/data_variance)
wb = np.exp(-chi2_data_bayes)
wm = np.exp(-chi2_data_median)
wi = np.exp(-chi2_data)
bayes = (wb*bayes + wm*median_filtered) / (wb + wm)
# bayes = (wb*bayes + wm*median_filtered + wi*posterior_mean) / (wb + wm + wi)
chi2_data_bayes = np.mean((data-bayes)**2/data_variance)
chi2_signal_bayes = np.mean((signal-bayes)**2/data_variance)
chi2_cross = np.mean((posterior_mean-bayes)**2/data_variance)


plt.figure()
plt.title(
    '{} : {:.3f} - {:.2f}/{:.4f} - {:.2f}/{:.4f} - {:.4f}'.format(
        iteration,
        np.mean(evidence/initial_evidence-1),
        # len(improved[0]), impovement,
        chi2_data, chi2_signal,
        chi2_data_bayes, chi2_signal_bayes, chi2_cross,
        ))

plt.plot(signal, 'r-', alpha=.3)
plt.plot(data, 'k-', alpha=.1)
band = np.sqrt(data_variance)
plt.plot(bayes, 'k-', alpha=1)
# plt.plot(posterior_mean, 'y-', alpha=1)
plt.xlim(zoom_center-zoom_radius, zoom_center+zoom_radius)
plt.ylim(np.percentile(data, 1), np.percentile(data, 99.5))
plt.show()


# bayes = median_filtered
# chi2_data_bayes = np.mean((data-bayes)**2/data_variance)
# chi2_signal_bayes = np.mean((signal-bayes)**2/data_variance)
# chi2_cross = np.mean((posterior_mean-bayes)**2/data_variance)

# f, (ax2) = plt.subplots(1, 1, sharex=True)

# ax2.set_title(
#     '{} : {:.3f} - {:.2f}/{:.4f} - {:.2f}/{:.4f} - {:.4f}'.format(
#         iteration,
#         np.mean(evidence/initial_evidence-1),
#         # len(improved[0]), impovement,
#         chi2_data, chi2_signal,
#         chi2_data_bayes, chi2_signal_bayes, chi2_cross,
#         ))
# # ax1.plot(evidence/initial_evidence, 'b-')
# # ax1.plot(np.max(evidence, axis=0)/evidence[0], 'k-')
# # ax1.plot(posterior_variance/data_variance, 'k-')
# # ax1.plot((signal-posterior_mean)**2/data_variance, 'r-')
# # ax1.grid('both')
# # ax1.set_ylim(-.1, 2.1)

# ax2.plot(signal, 'r-', alpha=.5)
# ax2.plot(data, 'k-', alpha=.1)
# band = np.sqrt(data_variance)
# # ax2.fill_between(index, data-band, data+band, 'k', alpha=.1)
# # ax2.plot(posterior_mean, 'r-')
# ax2.plot(bayes, 'k-', alpha=1)
# ax2.plot(posterior_mean, 'y-', alpha=.5)
# band = 2*np.sqrt(posterior_variance)
# # ax2.fill_between(index, posterior_mean-band, posterior_mean+band,
#                  # 'r', alpha=.1)
# ax2.set_xlim(zoom_center-zoom_radius, zoom_center+zoom_radius)
# # ax2.grid('both')
# ax2.set_ylim(np.percentile(data, 1), np.percentile(data, 99.5))

# plt.show()


# %% Bye
print("... Paranoy@ Rulz!")

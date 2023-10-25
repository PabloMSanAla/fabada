#!/Users/pmsa/opt/anaconda3/bin/python
# -*- coding: utf-8 -*-

"""

FABADA script for denoising fits files from the command line 

Sanchez-Alarcon, P.M. and  Ascasibar, Y. 2022.
"Fully Adaptive Bayesian Algorithm for Data Analysis. FABADA"

"""
from fabada import fabada
import argparse
from astropy.io import fits
import os
import numpy as np


def make_parser():
    """Create an argument parser"""
    parser = argparse.ArgumentParser(description="Runs fabada to denoise image given.")

    parser.add_argument(
        "filename", type=str, help="Location of input .fits file (string)"
    )

    parser.add_argument(
        "noise", help="Standard deviation estimation of the image (float or string)"
    )

    parser.add_argument(
        "-out",
        type=str,
        help="Location to save filtered image. Supports .fits filenames (string)",
        default=None,
    )

    parser.add_argument(
        "-hdu", type=int, help="HDU of image, default 0 (integer)", default=0
    )

    parser.add_argument(
        "-noise_hdu",
        type=int,
        help="HDU of noise image, default 0 (integer)",
        default=0,
    )

    parser.add_argument(
        "-res", type=bool, help="Save Residuals, default False (boolean)", default=False
    )

    parser.add_argument(
        "-iter",
        type=int,
        help="Maximum number of iteration of fabada, default 3000 (integer)",
        default=3000,
    )

    parser.add_argument(
        "-verbose",
        type=bool,
        help="Verbose paramater for fabada, default False (boolean)",
        default=True,
    )

    return parser


# Parse arguments

p = make_parser().parse_args()

extension = p.filename.split(".")[-1]
name = os.path.basename(os.path.splitext(p.filename)[0])
path = os.path.dirname(p.filename)

if p.verbose:
    print("Starting smoothing with fabada in %s image..." % (name + "." + extension))

# Read fits file

image = fits.open(p.filename)[p.hdu].data
header = fits.open(p.filename)[p.hdu].header

# Read noise

try:
    noise = float(p.noise)
except:
    noise = str(p.noise)
    noise = fits.open(p.noise)[p.noise_hdu].data

nan_mask = np.isnan(image)
if nan_mask.any():
        image[nan_mask]=0
        noise[nan_mask]=1e-10

# Run fabada

fabada_estimation = fabada(image, noise**2, max_iter=p.iter, verbose=p.verbose)

if nan_mask.any(): fabada_estimation[nan_mask] = np.nan
# Save results

if not p.out:
    p.out = name + "_fabada." + extension

if p.verbose:
    print("Saving result in " + p.out)

header.append(("COMMENTS", "FABADA smooth estimation", ""), end=True)
hdu_new = fits.PrimaryHDU(fabada_estimation, header=header)
hdu_new.writeto(p.out, overwrite=True)


# Save residuals if want it

if p.res:
    residuals = name + "_residuals." + extension
    if p.verbose:
        print("Saving residuals in " + residuals)

    header.append(("COMMENTS", "FABADA smooth residuals", ""), end=True)
    hdu_new = fits.PrimaryHDU(image - fabada_estimation, header=header)
    hdu_new.writeto(residuals, overwrite=True)

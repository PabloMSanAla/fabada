"""
FABADA example of denosing an astronomical grey image based on

P.M. Sanchez-Alarcon, Y. Ascasibar, 2022
"Fully Adaptive Bayesian Algorithm for Data Analysis. FABADA"
"""

from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import os

try:
    import cv2
except:
    print(
        "OpenCV Library is not install. Try installing it with",
        "\n"
        + "pip install cv2"
        + "\n"
        + "or\n"
        + "conda install -c conda-forge opencv",
    )
try:
    from skimage.metrics import structural_similarity as ssim
except:
    print("skimage is not install. SSIM would not be displayed.")
from fabada import fabada, PSNR


def main():
    # IMPORTING IMAGE
    y = cv2.imread("bubble.png", 0)

    # ADDING RANDOM GAUSSIAN NOISE
    np.random.seed(12431)
    sig = 15  # Standard deviation of noise
    noise = np.random.normal(0, sig, y.shape)
    z = y + noise
    variance = sig ** 2

    # APPLY FABABA FOR RECOVER
    y_recover = fabada(z, variance)

    # SHOW RESULTS
    show_results(y, z, y_recover)


def show_results(y, z, y_recover, save_fig=True):
    # PLOTTING RESULTS WITH MATPLOTLIB
    plt.ion()
    fig, ax = plt.subplots(1, 3, figsize=(11, 4), sharex=True, sharey=True)
    vmin, vmax = np.nanpercentile(y, [5, 97])

    titles = ["Original Signal", "Noisy Measurements", "Recovered Signal"]
    images = [y, z, y_recover]

    for i in range(len(images)):

        ax[i].set_title(titles[i], fontsize=13)
        ax[i].imshow(images[i], vmin=vmin, vmax=vmax, cmap="gray", origin="lower")
        ax[i].axis("off")

        if i > 0:
            try:
                ax[i].text(
                    0.7,
                    0.02,
                    "({:2.2f}/{:1.2})".format(
                        PSNR(images[i], y, L=255), ssim(images[i], y, data_range=255)
                    ),
                    va="bottom",
                    ha="center",
                    transform=ax[i].transAxes,
                    fontsize=13,
                    c="white",
                )

            except:
                ax[i].text(
                    0.7,
                    0.02,
                    "PSNR={:2.2f}".format(PSNR(images[i], y, L=255)),
                    va="bottom",
                    ha="center",
                    transform=ax[i].transAxes,
                    fontsize=13,
                    c="white",
                )

    plt.subplots_adjust(
        top=0.97, bottom=0.0, left=0.01, right=0.99, wspace=0.0, hspace=0.00
    )

    if save_fig:
        save_path = os.path.join(os.getcwd(), "..", "src", "images")
        plt.savefig(
            os.path.join(
                save_path, "bubble_fabada_{:2.2f}dB.jpg".format(PSNR(z, y, L=255))
            ),
            dpi=200,
        )

    plt.show()
    plt.pause(10)


if __name__ == "__main__":
    main()

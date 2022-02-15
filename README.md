<div id="top"></div>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![GNU License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO
<br />
<div align="center">
  <a href="https://github.com/PabloMSanAla/fabada">
    <img src="fabada-logo.svg" alt="Logo" width="80" height="80">
  </a> -->

<h3 align="center">Fully Adaptive Bayesian Algorithm for Data Analysis</h3>
<h3 align="center">FABADA</h3>

  <p align="center">
    FABADA is a novel non-parametric noise reduction technique which arise from the point of view of Bayesian inference that iteratively evaluates possible smoothed models of the data, obtaining an estimation of the underlying signal that is statistically compatible with the noisy measurements.
    Iterations stop based on the evidence $E$ and the $\chi^2$ statistic of the last smooth model, and we compute the expected value of the signal as a weighted average of the smooth models.
    You can find the entire paper describing the new method in <a href="https://arxiv.org/abs/2201.05145">Sánchez-Alarcón, P & Ascasibar, Y. 2022</a>.
    <br />
    <a href="https://colab.research.google.com/github/PabloMSanAla/fabada/blob/master/Results/show_results.ipynb#scrollTo=o1iHY5aE5O2o"><strong>Explore the results »</strong></a>
    <br />
    <br />
    <a href="https://github.com/PabloMSanAla/fabada#usage">View Demo</a>
    ·
    <a href="https://github.com/PabloMSanAla/fabada/issues">Report Bug</a>
    ·
    <a href="https://github.com/PabloMSanAla/fabada/issues">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Method</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#results">Results</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#cite">Cite</a></li>
    <!-- <li><a href="#acknowledgments">Acknowledgments</a></li> -->
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Method

<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com) -->

This automatic method is focused in astronomical data, such as images (2D) or spectra (1D). Although, this doesn't mean it can be treat like a general noise reduction algorithm and can be use in any kind of two and one-dimensional data reproducing reliable results.
The only requisite of the input data is an estimation of its associated variance.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

We try to make the usage of FABADA as simple as possible. For that purpose, we have create a PyPI package to install FABADA.

### Prerequisites

The first requirement is to have a version of Python greater than 3.5.
Although PyPI install the prerequisites itself, FABADA has two dependecies.

- [Numpy](https://numpy.org/)
- [Scipy](https://www.scipy.org/)

### Installation

Using pip you can either install the last [relase](https://github.com/PabloMSanAla/fabada/releases) by

```sh
  pip install fabada
```

or you can install the latest version of the code as 

```sh
  git clone git@github.com:PabloMSanAla/fabada.git
  cd fabada
  pip install -e .
```


<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->

## Usage

## Astronomy .fits usage

For the Astronomical community we give another file, the fabadaCMD.py file that will help you run
fabada for a fits file from the command line. The requisites to use this file is to have installed the
astropy and argparse packages. The positional arguments of this file are:

- filename: Which is the string variable of the localization and name of the image fits file
- noise: Either a float number of the estimation of the variance of the image or a string for the containing fits file

In this example you will use fabada to denoise a [SDSS](https://www.sdss.org/) mosaic image of the [NGC2870](http://simbad.u-strasbg.fr/simbad/sim-basic?Ident=ngc2870&submit=SIMBAD+search) galaxy using the [fabadaCMD.py](https://github.com/PabloMSanAla/fabada/blob/master/examples/fabadaCMD.py) file. The image file [ngc2870_sloan_r.fits](https://github.com/PabloMSanAla/fabada/blob/master/examples/ngc2870_sloan_r.fits) is a mosaic image in the R band with an associate noise of roughly 0.001 counts, which is also located in the file [ngc2870_sloan_r_noise.fits](https://github.com/PabloMSanAla/fabada/blob/master/examples/ngc2870_sloan_r_noise.fits). This value of the noise is a simple estimation of the variance of the image but you can either give your own estimation or fits file of the error. To run fabada from the command line you only have to run

```bash
    python fabadaCMD.py ngc2870_sloan_r.fits ngc2870_sloan_r_noise.fits
```

and the result will be saved in a fits file.

If you want to see the other optional parameters you only have to run

```bash
    python fabadaCMD.py -h
```

in the shell and the optional parameters along a short description will be shown.

## General Python usage

Along with the package two examples are given.

- _fabada_demo_image.py_

In here we show how to use fabada for an astronomical grey image (two dimensional)
First of all we have to import our library previously install and some dependecies

```python
    from fabada import fabada
    import numpy as np
    from PIL import Image
```

Then we read the [bubble image](https://github.com/PabloMSanAla/fabada/blob/master/examples/bubble.png) borrowed from the [Hubble Space Telescope gallery](https://www.nasa.gov/mission_pages/hubble/multimedia/index.html). In our case we use the [Pillow](https://pypi.org/project/Pillow/) library for that. We also add some random Gaussian white noise using [numpy.random](https://numpy.org/doc/1.16/reference/routines.random.html).

```python
    # IMPORTING IMAGE
    y = np.array(Image.open("bubble.png").convert('L'))

    # ADDING RANDOM GAUSSIAN NOISE
    np.random.seed(12431)
    sig      = 15             # Standard deviation of noise
    noise    = np.random.normal(0, sig ,y.shape)
    z        = y + noise
    variance = sig**2
```

Once the noisy image is generated we can apply fabada to produce an estimation of the underlying image, which we only have to call fabada and give it the variance of the noisy image

```python
    y_recover = fabada(z,variance)
```

And its done :wink:

As easy as one line of code.

The results obtained running this example would be:

![Image Results][image_results]

The left, middle and right panel corresponds to the true signal, the noisy meassurents and the estimation of fabada respectively. There is also shown the Peak Signal to Noise Ratio (PSNR) in dB and the Structural Similarity Index Measure (SSIM) at the bottom of the middle and right panel (PSNR/SSIM).

- _fabada_demo_spectra.py_

In here we show how to use fabada for an astronomical spectrum (one dimensional), basically is the same as the example above since fabada is the same for one and two-dimensional data.
First of all, we have to import our library previously install and some dependecies

```python
    from fabada import fabada
    import pandas as pd
    import numpy as np
```

Then we read the interacting galaxy pair [Arp 256](http://simbad.u-strasbg.fr/simbad/sim-basic?Ident=arp256&submit=SIMBAD+search) spectra, taken from the [ASTROLIB PYSYNPHOT](https://github.com/spacetelescope/pysynphot) package which is store in [arp256.csv](https://github.com/PabloMSanAla/fabada/blob/master/examples/arp256.csv). Again we add some random Gaussian white noise

```python
    # IMPORTING SPECTRUM
    y = np.array(pd.read_csv('arp256.csv').flux)
    y = (y/y.max())*255  # Normalize to 255

    # ADDING RANDOM GAUSSIAN NOISE
    np.random.seed(12431)
    sig      = 10             # Standard deviation of noise
    noise    = np.random.normal(0, sig ,y.shape)
    z        = y + noise
    variance = sig**2
```

Once the noisy image is generated we can, again, apply fabada to produce an estimation of the underlying spectrum, which we only have to call fabada and give it the variance of the noisy image

```python
    y_recover = fabada(z,variance)
```

And done again :wink:

Which is exactly the same as for two dimensional data.

The results obtained running this example would be:

![Spectra Results][spectra_results]

The red, grey and black line represents the true signal, the noisy meassurents and the estimation of fabada respectively. There is also shown the Peak Signal to Noise Ratio (PSNR) in dB and the Structural Similarity Index Measure (SSIM) in the legend of the figure (PSNR/SSIM).

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- Results Paper -->

## Results

All the results of the paper of this algorithm can be found in the folder [results](https://github.com/PabloMSanAla/fabada/tree/master/Results) along with a jupyter notebook that allows to explore all of them through an interactive interface. You can run the jupyter notebook through Google Colab in this link --> [`Explore the results`](https://colab.research.google.com/github/PabloMSanAla/fabada/blob/master/Results/show_results.ipynb#scrollTo=o1iHY5aE5O2o).

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTRIBUTING -->

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- LICENSE -->

## License

Distributed under the GNU General Public License. See [`LICENSE.txt`](https://github.com/PabloMSanAla/fabada/blob/master/LICENSE) for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTACT -->

## Contact

Pablo M Sánchez Alarcón - pablom.sanala@gmail.com

Yago Ascasibar Sequeiros - yago.ascasibar@uam.es

Project Link: [https://github.com/PabloMSanAla/fabada](https://github.com/PabloMSanAla/fabada)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CITE -->

## Cite

Thank you for using FABADA.

Citations and acknowledgement are vital for the continued work on this kind of algorithms.

Please cite the following record if you used FABADA in any of your publications.

@ARTICLE{2022arXiv220105145S,<br />
author = {{Sanchez-Alarcon}, Pablo M and {Ascasibar Sequeiros}, Yago},<br />
title = "{Fully Adaptive Bayesian Algorithm for Data Analysis, FABADA}",<br />
journal = {arXiv e-prints},<br />
keywords = {Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Astrophysics of Galaxies, Astrophysics - Solar and Stellar Astrophysics, Computer Science - Computer Vision and Pattern Recognition, Physics - Data Analysis, Statistics and Probability},<br />
year = 2022,<br />
month = jan,<br />
eid = {arXiv:2201.05145},<br />
pages = {arXiv:2201.05145},<br />
archivePrefix = {arXiv},<br />
eprint = {2201.05145},<br />
primaryClass = {astro-ph.IM},<br />
adsurl = {https://ui.adsabs.harvard.edu/abs/2022arXiv220105145S}<br />
}

Sanchez-Alarcon, P. M. and Ascasibar Sequeiros, Y., “Fully Adaptive Bayesian Algorithm for Data Analysis, FABADA”, <i>arXiv e-prints</i>, 2022.

https://arxiv.org/abs/2201.05145

<p align="right">(<a href="#top">back to top</a>)</p>

Readme file taken from [Best README Template](https://github.com/othneildrew/Best-README-Template).

<!-- ACKNOWLEDGMENTS
## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#top">back to top</a>)</p> -->

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/PabloMSanAla/fabada.svg?style=plastic&logo=appveyor
[contributors-url]: https://github.com/PabloMSanAla/fabada/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/PabloMSanAla/fabada.svg?style=plastic&logo=appveyor
[forks-url]: https://github.com/PabloMSanAla/fabada/network/members
[stars-shield]: https://img.shields.io/github/stars/PabloMSanAla/fabada.svg?style=plastic&logo=appveyor
[stars-url]: https://github.com/PabloMSanAla/fabada/stargazers
[issues-shield]: https://img.shields.io/github/issues/PabloMSanAla/fabada.svg?style=plastic&logo=appveyor
[issues-url]: https://github.com/PabloMSanAla/fabada/issues
[license-shield]: https://img.shields.io/github/license/PabloMSanAla/fabada.svg?style=plastic&logo=appveyor
[license-url]: https://github.com/PabloMSanAla/fabada/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=plastic&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[image_results]: src/images/bubble_fabada_24.63dB.jpg
[spectra_results]: src/images/arp256_fabada_28.22dB.jpg

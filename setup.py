from setuptools import setup, find_packages



setup(
    name="fabada",
    version='0.2',
    description='Fully Adaptive Bayesian Algorithm for Data Analysis FABADA',
    long_description='FABADA is a novel non-parametric noise reduction technique which arise from the point of view of Bayesian inference that iteratively evaluates possible smoothed models of the data, obtaining an estimation of the underlying signal that is statistically compatible with the noisy measurements. Iterations stop based on the evidence $E$ and the $\chi^2$ statistic of the last smooth model, and we compute the expected value of the signal as a weighted average of the smooth models.\nYou can find the entire paper describing the new method in Sánchez-Alarcón, P & Ascasibar,Y. 2022.',
    long_description_content_type="text/markdown",
    include_package_data=True,
    author='Pablo M. Sánchez Alarcón',
    author_email='pablom.sanala@gmail.com',
    url = 'https://github.com/PabloMSanAla/fabada',
    download_url='https://github.com/PabloMSanAla/fabada/archive/refs/tags/v0.2.tar.gz',
    packages=['fabada'],
    python_requires='>=3.5',
    keywords=['Astronomy','Image Denoising','Bayesian'],
    install_requires=['numpy', 'scipy'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: Free for non-commercial use',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules']
)

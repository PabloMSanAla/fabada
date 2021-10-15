from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="fabada",
    version='0.0.1',
    description='Fully Adaptive Bayesiant Algorithm for Data Analysis FABADA',
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    author='Pablo M. Sánchez Alarcón',
    author_email='pablom.sanala@gmail.com',
    packages=['fabada'],
    python_requires='>=3.5',
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
        'Topic :: Software Development :: Libraries :: Python Modules'
        'Cite :: Sanchez-Alarcon,P.M; Ascasibar,Yago ; '+\
                r'\texit{Fully Adaptive Bayesian Algorithm for Data Analisys,FABADA}'+\
                r'2020, IEEE Transactions in Image [...]']
)
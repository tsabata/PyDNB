from setuptools import setup, find_packages  # Always prefer setuptools over distutils
from codecs import open  # To use a consistent encoding
from os import path
import pydnb

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pydnb',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # http://packaging.python.org/en/latest/tutorial.html#version
    version=pydnb.__version__,

    description='Implementation of dynamic naive Bayes (extension for hidden Markov model)',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/tsabata/PyDNB',

    # Author details
    author='Tomas Sabata',
    author_email='sabata.tomas@mail.com',

    # Choose your license
    license='MIT',


    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().

    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    # packages=['my_package','my_package.my_subpackage1','my_package.my_subpackage2'],

    # List run-time dependencies here.  These will be installed by pip when your
    # project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/technical.html#install-requires-vs-requirements-files
    install_requires=['numpy', 'scipy', 'pandas'],
)

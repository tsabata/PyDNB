from codecs import open  # To use a consistent encoding
import setuptools
import os
import re

with open("README.md", "r", encoding='utf-8') as f:
    long_description = f.read()


with open('pydnb/VERSION') as f:
    version_content = [line for line in f.readlines() if re.search(r'([\d.]+)',line)]

if len(version_content) != 1:
    raise RuntimeError('Invalid format of VERSION file.')


setuptools.setup(
    name='pydnb',
    version=version_content[0] + os.environ.get('PY_PKG_DEV_VERSION', ''),
    author='Tomas Sabata',
    author_email='sabata.tomas@mail.com',
    description="Implementation of dynamic naive Bayes classifier (extension of hidden Markov model)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tsabata/PyDNB",
    license='MIT',
    packages=setuptools.find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    classifiers=(
        "Programming Language :: Python :: 3",
    ),
    python_requires='>=3.4',
    install_requires=['numpy',
                      'scipy',
                      'pandas'],
    include_package_data=True,
)
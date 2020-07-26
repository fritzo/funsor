# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import sys

from setuptools import find_packages, setup

# READ README.md for long description on PyPi.
# This requires uploading via twine, e.g.:
# $ python setup.py sdist bdist_wheel
# $ twine upload --repository-url https://test.pypi.org/legacy/ dist/*  # test version
# $ twine upload dist/*
try:
    long_description = open('README.md', encoding='utf-8').read()
except Exception as e:
    sys.stderr.write('Failed to convert README.md to rst:\n  {}\n'.format(e))
    sys.stderr.flush()
    long_description = ''

# Remove badges since they will always be obsolete.
# This assumes the first 4 lines contain badge info.
long_description = '\n'.join(line for line in long_description.split('\n')[4:])

setup(
    name='funsor',
    version='0.1.2',
    description='A tensor-like library for functions and distributions',
    packages=find_packages(include=['funsor', 'funsor.*']),
    url='https://github.com/pyro-ppl/funsor',
    project_urls={
        "Documentation": "https://funsor.pyro.ai",
    },
    author='Uber AI Labs',
    author_email='fritzo@uber.com',
    python_requires=">=3.5",
    install_requires=[
        'makefun',
        'multipledispatch',
        'numpy==1.16.2',
        'opt_einsum==3.2.1',
    ],
    extras_require={
        'torch': [
            'pyro-ppl>=0.5',
            'torch>=1.3.0',
        ],
        'jax': [
            'jax>=0.1.65',
            'jaxlib>=0.1.45',
            'numpyro @ git+https://github.com/pyro-ppl/numpyro.git@e3accb5c71c8e96991abe9dd1f823e4435a40618#egg=numpyro'
        ],
        'test': [
            'flake8',
            'pandas',
            'pyro-api>=0.1.2',
            'pytest==4.3.1',
            'pytest-xdist==1.27.0',
            'pillow-simd',
            'scipy==1.5.1',
            'torchvision',
        ],
        'dev': [
            'flake8',
            'isort',
            'pandas',
            'pytest==4.3.1',
            'pytest-xdist==1.27.0',
            'scipy',
            'sphinx>=2.0',
            'sphinx_rtd_theme',
            'pillow-simd',
            'torchvision',
        ],
    },
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='probabilistic machine learning bayesian statistics pytorch',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache License 2.0',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3.5',
    ],
)

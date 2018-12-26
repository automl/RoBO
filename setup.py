import os
import sys
from setuptools import setup, find_packages
here = os.path.abspath(os.path.dirname(__file__))

requires = [
    'george',
    'emcee',
    'pyrfr',
    'pybnn',
    'cython',
    'scipy >= 0.12',
    'numpy >= 1.7'
    ]

setup(name='RoBO',
      version='0.3.1',
      description='a python framework for Bayesian optimization',
      author='Aaron Klein',
      author_email='kleinaa@cs.uni-freiburg.de',
      url='http://automl.github.io/RoBO/',
      keywords='Bayesian Optimization',
      packages=find_packages(),
      license='LICENSE.txt',
      test_suite='robo',
      install_requires=requires)

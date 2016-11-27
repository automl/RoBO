import os
import sys
from setuptools import setup, find_packages
here = os.path.abspath(os.path.dirname(__file__))

requires = [
    'DIRECT',
    'george',
    'emcee',
    'scipy >= 0.12',
    'numpy >= 1.7'
    ]
dependency_links = [
      'https://github.com/sfalkner/george/tarball/master#egg=gerorge'
      ]

optional_dependencies = [
    'cython',
    'rf',
    'cma',
    'matplotlib',
    'theano',
    'lasagne',
    'sgmcmc',
    'pymatbridge'
    ]

opt_dependency_links = {
    'rf': 'https://bitbucket.org/aadfreiburg/random_forest_run/get/master.zip#egg=rf'
}

user_provided_depend = sys.argv[2:len(sys.argv)]
for i in user_provided_depend:
    dependency_name = i[2:len(i)]
    if dependency_name in optional_dependencies:
        if dependency_name in opt_dependency_links:
            dependency_links.append(opt_dependency_links[dependency_name])
        requires.append(dependency_name)
        sys.argv.remove(i)

setup(name='RoBO',
      version='0.2.0',
      description='Framework for Bayesian optimization',
      author='Aaron Klein',
      author_email='kleinaa@cs.uni-freiburg.de',
      url='http://automl.github.io/RoBO/',
      keywords='Bayesian Optimization',
      packages=find_packages(),
      license='LICENSE.txt',
      test_suite='robo',
      install_requires=requires,
      dependency_links=dependency_links)

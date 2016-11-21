import os
import sys
from setuptools import setup, find_packages
here = os.path.abspath(os.path.dirname(__file__))
#README = open(os.path.join(here, 'README.txt')).read()
#CHANGES = open(os.path.join(here, 'CHANGES.txt')).read()
requires = [
	'DIRECT',
	'george',
	'emcee==2.1.0',
	'scipy >= 0.12',
	'numpy >= 1.7'
    ]
dependency_links = [
      'https://github.com/sfalkner/george/tarball/master#egg=gerorge'
      ]

optional_dependancies = [
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
	'rf':'https://bitbucket.org/aadfreiburg/random_forest_run/get/master.zip#egg=rf'
}

user_provided_depend  = sys.argv[2:len(sys.argv)]
for i in user_provided_depend:
	dependancy_name = i[2:len(i)]
	if dependancy_name in optional_dependancies:
		if dependancy_name in opt_dependency_links:
			dependency_links.append(opt_dependency_links[dependancy_name])
		requires.append(dependancy_name)
		sys.argv.remove(i)

setup(name='robo',
      version='0.1',
      description='Framework for Bayesian optimization',
      long_description='',
      classifiers=[
        "Programming Language :: Python",
        ],
      author='Aaron Klein',
      author_email='kleinaa@cs.uni-freiburg.de',
      url='http://automl.github.io/RoBO/',
      keywords='Bayesian Optimization',
      packages=find_packages(),
      include_package_data=True,
      test_suite='robo',
      install_requires= requires,
      dependency_links = dependency_links,
      entry_points=dict(
      		console_scripts = [
		'robo_visualize = robo.scripts.visualize_sh:main',
		'robo_examples = robo.scripts.examples:main'
	  ]
      )
	)

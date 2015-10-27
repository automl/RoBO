import os

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
#README = open(os.path.join(here, 'README.txt')).read()
#CHANGES = open(os.path.join(here, 'CHANGES.txt')).read()

requires = [
	'DIRECT',
	'emcee==2.1.0',
	'numpy >= 1.7',
	'scipy >= 0.12',
	'matplotlib >= 1.3',
	'cma'
    ]


setup(name='robo',
      version='0.1',
      description='',
      long_description='',
      classifiers=[
        "Programming Language :: Python",
        ],
      author='',
      author_email='',
      url='',
      keywords='',
      packages=find_packages(),
      include_package_data=True,
      test_suite='robo',
      install_requires=requires,
      entry_points=dict(
      		console_scripts = [
		'robo_visualize = robo.scripts.visualize_sh:main',
		'robo_examples = robo.scripts.examples:main'
	  ]
      )
	)

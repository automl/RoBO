import os

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
#README = open(os.path.join(here, 'README.txt')).read()
#CHANGES = open(os.path.join(here, 'CHANGES.txt')).read()

requires = [
	'GPy',
	'DIRECT'
    ]

setup(name='robo',
      version='0.0',
      description='',
      long_description='',
      classifiers=[
        "Programming Language :: Python",
        "Framework :: Pylons",
        "Topic :: Internet :: WWW/HTTP",
        "Topic :: Internet :: WWW/HTTP :: WSGI :: Application",
        ],
      author='joel',
      author_email='joel@renumi.de',
      url='http://renumi.de',
      keywords='',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      test_suite='robo',
      install_requires=requires,
      entry_points="""\
      """,
      )

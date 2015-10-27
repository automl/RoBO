Installation
============

.. role:: bash(code)
    :language: bash

Dependencies
------------

 - numpy >= 1.7
 - scipy >= 0.12
 - GPy==0.6.1
 - emcee==2.1.0
 - matplolib >= 1.3
 - direct
 - cma
 
How to install RoBO
-------------------

You can install RoBO by cloning the repository and executing the setup script:

	:bash:`git clone https://github.com/automl/RoBO`

	:bash:`cd RoBO/`

	:bash:`for req in $(cat requirements.txt); do pip install $req; done`

	:bash:`python setup.py install`
 
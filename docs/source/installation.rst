.. raw:: html

    <a href="https://github.com/automl/RoBO"><img style="position: fixed; top: 50px; right: 0; border: 0;" src="https://camo.githubusercontent.com/365986a132ccd6a44c23a9169022c0b5c890c387/68747470733a2f2f73332e616d617a6f6e6177732e636f6d2f6769746875622f726962626f6e732f666f726b6d655f72696768745f7265645f6161303030302e706e67" alt="Fork me on GitHub" data-canonical-src="https://s3.amazonaws.com/github/ribbons/forkme_right_red_aa0000.png"></a>


Installation
============

Dependencies
------------
RoBO needs the following dependencies to be installed for it’s core functionality.

* scipy >= 0.12
* numpy >= 1.7
* direct
* cma
* george
* emcee

Additionally RoBO has some optional dependencies that are only needed for specific modules:

* cython
* `pyrfr <https://bitbucket.org/aadfreiburg/random_forest_run/>`_
* theano
* matplotlib
* lasagne
* `sgmcmc <https://github.com/stokasto/sgmcmc>`_
* `hpolib2 <https://github.com/automl/HPOlib2>`_

**Note**: RoBO works only with Python3. Python2 is not supported anymore.

-------------------
Manual Installation
-------------------

RoBO uses the Gaussian processes library `george <https://github.com/dfm/george>`_  and the random forests library `pyrfr <https://github.com/automl/random_forest_run>`_. In order to use this library make sure the libeigen and swig are installed:

.. code:: bash

     sudo apt-get install libeigen3-dev swig gfortran

Download RoBO and then change into the new directory:

.. code:: bash

	git clone https://github.com/automl/RoBO
	cd RoBO/

Before you install RoBO you have to install the required dependencies. We use a for loop because we want to preserve the installation order of the list of dependencies in the requirments file.

.. code:: bash

     for req in $(cat requirements.txt); do pip install $req; done

This will install the basis requirements that you need to run RoBO's core functionality. If you want to make
use of the full functionality (for instance Bohamiann, Fabolas, ...) you can install all necessary dependencies
by:

.. code:: bash
     
     for req in $(cat all_requirements.txt); do pip install $req; done

**Note**: This may take a while to install all dependencies.

Finally you can install RoBO by:

.. code:: bash

     python setup.py install

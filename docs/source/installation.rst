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

**Note**: RoBO works only with Python3. Python2 is not support anymore.

-------------------
Manual Installation
-------------------

RoBO uses the Gaussian processes library `george <https://github.com/dfm/george>`_ . In order to use this library make sure the libeigen is installed:

.. code:: bash

     sudo apt-get install libeigen3-dev

then change into the new directory:

.. code:: bash

	cd RoBO/

Before you install RoBO you have to install the required dependencies

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

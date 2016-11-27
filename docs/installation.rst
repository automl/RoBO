.. _installation

============
Installation
============


------------
Dependencies
------------

RoBO needs the following dependencies to be installed for it's core functionality.

* scipy >= 0.12
* numpy >= 1.7
* DIRECT
* george
* emcee

Additionally it has some optional dependencies that are only needed for specific modules:

* cython
* `pyrfr <https://bitbucket.org/aadfreiburg/random_forest_run/>`_
* theano
* cma
* matplotlib
* lasagne
* `sgmcmc <https://github.com/stokasto/sgmcmc>`_
* `hpolib2 <https://github.com/automl/HPOlib2>`_


**Note**: RoBO requires python 3 and does not support python 2 anymore.

-------------------
Manual Installation
-------------------

To install RoBO first clone the repository:

.. code:: bash

    git clone https://github.com/automl/RoBO.git

then change into the new directory:

.. code:: bash

    cd RoBO/


and install it:

.. code:: bash

    python setup.py install


This will install RoBO's core functionality.
Again ff you want to use additional modules such as for instance
Bohamiann you have to install some extra optional dependencies (see above).
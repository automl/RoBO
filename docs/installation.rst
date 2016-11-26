.. _installation

============
Installation
============

RoBO has a few dependencies that need to be installed for it's core functionality. Additionally it has some optional dependencies
that are only needed for specific modules (e.g. Theano).
RoBO requires python 3 and does not support python 2 anymore.

To install RoBO first clone the repository:

.. code:: bash

    git clone https://github.com/automl/RoBO.git

then change into the new directory:

.. code:: bash

    cd RoBO/


and install it:

.. code:: bash

    python setup.py install


This will install RoBO's core functionality. If you want to use additional modules such as for instantce
Bohamiann you have to install some extra optional dependencies (e.g. Theano, Lasagne).

You can install optional dependencies by:


.. code:: bash

    python setup.py install

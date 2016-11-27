.. _tutorials

=========
Tutorials
=========

The following tutorials will show you how to use the different Bayesian optimization methods
that are implemented in RoBO.

The code for all the tutorials and more examples can be found in the ``examples`` folder.

---------------------
Bayesian Optimization
---------------------

This tutorial will show you how to use vanilla Bayesian optimization with Gaussian processes and
different acquisition functions.

The first thing we have to do is to import numpy (for the objective function) and
the bayesian_optimization function

.. code-block:: python

  import numpy as np

  from robo.fmin import bayesian_optimization


To use RoBO we have to define a function that symbolizes the objective function we want to minimize.
The objective function gets an d-dimensional vector x and returns the corresponding scalar target value.

.. code-block:: python

    def objective_function(x):
        y = np.sin(3 * x[0]) * 4 * (x[0] - 1) * (x[0] + 2)
        return y


Before we can apply Bayesian optimization we have to define the lower and upper bound of our input
search space first.
In this case we have just a one dimensional optimization problem.

.. code-block:: python

    lower = np.array([0])
    upper = np.array([6])


Now we have everything we need and can now run Bayesian optimization for 50 iterations:

.. code-block:: python

    results = bayesian_optimization(objective_function, lower, upper, num_iterations=50)


At the end we get a dictionary back with all the results and some additional meta information

.. code-block:: python

    print(results["x_opt"])


By default RoBO uses Gaussian processes (with MCMC sampling to obtain the GP#s hyperparameters) and logarithmic
expected improvement as acquisition function.
If you would like to use a different acquisition function such as for instance the lower confidence bound
you can simple :

.. code-block:: python

    results = bayesian_optimization(objective_function, lower, upper, acquisition_func='lcb')

See the API documentation for different possible choices of acquisition functions.

If you want to have a deeper look what RoBO is doing under the hood you can activate RoBO's logging
mechanism by adding the following two lines on top of your python script:

.. code-block:: python

    import logging
    logging.basicConfig(level=logging.INFO)


---------
Bohamiann
---------

RoBO offers an simple interface for Bayesian Optimization with Hamiltonian Monte Carlo Artificial Neural Networks
(BOHAMIANN) which was introduced by Sprigenberg et al.

If you want to use Bohamiann make sure that you have Lasagne and Theano installed:

.. code-block:: python

    pip install Lasagne
    pip install theano


and that the `sgmcmc package <https://github.com/stokasto/sgmcmc>`_ is in your PYTHONPATH:

The interface to Bohamiann is exactly the same as for the GP based Bayesian optimization interface:

.. code-block:: python

    from robo.fmin import bohamiann

    results = bohamiann(objective_function, lower, upper, num_iterations=50)

Again this will return a dictionary with the results and some meta information.

@inproceedings{springenberg-nips2016,
       booktitle = {Advances in Neural Information Processing Systems 29},
       month = {December},
       title = {Bayesian optimization with robust Bayesian neural networks},
       author = {J. T. Springenberg and A. Klein and S.Falkner and F. Hutter},
       year = {2016}
}

-------
Fabolas
-------

The idea of Fabolas (Klein et al.) is to take the training data set size as an additional input into account that
can be freely chosen during the optimization procedure. However the goal is still to find
the best configuration for the full training dataset.

By additionally modelling the cost of training single configurations, Fabolas uses the information gain per unit
cost to pick and evaluate configurations on small subset of the training data that give the most information
about the global minimum on the full dataset.

The objective function gets besides a configuration also the training dataset size as input. After training
the configuration on a subset of the training data it returns the validation error on the full
validation data set as well as the time it took to train this configuration.

.. code-block:: python

    from robo.fmin import fabolas

    def objective_function(x, s):
            # Train your algorithm here with x on the dataset subset with length s
            # Estimate the validation error and the cost on the validation data set
            return validation_error, cost

Additionally you have to define the bounds of the input space for the configurations and the minimum and
maximum data set size.

.. code-block:: python

    lower = np.array([-10, -10])
    upper = np.array([10, 10])
    s_min = 100
    s_max = 50000

Then you can call Fabolas by:

.. code-block:: python

    res = fabolas(objective_function,
                  lower=lower,
                  upper=upper,
                  s_min=s_min,
                  s_max=s_max,
                  num_iterations=100)

You can find a full example for training a support vector machine on MNIST here

@article{klein-corr16,
 author    = {A. Klein and S. Falkner and S. Bartels and P. Hennig and F. Hutter},
 title     = {Fast Bayesian Optimization of Machine Learning Hyperparameters on Large Datasets},
 journal = corr,
 llvolume    = {abs/1605.07079},
 lurl = {http://arxiv.org/abs/1605.07079},
 year      = {2016}
}
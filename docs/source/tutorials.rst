Tutorials
=========
The following tutorials will help you to get familiar with RoBO.
You can find the code for all the tutorials and additional examples
in the ``examples`` folder.


1. Blackbox function optimization with RoBO

2. Bohamiann

3. Fabolas

4. Multi-Task Bayesian optimization

5. RoBO on HPOLIB2 benchmarks

6. Fitting a Bayesian neural network

Blackbox function optimization with RoBO
----------------------------------------

This tutorial will show you how to use standard Bayesian optimization with Gaussian processes and different acquisition functions to find the global minimizer of your (python) function. Note that RoBO so far only support continuous input space and it is not able to handle multi-objective functions.


The first thing we have to do is to import numpy (for the objective function) and the bayesian_optimization interface

    .. code-block:: python

	import numpy as np

	from robo.fmin import bayesian_optimization

To use RoBO we have to define a function that symbolizes the objective function we want to minimize. Interface of objective function is fairly simple, it gets an d-dimensional vector x and returns the corresponding scalar function value.

    .. code-block:: python

	def objective_function(x):
    		y = np.sin(3 * x[0]) * 4 * (x[0] - 1) * (x[0] + 2)
    		return y

Before we run Bayesian optimization we first have to define the lower and upper bound of our input search space. In this case our search space contains only one dimension, but Bayesian optimization is not restricted to that and normally works find up to 10 (continuous) dimensions.

    .. code-block:: python
	
	lower = np.array([0])
	upper = np.array([6])

Now we have everything we need and can now run Bayesian optimization for 50 iterations:

    .. code-block:: python

	results = bayesian_optimization(objective_function, lower, upper, num_iterations=50)

At the end we get a dictionary back with contains the following entries:

* “x_opt” : the best found data point
* “f_opt” : the corresponding function value
* “incumbents”: the incumbent (best found value) after each iteration
* “incumbent_value”: the function values of the incumbents
* “runtime”: the runtime in seconds after each iteration
* “runtime”: the optimization overhead (i.e. time data we do not spend to evaluate the function) of each iteration
* “X”: all data points that have been evaluated
* “y”: the corresponding function evaluations

By default RoBO uses Gaussian processes (with MCMC sampling to obtain the GP’s hyperparameters) and logarithmic expected improvement as acquisition function. If you would like to use a different acquisition function such as for instance the lower confidence bound you can simple :

    .. code-block:: python

	results = bayesian_optimization(objective_function, lower, upper, acquisition_func='lcb')

See the API documentation for different possible choices of acquisition functions.

If you want to have a deeper look what RoBO is doing under the hood you can activate RoBO’s logging mechanism by adding the following two lines on top of your python script:

    .. code-block:: python

	import logging
	logging.basicConfig(level=logging.INFO)

Besides standard Bayesian optimization, RoBO also contains an interface for plain random search and entropy search by Hennig et. al. Both methods follow the exact same interface.

    .. code-block:: python

	from robo.fmin import entropy_search
	from robo.fmin import random_search

	results = entropy_search(objective_function, lower, upper)
	results = random_search(objective_function, lower, upper)

Bohamiann
---------

RoBO offers an simple interface for Bayesian Optimization with Hamiltonian Monte Carlo Artificial Neural Networks (BOHAMIANN) which was introduced by Sprigenberg et al.

If you want to use Bohamiann make sure that you have Lasagne and Theano installed:

    .. code-block:: bash

	pip install Lasagne
	pip install theano



and that the `sgmcmc package <https://github.com/stokasto/sgmcmc>`_ is in your PYTHONPATH:

The interface to Bohamiann is exactly the same as for the GP based Bayesian optimization interface:

    .. code-block:: python

	from robo.fmin import bohamiann

	results = bohamiann(objective_function, lower, upper, num_iterations=50)

This will return a dictionary with the same meta information as described above.

@inproceedings{springenberg-nips2016, booktitle = {Advances in Neural Information Processing Systems 29}, month = {December}, title = {Bayesian optimization with robust Bayesian neural networks}, author = {J. T. Springenberg and A. Klein and S.Falkner and F. Hutter}, year = {2016} }

Fabolas
-------

The idea of Fabolas (Klein et al.) is to take the training data set size as an additional input into account that can be freely chosen during the optimization procedure but is fixed afterwards. The idea is to speed up the optimization by evaluating single configurations only on much cheaper subset and to extrapolate their performance on the full dataset.

By additionally modelling the cost of training single configurations, Fabolas uses the information gain per unit cost to pick and evaluate configurations on small subset of the training data that give the most information about the global minimum on the full dataset.

Now the objective function gets besides a configuration also the training dataset size as an additional input. After training the configuration on a subset of the training data it returns the validation error on the full validation data set as well as the time it took to train this configuration.

    .. code-block:: python

	from robo.fmin import fabolas

	def objective_function(x, s):
	    # Train your algorithm here with x on the dataset subset with length s
	    # Estimate the validation error and the cost on the validation data set
	    return validation_error, cost

Additionally you have to define the bounds of the input space for the configurations and the minimum and maximum data set size.

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


You can find a full example for training a support vector machine on MNIST `here <https://github.com/automl/RoBO/blob/master/examples/example_fabolas.py>`_

@article{klein-corr16, author = {A. Klein and S. Falkner and S. Bartels and P. Hennig and F. Hutter}, title = {Fast Bayesian Optimization of Machine Learning Hyperparameters on Large Datasets}, journal = corr, llvolume = {abs/1605.07079}, lurl = {http://arxiv.org/abs/1605.07079}, year = {2016} }

RoBO on HPOLIB2 benchmarks
--------------------------

`HPOlib2 <https://github.com/automl/HPOlib2>`_ contains a set of benchmarks with an unified interface for hyperparameter optimization of machine learning algorithms. In the following example we want to assume the often used synthetic function branin. Make sure that you installed HPOlib2.

First we load the benchmark and get the bound of the configuration space

    .. code-block:: python

	from hpolib.benchmarks.synthetic_functions import Branin
	f = Branin()
	info = f.get_meta_information()
	bounds = np.array(info['bounds'])

Than we can simply run RoBO by:

    .. code-block:: python

	results = bayesian_optimization(f, bounds[:, 0], bounds[:, 1], num_iterations=50)

HPOlib2 allows to evaluate single configurations only subsets of the data which allows us to use Fabolas or MTBO. If want to use Fabolas to optimize let’s say a support vector machine on MNIST we first have to wrap the HPOlib2 benchmarks class in order to pass the correct ration of the dataset size:

    .. code-block:: python

	from hpolib.benchmarks.ml.svm_benchmark import SvmOnMnist

	f = SvmOnMnist()

	def objective(x, s):
	    dataset_fraction = s / s_max

	    res = f.objective_function(x, dataset_fraction=dataset_fraction)
	    return res["function_value"], res["cost"]

Than we can run Fabolas simply by:

    .. code-block:: python

	info = f.get_meta_information()
	bounds = np.array(info['bounds'])
	lower = bounds[:, 0]
	upper = bounds[:, 1]

	results = fabolas(objective_function=objective, lower=lower, upper=upper,
		          s_min=100, s_max=s_max, n_init=10, num_iterations=80, n_hypers=20, subsets=[64., 32, 16, 8])

Fitting a Bayesian neural network
---------------------------------

The following tutorial we will see how we can train a Bayesian neural networks with stochastic MCMC sampling on our dataset. Note all models in RoBO implement the same interface and you can easily replace the Bayesian neural network by another model (Gaussian processes, Random Forest, …).

Assume we collect some data point of a sinc function:

    .. code-block:: python

	import matplotlib.pyplot as plt
	import numpy as np

	from robo.models.bnn import BayesianNeuralNetwork
	from robo.initial_design.init_random_uniform import init_random_uniform


	def f(x):
	    return np.sinc(x * 10 - 5).sum(axis=1)

	X = init_random_uniform(np.zeros(1), np.ones(1), 20, rng)
	y = f(X)

We can now create and train a neural network by:

    .. code-block:: python

	model = BayesianNeuralNetwork(sampling_method="sghmc",
		                      l_rate=np.sqrt(1e-4),
		                      mdecay=0.05,
		                      burn_in=3000,
		                      n_iters=50000,
		                      precondition=True,
		                      normalize_input=True,
		                      normalize_output=True)
	model.train(X, y)

After training we can use our model to predict the mean and variance for arbitrary test points:

    .. code-block:: python

	x = np.linspace(0, 1, 100)[:, None]
	mean_pred, var_pred = model.predict(x)






















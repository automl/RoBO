.. RoBo documentation master file, created by
   sphinx-quickstart on Mon Feb  2 15:56:53 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Robust Bayesian Optimization
============================




Dependencies
============

 - numpy >= 1.7
 - scipy >= 0.12
 - GPy==0.6.0
 - emcee==2.1.0
 - matplolib >= 1.3
 
Basic Usage
===========

RoBO is a flexible framework for Bayesian optimization. In a nutshell we can distinguish between different components 
that are necessary for BO, i.e an acquisition function, a model, and a method to optimize the acquisition function. RoBO treats all of those components as modules,
which allows us to easily change and add new methods.
 


Defining an objective function
------------------------------

RoBo can optimize any function :math:`X \rightarrow Y` with X as an :math:`N\times D` numpy array and Y as an :math:`N\times 1` numpy array. Thereby :math:`N` is the number of points you want to 
evaluate at and :math:`D` is the dimension of the input X. An example objective function could look like this:

.. code-block:: python

    import numpy as np
    def objective_function(x):
        return  np.sin(3*x) * 4*(x-1)* (x+2)
	    
Furthermore, we also have to specify the bounds of the objective function is defined:

.. code-block:: python
   
    X_lower = np.array([0])
    X_upper = np.array([6])
    dims = 1


Building a model 
----------------

The first step to optimize this objective function is to define a model that captures the current believe of potential functions. The probably most used method in 
Bayesian optimization for modeling the objective function are Gaussian processes. RoBO uses the well-known `GPy http://sheffieldml.github.io/GPy/`_ library as implementation for Gaussian processes. The following code snippet
shows how to use a GPy model via RoBO:

.. code-block:: python

   import GPy
   from robo.models.GPyModel import GPyModel
   
   kernel = GPy.kern.Matern52(input_dim=dims)
   model = GPyModel(kernel, optimize=True, noise_variance = 1e-4, num_restarts=10)

RoBO offers a wrapper interface GPyModel to access the Gaussian processes in GPy. We have to specify a kernel from GPy library as covariance function when we
initialize the model. For further details on those kernels visit `GPy http://sheffieldml.github.io/GPy/`_. We can either use fix kernel hyperparameter or optimize them by optimizing
the marginal likelihood. This is achieved by setting the optimize flag to True.

   
Creating the Acquisition Function
---------------------------------

After we defined a model we can define an acquisition function as a surrogate function that is used to pick the next point to evaluate. RoBO offers the following acquisition
functions in the acquisition package:
 .. toctree::
   :maxdepth: 1

   acquisition_func


In order to use an acquisition function (in this case Expected Improvement) you have to pass it the models as well as the bounds of the input space:


.. code-block:: python
	
    from robo.acquisition.EI import EI
    from robo.recommendation.incumbent import compute_incumbent
    acquisition_func = EI(model, X_upper=X_upper, X_lower=X_lower, compute_incumbent=compute_incumbent, par=0.1)


Expected Improvement as well as Probability of Improvement need as additional input the current best configuration (i.e. incumbent). There are different ways to determine 
the incumbent. You can easily plug in any method by giving Expected Improvement a function handle (via compute_incumbent). This function is supposed to return a
configuration and expects the model as input. 

Maximizing the acquisition function:
------------------------------------

The last component is the maximizer which will be used to optimize the acquisition function in order to get a new configuration to evaluate. RoBO offers different ways to
optimize the acquisition functions such as:

 - grid search
 - DIRECT
 - CMA-ES
 - stochastic local search
 

Here we will use a simple grid search to determine the configuration with the highest acquisition value:

.. code-block:: python

    from robo.maximizers.maximize import grid_search
    maximizer = grid_search
    
Implementing a main loop
------------------------

Now we have all the ingredients to optimize our objective function. We can put all the above described components in the BayesianOptimization class

.. code-block:: python

   from robo import BayesianOptimization

   bo = BayesianOptimization(acquisition_fkt=acquisition_func,
	                          model=model,
	                          maximize_fkt=maximizer,
	                          X_lower=X_lower,
	                          X_upper=X_upper,
	                          dims=dims,
	                          objective_fkt=objective_function)

Afterwards we can run it by:

.. code-block:: python
	
	bo.run(num_iterations=10)







    
Putting it all together:
------------------------

In the one dimensional case you can easily plot all the methods used:

.. code-block:: python

    import GPy
    import matplotlib; matplotlib.use('GTKAgg')
    import matplotlib.pyplot as plt;
    import numpy as np
    import random

    from robo.models.GPyModel import GPyModel
    from robo.acquisition.EI import EI
    from robo.maximizers.maximize import stochastic_local_search
    from robo.recommendation.incumbent import compute_incumbent

    def objective_function(x):
        return  np.sin(3*x) * 4*(x-1)* (x+2)

    X_lower = np.array([0])
    X_upper = np.array([6])
    dims = 1
    maximizer = stochastic_local_search

    kernel = GPy.kern.Matern52(input_dim=dims)
    model = GPyModel(kernel, optimize=True, noise_variance = 1e-4, num_restarts=10)
    acquisition_func = EI(model, X_upper=X_upper, X_lower=X_lower, compute_incumbent=compute_incumbent, par=0.1) #par is the minimum improvement


    X = np.empty((1, dims))
    for i in xrange(dims):
        X[0,i] = random.random() * (X_upper[i] - X_lower[i]) + X_lower[i];
        x = np.array(X)

    Y = objective_function(X)
    for i in xrange(10):
        model.train(X, Y)
        acquisition_func.update(model)
        new_x = maximizer(acquisition_func, X_lower, X_upper)
        new_y = objective_function(np.array(new_x))
        X = np.append(X, new_x, axis=0)
        Y = np.append(Y, new_y, axis=0)
        
        #
        # plot it
        #
        fig = plt.figure()
        ax1 =  fig.add_subplot(1, 1, 1)
        plotting_range = np.linspace(X_lower[0], X_upper[0], num=1000)
        ax1.plot(plotting_range, objective_function(plotting_range[:, np.newaxis]), color='b', linestyle="--")
        _min_y1, _max_y1 = ax1.get_ylim()
        model.visualize(ax1, X_lower[0], X_upper[0])
        _min_y2, _max_y2 = ax1.get_ylim()
        ax1.set_ylim(min(_min_y1, _min_y2), max(_max_y1, _max_y2))
        mu, var = model.predict(new_x)
        ax1.plot(new_x[0], mu[0], "r." , markeredgewidth=5.0)
        ax2 = acquisition_func.plot(fig, X_lower[0], X_upper[0], plot_attr={"color":"red"}, resolution=1000)

    plt.show(block=True)
       
   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. RoBo documentation master file, created by
   sphinx-quickstart on Mon Feb  2 15:56:53 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Dependencies
============

 - numpy >= 1.7
 - scipy >= 0.12
 - GPy==0.6.0
 - emcee==2.1.0
 - matplolib >= 1.3
 
Basic Usage
===========

Defining an objective function
------------------------------

RoBo can optimize any function that gets an :math:`N\times D` numpy array and returns an :math:`N\times 1` numpy array, where :math:`N` is the number of points you want to 
evaluate at and :math:`D` is the dimension of X. So perhaps it would end up with something similar to this:

.. code-block:: python

    import numpy as np
    def objective_function(x):
        return  np.sin(3*x) * 4*(x-1)* (x+2)
	    
You also have to define the bounds where your objective function can be evaluated at.

.. code-block:: python
   
    X_lower = np.array([0])
    X_upper = np.array([6])
    dims = 1

Building a model 
----------------

GPyModel is the only model implemented so far, and is a wrapper around the GPy library. You have to specify a kernel to initialize it. For further details visit `GPy homepage`_. 
You can either define the model hyperparameters of the kernel by yourself or optimize it with respect to the marginal likelihood by setting the optimize flag to True. Also the 
noise variance can be set to a constant value and won't be optimized, except it is None.

.. _GPy homepage: http://sheffieldml.github.io/GPy/

.. code-block:: python

   import GPy
   from robo.models import GPyModel
   
   kernel = GPy.kern.Matern52(input_dim=dims)
   model = GPyModel(kernel, optimize=True, noise_variance = 1e-4, num_restarts=10)
   
Creating the Acquisition Function
---------------------------------

An acquisition function is a utility function that has high values at points where it is worth to evaluate next.

.. code-block:: python
	
    from robo.acquisition import EI
    acquisition_func = EI(model, X_upper= X_upper, X_lower=X_lower, par =0.1) #par is the minimum improvement

Maximizing the acquisition function:
------------------------------------

Choose a maximizer to get the best value to evaluate at.

.. code-block:: python

    from robo.maximize import stochastic_local_search
    maximizer = stochastic_local_search
    
Implementing a main loop
------------------------

Now you can implement the main loop to optimize your objective function. Sample some random point to start with 

.. code-block:: python
   import random
   X = np.empty((1, dims)) 
   for i in x.range(dims):
       X[0,i] = random.random() * (X_upper[i] - X_lower[i]) + X_lower[i];
       x = np.array(X)
       
   Y = objective_function(X)
   for i in xrange(4): 
       model.train(X, Y)
       acquisition_func.update(model)
       new_x = maximizer(acquisition_func, X_lower, X_upper)
       new_y = objective_fkt(np.array(new_x))
       X = np.append(X, new_x, axis=0)
       Y = np.append(Y, new_y, axis=0)







    
Putting it all together:
------------------------

In the one dimensional case you can easily plot all the metods used:

.. code-block:: python

    import GPy
    import matplotlib; matplotlib.use('GTKAgg')
    import matplotlib.pyplot as plt;
    import numpy as np
    import random

    from robo.models import GPyModel
    from robo.acquisition import EI
    from robo.maximize import stochastic_local_search

    def objective_function(x):
        return  np.sin(3*x) * 4*(x-1)* (x+2)

    X_lower = np.array([0])
    X_upper = np.array([6])
    dims = 1
    maximizer = stochastic_local_search

    kernel = GPy.kern.Matern52(input_dim=dims)
    model = GPyModel(kernel, optimize=True, noise_variance = 1e-4, num_restarts=10)
    acquisition_func = EI(model, X_upper= X_upper, X_lower=X_lower, par =0.1) #par is the minimum improvement


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
       
   
Contents:
=========

.. toctree::
   :maxdepth: 2

   acquisition_func
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


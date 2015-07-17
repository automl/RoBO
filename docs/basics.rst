
Basic Usage
===========

RoBO in a single line of code
-------------------------

Define the objective function

.. code-block:: python

	import numpy as np
	from robo.fmin import fmin
	
	def objective_function(x):
	    return  np.sin(3 * x) * 4 * (x - 1) * (x + 2)
	
	X_lower = np.array([0])
	X_upper = np.array([6])
	
Start RoBO

.. code-block:: python

	x_best, fval = fmin(objective_function, X_lower, X_upper)

Bayesian optimization with RoBO
-------------------------------

RoBO is a flexible framework for Bayesian optimization. In a nutshell we can distinguish between different components 
that are necessary for BO, i.e an acquisition function, a model, and a method to optimize the acquisition function. RoBO treats all of those components as modules,
which allows us to easily change and add new methods.
 


Defining an objective function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
configuration and expects the model as input. In the case of EI and PI you additionally have to specify the parameter "par" which controls the balance between exploration and 
exploitation of the acquisition function. 

Maximizing the acquisition function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
    
Putting it all together
^^^^^^^^^^^^^^^^^^^^^^^

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


Saving output
^^^^^^^^^^^^^

    
Implementing the Bayesian optimization loop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to implement the main Bayesian optimization loop by yourself because for instance you want to have a more detail look in what's going you can easily do it
In the one dimensional case you can easily plot all the methods used:

.. code-block:: python

    import GPy
    import matplotlib; matplotlib.use('GTKAgg')
    import matplotlib.pyplot as plt
    import numpy as np

    from robo.models.GPyModel import GPyModel
    from robo.acquisition.EI import EI
    from robo.maximizers.maximize import stochastic_local_search
    from robo.recommendation.incumbent import compute_incumbent


    # The optimization function that we want to optimize. It gets a numpy array with shape (N,D) where N >= 1 are the number of datapoints and D are the number of features
    def objective_function(x):
        return  np.sin(3 * x) * 4 * (x - 1) * (x + 2)

    # Defining the bounds and dimensions of the input space
    X_lower = np.array([0])
    X_upper = np.array([6])
    dims = 1

    # Set the method that we will use to optimize the acquisition function
    maximizer = stochastic_local_search

    # Defining the method to model the objective function
    kernel = GPy.kern.Matern52(input_dim=dims)
    model = GPyModel(kernel, optimize=True, noise_variance=1e-4, num_restarts=10)

    # The acquisition function that we optimize in order to pick a new x
    acquisition_func = EI(model, X_upper=X_upper, X_lower=X_lower, compute_incumbent=compute_incumbent, par=0.1)  # par is the minimum improvement that a point has to obtain

    # Draw one random point and evaluate it to initialize BO
    X = np.array([np.random.uniform(X_lower, X_upper, dims)])
    Y = objective_function(X)

    # This is the main Bayesian optimization loop
    for i in xrange(10):
        # Fit the model on the data we observed so far
        model.train(X, Y)

        # Update the acquisition function model with the retrained model
        acquisition_func.update(model)

        # Optimize the acquisition function to obtain a new point 
        new_x = maximizer(acquisition_func, X_lower, X_upper)

        # Evaluate the point and add the new observation to our set of previous seen points
        new_y = objective_function(np.array(new_x))
        X = np.append(X, new_x, axis=0)
        Y = np.append(Y, new_y, axis=0)

        # Visualize the objective function, model and the acquisition function
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        plotting_range = np.linspace(X_lower[0], X_upper[0], num=1000)
        ax1.plot(plotting_range, objective_function(plotting_range[:, np.newaxis]), color='b', linestyle="--")
        _min_y1, _max_y1 = ax1.get_ylim()
        model.visualize(ax1, X_lower[0], X_upper[0])
        _min_y2, _max_y2 = ax1.get_ylim()
        ax1.set_ylim(min(_min_y1, _min_y2), max(_max_y1, _max_y2))
        mu, var = model.predict(new_x)
        ax1.plot(new_x[0], mu[0], "r.", markeredgewidth=5.0)
        ax2 = acquisition_func.plot(fig, X_lower[0], X_upper[0], plot_attr={"color": "red"}, resolution=1000)

    plt.show(block=True)
   
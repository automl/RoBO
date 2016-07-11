
Basic Usage
===========


.. _fmin:

RoBO in a few lines of code
---------------------------

RoBO offers a simple interface such that you can use it as a optimizer for black box function without knowing what's going on inside. In order to do that you first have to 
define the objective function and the bounds of the configuration space:

.. code-block:: python

	import numpy as np
	from robo.fmin import fmin
	
	def objective_function(x):
	    return  np.sin(3 * x) * 4 * (x - 1) * (x + 2)
	
	X_lower = np.array([0])
	X_upper = np.array([6])
	
The you can start RoBO with the following command and it will return the best configuration / function value it found:

.. code-block:: python

	x_best, fval = fmin(objective_function, X_lower, X_upper)

Note: Make sure that your objective functions always returns a 2 dimensional numpy array np.ndarray(N,D) where N corresponds to the number of data points and D of the number of objectives (normally D=1).

Bayesian optimization with RoBO
-------------------------------

RoBO is a flexible modular framework for Bayesian optimization. It distinguishes between different components 
that are necessary for Bayesian optimization and  treats all of those components as modules which allows us to easily switch between different modules and add new-modules:

* :ref:`task`: This module contains the necessary information that RoBO needs to optimize the objective function (for example an interface for the objective function the input bounds and the dimensionality of the objective function) 
* :ref:`models`: This is the regression method to model the current believe of the objective function 
* :ref:`acquisitionfunctions`: This module represents the acquisition function which acts as a surrogate that determines which configuration will be evaluated in the next step.
* :ref:`maximizers` This module is used to optimize the acquisition function to pick the next configuration



Defining an objective function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

RoBo can optimize any function :math:`X \rightarrow Y` with X as an :math:`N\times D` numpy array and Y as an :math:`N\times K` numpy array. Thereby :math:`N` is the number of points you want to 
evaluate at, :math:`D` is the dimension of the input X and :math:`K` the number of output dimensions (mostly :math:`K = 1`). In order to optimize any function you have to define a task object that implements the interface :ref:`BaseTask`. This class
should contain the objective function and the bounds of the input space.  

.. code-block:: python

    import numpy as np

	from robo.task.base_task import BaseTask

    class ExampleTask(BaseTask):

	    def __init__(self):
	        X_lower = np.array([0])
	        X_upper = np.array([6])
	        super(ExampleTask, self).__init__(X_lower, X_upper)
	
	    def objective_function(self, x):
	        return np.sin(3 * x) * 4 * (x - 1) * (x + 2)

	task = ExampleTask()

Building a model 
^^^^^^^^^^^^^^^^

The first step to optimize this objective function is to define a model that captures the current believe of potential functions. The probably most used method in 
Bayesian optimization for modeling the objective function are Gaussian processes. RoBO uses the well-known `GPy`_ library as implementation for Gaussian processes. The following code snippet
shows how to use a GPy model via RoBO:

.. _GPy: http://sheffieldml.github.io/GPy/

.. code-block:: python

   import GPy

   from robo.models.GPyModel import GPyModel
   
   kernel = GPy.kern.Matern52(input_dim=task_ndims)
   model = GPyModel(kernel, optimize=True, noise_variance = 1e-4, num_restarts=10)

RoBO offers a wrapper interface GPyModel to access the Gaussian processes in GPy. We have to specify a kernel from GPy library as covariance function when we
initialize the model. For further details on those kernels visit `GPy`_. We can either use fix kernel hyperparameter or optimize them by optimizing
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
	
    from robo.acquisition.ei import EI

    acquisition_func = EI(model, X_upper=task.X_upper, X_lower=task.X_lower, par=0.1)


Expected Improvement as well as Probability of Improvement need as additional input the current best configuration (i.e. incumbent). There are different ways to determine 
the incumbent. You can easily plug in any method by giving Expected Improvement a module that is derived from the IncumbentEstimation interface. This module is supposed to return a
configuration and expects the model as input (see the API for more information). In the case of EI and PI you additionally have to specify the parameter "par" which controls the balance between exploration and 
exploitation of the acquisition function. 

Maximizing the acquisition function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The last component is the maximizer which will be used to optimize the acquisition function in order to get a new configuration to evaluate. RoBO offers different ways to
optimize the acquisition functions such as:

 - grid search
 - DIRECT
 - CMA-ES
 - stochastic local search
 

Here we will use the global optimization method Direct to determine the configuration with the highest acquisition value:

.. code-block:: python

	from robo.maximizers.direct import Direct

	maximizer = Direct(acquisition_func, task.X_lower, task.X_upper)
    
Putting it all together
^^^^^^^^^^^^^^^^^^^^^^^

Now we have all the ingredients to optimize our objective function. We can put all the above described components in the BayesianOptimization class

.. code-block:: python

	from robo.solver.bayesian_optimization import BayesianOptimization

	bo = BayesianOptimization(acquisition_fkt=acquisition_func,
	                          model=model,
	                          maximize_fkt=maximizer,
	                          task=task)

Afterwards we can run it by:

.. code-block:: python
	
	bo.run(num_iterations=10)


Saving output
^^^^^^^^^^^^^

You can save RoBO's output by passing the parameters 'save_dir' and 'num_save'. The first parameter 'save_dir' specifies where the results will be saved and
the second parameter 'num_save' after how many iterations the output should be saved. RoBO will save the ouput both in .csv and Json format.

.. code-block:: python

	bo = BayesianOptimization(acquisition_fkt=acquisition_func,
	                          model=model,
	                          maximize_fkt=maximizer,
	                          task=task)
                      		  save_dir="path_to_directory",
                      		  num_save=1)

RoBO will save then the following information in the CSV file:

 - X: The configuration it evaluated so far
 - y: Their corresponding function values
 - incumbent: The best configuration it found so far
 - incumbent_value: Its function value 
 - time_function: The time each function evaluation took
 - optimizer_overhead: The time RoBO needed to pick a new configuration

Following information will be saved in Json in below shown format.

.. code-block:: javascript
	{
	"Acquisiton":{"type" },
	"Model":{"Y" ,"X" ,"hyperparameters" },
	"Task":{"opt": ,"fopt": ,"original_X_lower": ,"original_X_upper": , },
	"Solver":{"optimization_overhead" ,"incumbent_fval" ,"iteration" ,"time_func_eval" ,"incumbent" ,"runtime"  }
	}


    
Implementing the Bayesian optimization loop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example illustrates how you can implement the main Bayesian optimization loop by yourself:

.. code-block:: python

	import GPy
	import matplotlib.pyplot as plt
	import numpy as np
	
	from robo.models.GPyModel import GPyModel
	from robo.acquisition.ei import EI
	from robo.maximizers.direct import Direct
	from robo.task.base_task import BaseTask

	
	# The optimization function that we want to optimize. It gets a numpy array with shape (N,D) where N >= 1 are the number of datapoints and D are the number of features
	class ExampleTask(BaseTask):
	    def __init__(self):
	        X_lower = np.array([0])
	        X_upper = np.array([6])
	        super(ExampleTask, self).__init__(X_lower, X_upper)
	
	    def objective_function(self, x):
	        return np.sin(3 * x) * 4 * (x - 1) * (x + 2)
	
	task = ExampleTask()
	
	# Defining the method to model the objective function
	kernel = GPy.kern.Matern52(input_dim=task.n_dims)
	model = GPyModel(kernel, optimize=True, noise_variance=1e-4, num_restarts=10)
	
	# The acquisition function that we optimize in order to pick a new x
	acquisition_func = EI(model, X_upper=task.X_upper, X_lower=task.X_lower, par=0.1)  # par is the minimum improvement that a point has to obtain
	
	
	# Set the method that we will use to optimize the acquisition function
	maximizer = Direct(acquisition_func, task.X_lower, task.X_upper)
	
	
	# Draw one random point and evaluate it to initialize BO
	X = np.array([np.random.uniform(task.X_lower, task.X_upper, task.n_dims)])
	Y = task.evaluate(X)
	
	# This is the main Bayesian optimization loop
	for i in xrange(10):
	    # Fit the model on the data we observed so far
	    model.train(X, Y)
	
	    # Update the acquisition function model with the retrained model
	    acquisition_func.update(model)
	
	    # Optimize the acquisition function to obtain a new point
	    new_x = maximizer.maximize()
	
	    # Evaluate the point and add the new observation to our set of previous seen points
	    new_y = task.objective_function(np.array(new_x))
	    X = np.append(X, new_x, axis=0)
	    Y = np.append(Y, new_y, axis=0)
   

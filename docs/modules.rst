
Modules
=======

.. _task:

Task
------

In order to optimize any function, RoBO expects a  task object that is derived from the BaseTask class. If you want to optimize your own objective function you need to derive from 
this base class and implement at least the objective_function(self, x): method as well as the self.X_lower and self.X_upper. However you can add any additional information here. For example
the well-known synthetic benchmark function Branin would look like:

.. code-block:: python
	
	import numpy as np
	
	from robo.task.base_task import BaseTask
	
	
	class Branin(BaseTask):
	
	    def __init__(self):
	        self.X_lower = np.array([-5, 0])
	        self.X_upper = np.array([10, 15])
	        self.n_dims = 2
	        self.opt = np.array([[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]])
	        self.fopt = 0.397887
	
	    def objective_function(self, x):
	        y = (x[:, 1] - (5.1 / (4 * np.pi ** 2)) * x[:, 0] ** 2 + 5 * x[:, 0] / np.pi - 6) ** 2
	        y += 10 * (1 - 1 / (8 * np.pi)) * np.cos(x[:, 0]) + 10
	
	        return y[:, np.newaxis]

In this case we can also set the known global optimas and the best function value. This allows to plot the distance between the best found function value and the global optimum.
However, of course for real world benchmark we do not have this information so you can just drop them.
 
Note that the method objective_function(self, x) expects a 2 dimensional numpy array and also returns a two dimension numpy array. Furthermore bounds are also specified as
numpy arrays:

.. code-block:: python

        self.X_lower = np.array([-5, 0])
        self.X_upper = np.array([10, 15])


 
.. _models:

Models
------

The model class contains the regression model that is used to model the objective function. To use any kind of regression model in RoBO it has to implement the interface from them BaseModel class.
Also each model has its own hyperparameters (for instance the type of kernel for GaussianProcesses). Here is an example how to use GPs in RoBO:


.. code-block:: python

	import GPy
	from robo.models.GPyModel import GPyModel

	kernel = GPy.kern.Matern52(input_dim=task.n_dims)
	model = GPyModel(kernel, optimize=True, noise_variance=1e-4, num_restarts=10)
	model.train(X, Y)
	mean, var = model.predict(X_test)
 


.. _acquisitionfunctions:

Acquisition functions
---------------------


.. Acquisition functions that are implemented in RoBO
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The role of an acquisition function in Bayesian optimization is to compute how useful it is to evaluate a candidate x. In each iteration RoBO maximizes the acquisition function in
order to pick a new configuration which will be then evaluated. The following acquisition functions are currently implemented in RoBO and each of them has its own properties.

* :ref:`Expected Improvement`
* :ref:`Log Expected Improvement`
* :ref:`Probability of Improvement`
* :ref:`Entropy`
* :ref:`EntropyMC`
* :ref:`Upper Confidence Bound`

Each acquisition function expects at least a model and a the input bounds of the task as input, for example:

.. code-block:: python

    acquisition_func = EI(model, X_upper=task.X_upper, X_lower=task.X_lower)

Furthermore, every acquisition functions has its own individual parameters that control its computations.
To compute now the for a specific x its acquisition value you can call. The input point x has to be a :math:`1\times D` numpy array:

.. code-block:: python

	val = acquisition_func(x)

If you marginalize over the hyperparameter of a Gaussian Process via the GPyMCMC module this command will compute the sum over the acquisition value computed based on every single GP

Some acquisition functions allow to compute gradient, you can compute them by:

.. code-block:: python

	val, grad = acquisition_func(x, derivative=True)

If you updated your model with new data you also have to update you acquisition function by:

.. code-block:: python

	acquisition_func.update(model)

	
.. _maximizers:

Maximizers
----------

The role of the maximizers is to optimize the acquisition function in order to find a new configuration which will be evaluated in the next iteration. All maximizer have to implement
the BaseMaximizer interface. Ever maximizer has its own parameter (see here for more information) but all expect at least an acquisition function object as well as the bounds of the
input space:

.. code-block:: python

	maximizer = CMAES(acquisition_func, task.X_lower, task.X_upper)
	
Afterwards you can easily optimize the acquisition function by:

.. code-block:: python

	x_new = maximizer.maximize()
	

.. _solver:

Solver
------

The solver module represents the actual Bayesian optimizer. The standard module is BayesianOptimization which implements the vanilla BO procedure. 

.. code-block:: python

    bo = BayesianOptimization(acquisition_fkt=acquisition_func,
                      model=model,
                      maximize_fkt=maximizer,
                      task=task,
                      save_dir=os.path.join(save_dir, acq_method + "_" + max_method, "run_" + str(run_id)),
                      num_save=1)

    bo.run(num_iterations)
    
If you just want to perform one single iteration based on some given data to get a new configuration you can call:

.. code-block:: python

	new_x = bo.choose_next(X, Y)

It also offers functions to save the output and measure the time of each function evaluation and the optimization overhead. If you develop a new BO strategy it might be a good idea to derive from this class and uses those functionalities to be 
compatible with RoBO's tools.

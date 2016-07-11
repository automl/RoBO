Advanced
========


REMBO
-----

Random EMbedding  Bayesian Optimization (REMBO) tackles the problem of high dimensional input spaces with low effective dimensionality. It creates a random matrix to perform a random
projection from a high dimensional space into a smaller embedded subspace (`rembo-paper`_).
If you want to use REMBO for you objective function you just have to derive from the REMBO task an call its __init__() function in the constructor:

.. _rembo-paper: http://www.cs.ubc.ca/~hutter/papers/13-IJCAI-BO-highdim.pdf

.. code-block:: python

	class BraninInBillionDims(REMBO):
	    def __init__(self):
	        self.b = Branin()
	        X_lower = np.concatenate((self.b.X_lower, np.zeros([999998])))
	        X_upper = np.concatenate((self.b.X_upper, np.ones([999998])))
	        super(BraninInBillionDims, self).__init__(X_lower, X_upper, d=2)
	
	    def objective_function(self, x):
	        return self.b.objective_function(x[:, :2])

Afterwards you can simply optimize your task such as any other task. It will then automatically perform Bayesian optimization in the lower embedded subspace to find a new configuration.
To evaluate a configuration it will be transformed back to the original space. 

.. code-block:: python

	task = BraninInBillionDims()
	kernel = GPy.kern.Matern52(input_dim=task.n_dims)
	model = GPyModel(kernel, optimize=True, noise_variance=1e-3, num_restarts=10)
	acquisition_func = EI(model, task.X_lower, task.X_upper, compute_incumbent)
	maximizer = CMAES(acquisition_func, task.X_lower, task.X_upper)
	bo = BayesianOptimization(acquisition_fkt=acquisition_func,
	                      model=model,
	                      maximize_fkt=maximizer,
	                      task=task)
	
	bo.run(500)


Bayesian optimization with MCMC sampling of the GP's hyperparameters
--------------------------------------------------------------------

So far we optimized the GPy's hyperparameter by maximizing the marginal loglikelihood. If you want to marginalise over hyperparameter you can use the GPyModelMCMC module:

.. code-block:: python

	kernel = GPy.kern.Matern52(input_dim=branin.n_dims)
	model = GPyModelMCMC(kernel, burnin=20, chain_length=100, n_hypers=10)
	
It used the HMC method implemented in GPy to sample the marginal loglikelihood. Afterwards you can simply plug it into your acquisition functions

.. code-block:: python

	acquisition_func = EI(model, X_upper=branin.X_upper, X_lower=branin.X_lower, compute_incumbent=compute_incumbent, par=0.1)

	maximizer = Direct(acquisition_func, branin.X_lower, branin.X_upper)
	bo = BayesianOptimization(acquisition_fkt=acquisition_func,
                          model=model,
                          maximize_fkt=maximizer,
                          task=task)

	bo.run(10)

RoBO will then compute an marginalised acquistion value by computing the acquisition value based on each single GP and sum over all of them.


Fabolas
-------

The general idea of Fabolas is to expand the traditional way to model the objective function :math:`f(\bm{x}, s)` by an additional input :math:`s` that specifies the amount of training data to evaluate a point :math:`\bm{x}`:


At the end we want to find the best points :math:`\bm{x}_{\star}` on the full dataset :math:`s=s_{max}`. Because of that Fabolas uses the information gain acquisition function but models the distribution over the minimum :math:`p_{min}(\bm{x, s})` only on the subspace :math:`s_{max}` such that :math:`p_{min}(\bm{x, s=s_{max}})`.

By additionally modeling the evaluation time of a point :math:`\bm{x}` and dividing the information gain by the cost :math:`c(\bm{x}, s)` it would take to evaluate :math:`\bm{x}` on :math:`s`, Fabolas evaluate points only  on small subsets of the data and extrapolates their error to the full dataset size. For more details have a look at the paper http://arxiv.org/abs/1605.07079


Fabolas has the same interface as RoBO`s fmin function (see :ref:`fmin`). First you have to define your objective function which now should depend on :math:`\bm{x}` and :math:`s`:

.. code-block:: python

        def objective_function(x, s):
                # Train your algorithm here with x on the dataset subset with length s
                # Estimate the validation error and the cost on the validation data set
                return np.array([[validation_error]]), np.array([[cost]])

Your objective function should return the validation error and the total cost :math:`c(\bm{x}, s)` of the point :math:`\bm{x}`. Normally the cost is the time it took to train and validate :math:`\bm{x}`.
After defining your objective function you also have to define the input bounds for :math:`\bm{x}` and :math:`s`. Make sure that the dataset size :math:`s` is the last dimension.
It is often a good idea to set the data set size on a log scale. 

.. code-block:: python

        X_lower = np.array([-10, -10, s_min])
        X_upper = np.array([10, 10, s_max])

Then you can call Fabolas by:

.. code-block:: python

        x_best = fabolas_fmin(objective_function, X_lower, X_upper, num_iterations=100)

You can find a full example for training a support vector machine on MNIST `here <http://https://github.com/automl/RoBO/blob/development/examples/example_fmin_fabolas.py>`_

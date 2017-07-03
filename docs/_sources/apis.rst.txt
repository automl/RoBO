APIs
====

-------------
Main Modules:
-------------



Acquisition:
------------

.. autoclass:: robo.acquisition_functions.base_acquisition.BaseAcquisitionFunction
   :members:

.. autoclass:: robo.acquisition_functions.ei.EI
   :members:

.. autoclass:: robo.acquisition_functions.information_gain.InformationGain
   :members:

.. autoclass:: robo.acquisition_functions.information_gain_mc.InformationGainMC
   :members:

.. autoclass:: robo.acquisition_functions.information_gain_per_unit_cost.InformationGainPerUnitCost
   :members:

.. autoclass:: robo.acquisition_functions.integrated_acquisition.IntegratedAcquisition
   :members:

.. autoclass:: robo.acquisition_functions.lcb.LCB
   :members:

.. autoclass:: robo.acquisition_functions.log_ei.LogEI
   :members:

.. autoclass:: robo.acquisition_functions.marginalization.MarginalizationGPMCMC
   :members:

.. autoclass:: robo.acquisition_functions.pi.PI
   :members:


Models:
------------

.. autoclass:: robo.models.base_model.BaseModel
   :members:

.. autoclass:: robo.models.bagged_networks.BaggedNets
   :members:

.. autoclass:: robo.models.bayesian_linear_regression.BayesianLinearRegression
   :members:

.. autoclass:: robo.models.bnn.BayesianNeuralNetwork
   :members:

.. autoclass:: robo.models.dngo.DNGO
   :members:

.. autoclass:: robo.models.fabolas_gp.FabolasGPMCMC
   :members:

.. autoclass:: robo.models.gaussian_process.GaussianProcess
   :members:

.. autoclass:: robo.models.gaussian_process_mcmc.GaussianProcessMCMC
   :members:

.. autoclass:: robo.models.neural_network.SGDNet
   :members:

.. autoclass:: robo.models.random_forest.RandomForest
   :members:


Maximizers:
------------

.. autoclass:: robo.maximizers.cmaes.CMAES
   :members:

.. autoclass:: robo.maximizers.direct.Direct
   :members:

.. autoclass:: robo.maximizers.grid_search.GridSearch
   :members:

.. autoclass:: robo.maximizers.scipy_optimizer.SciPyOptimizer
   :members:

.. autoclass:: robo.maximizers.base_maximizer.BaseMaximizer
   :members:


Priors:
-------

.. autoclass:: robo.priors.base_prior.BasePrior
   :members:

.. autoclass:: robo.priors.base_prior.TophatPrior
   :members:

.. autoclass:: robo.priors.base_prior.HorseshoePrior
   :members:

.. autoclass:: robo.priors.base_prior.LognormalPrior
   :members:

.. autoclass:: robo.priors.base_prior.NormalPrior
   :members:

.. autoclass:: robo.priors.bayesian_linear_regression_prior.BayesianLinearRegressionPrior
   :members:

.. autoclass:: robo.priors.default_priors.DefaultPrior
   :members:

.. autoclass:: robo.priors.env_priors.EnvPrior
   :members:

.. autoclass:: robo.priors.env_priors.EnvNoisePrior
   :members:

.. autoclass:: robo.priors.env_priors.MTBOPrior
   :members:

Solver:
-------

.. autoclass:: robo.solver.base_solver.BaseSolver
   :members:

.. autoclass:: robo.solver.bayesian_optimization.BayesianOptimization
   :members:

.. autoclass:: robo.solver.contextual_bayesian_optimization._ContextualAcquisitionFunction
   :members:

.. autoclass:: robo.solver.contextual_bayesian_optimization.MeanAcquisitionFunction
   :members:

.. autoclass:: robo.solver.contextual_bayesian_optimization.ContextualBayesianOptimization
   :members:

.. autoclass:: robo.solver.hyperband_datasets_size.HyperBand_DataSubsets
   :members:

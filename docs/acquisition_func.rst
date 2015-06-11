
Acquisition functions:
=====================


Acquisition functions that are implemented in RoBO:
---------------------------------------------------

The role of an acquisition function in Bayesian optimization is to compute how useful it is to evaluate a candidate x. In each iteration RoBO maximizes the acquisition function in
order to pick a new configuration which will be then evaluated. The following acquisition functions are implemented in RoBO and each of them has its own properties.


Expected Improvement
""""""""""""""""""""
.. autoclass:: robo.acquisition.EI.EI
   :members:
   :private-members: 
   :special-members: __call__

Log Expected Improvement
""""""""""""""""""""""""

.. autoclass:: robo.acquisition.LogEI.LogEI
   :members:
   :private-members: 
   :special-members: __call__
   

Probability of Improvement
""""""""""""""""""""""""""

.. autoclass:: robo.acquisition.PI.PI
   :members:
   :private-members: 
   :special-members: __call__

Entropy Search
""""""""""""""

.. autoclass:: robo.acquisition.Entropy
   :members:
   :private-members: 
   :special-members: __call__
 
Entropy Search (Monte Carlo)
""""""""""""""""""""""""""""

.. autoclass:: robo.acquisition.EntropyMC.EntropyMC
   :members:
   :private-members: 
   :special-members: __call__

Upper Confidence Bound
""""""""""""""""""""""

.. autoclass:: robo.acquisition.UCB.UCB
   :members:
   :private-members: 
   :special-members: __call__


How to implement your own acquisition function:
-----------------------------------------------

If you want to implement your own acquisition function such that it can be used in RoBO you have to derive it from the base class and implement its abstract functions.

Base class
""""""""""
.. autoclass:: robo.acquisition.base.AcquisitionFunction
   :members: 
   :private-members: 
   :special-members: __call__
   
Acquisition functions:
----------------------

Base class
==========

.. autoclass:: robo.acquisition.base.AcquisitionFunction
   :members: 
   :private-members: 
   :special-members: __call__
   

Expected Improvement
====================

.. autoclass:: robo.acquisition.EI.EI
   :members:
   :private-members: 
   :special-members: __call__

Log Expected Improvement
========================

.. autoclass:: robo.acquisition.LogEI.LogEI
   :members:
   :private-members: 
   :special-members: __call__
   

Probability of Improvement
==========================

.. autoclass:: robo.acquisition.PI.PI
   :members:
   :private-members: 
   :special-members: __call__

Entropy Search
==============

.. autoclass:: robo.acquisition.Entropy
   :members:
   :private-members: 
   :special-members: __call__
 
Entropy Search (Monte Carlo)
============================

.. autoclass:: robo.acquisition.EntropyMC.EntropyMC
   :members:
   :private-members: 
   :special-members: __call__

Upper Confidence Bound
======================

.. autoclass:: robo.acquisition.UCB.UCB
   :members:
   :private-members: 
   :special-members: __call__
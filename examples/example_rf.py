'''
Created on Jun 23, 2015

@author: Aaron Klein
'''

from robo.models.random_forest import RandomForest
from robo.acquisition.ei import EI
#from robo.maximizers.direct import Direct
from robo.maximizers.cmaes import CMAES
from robo.task.synthetic_functions.branin import Branin
from robo.solver.bayesian_optimization import BayesianOptimization
from robo.incumbent.posterior_optimization import PosteriorMeanAndStdOptimization


# Specifies the task object that defines the objective functions and
# the bounds of the input space
branin = Branin()

# Instantiate the random forest. Branin does not have any categorical
# values thus we pass a np.zero vector here.
model = RandomForest(branin.types)

# Define the acquisition function
acquisition_func = EI(model,
                     X_upper=branin.X_upper,
                     X_lower=branin.X_lower)

# Strategy of estimating the incumbent
rec = PosteriorMeanAndStdOptimization(model, branin.X_lower,
                                      branin.X_upper, with_gradients=False)

# Define the maximizer
maximizer = CMAES(acquisition_func, branin.X_lower, branin.X_upper)

# Now we defined everything we need to instantiate the solver
bo = BayesianOptimization(acquisition_func=acquisition_func,
                          model=model,
                          maximize_func=maximizer,
                          task=branin,
                          incumbent_estimation=rec)

bo.run(100)

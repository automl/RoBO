import numpy as np

from robo import BayesianOptimization
from robo.models.sklearn_gp import SkLearnGP
from robo.models.GPyModel import GPyModel, GPy
from robo.acquisition.EI import EI
from robo.maximizers.maximize import cma
from robo.benchmarks.branin import branin, get_branin_bounds


maximize_fkt = cma
model = SkLearnGP()

init_X = np.random.randn(2, 2)
init_y = np.zeros([2])
init_y[0] = branin(init_X[0])
init_y[1] = branin(init_X[1])


X_lower, X_upper, dims = get_branin_bounds()
ei = EI(model, X_upper=X_upper, X_lower=X_lower, par=0.3)

bo = BayesianOptimization(acquisition_fkt=ei,
                          model=model,
                          maximize_fkt=maximize_fkt,
                          X_lower=X_lower,
                          X_upper=X_upper,
                          dims=dims,
                          objective_fkt=branin,
                          save_dir=None)

_, config, value, _, _ = bo.run(50, X=init_X, Y=init_y)
print config
print value

print branin(config)
import os
import sys
import json
import numpy as np

import logging
logging.basicConfig(level=logging.DEBUG)
from robo.fmin import bayesian_optimization, entropy_search, bohamiann, random_search

import hpolib.benchmarks.synthetic_functions as hpobench


run_id = int(sys.argv[1])
benchmark = sys.argv[2]
method = sys.argv[3]

rng = np.random.RandomState(run_id)


if len(sys.argv) > 4:
    acquisition = sys.argv[4]
    maximizer = sys.argv[5]
else:
    acquisition = "log_ei"
    maximizer = "random"

n_iters = 200
n_init = 2
output_path = "./experiments/RoBO/synthetic_funcs"

if benchmark == "branin":
    f = hpobench.Branin()
elif benchmark == "hartmann3":
    f = hpobench.Hartmann3()
elif benchmark == "hartmann6":
    f = hpobench.Hartmann6()
elif benchmark == "camelback":
    f = hpobench.Camelback()
elif benchmark == "goldstein_price":
    f = hpobench.GoldsteinPrice()
elif benchmark == "rosenbrock":
    f = hpobench.Rosenbrock()
elif benchmark == "sin_one":
    f = hpobench.SinOne()
elif benchmark == "sin_two":
    f = hpobench.SinTwo()
elif benchmark == "bohachevsky":
    f = hpobench.Bohachevsky()
elif benchmark == "levy":
    f = hpobench.Levy()

info = f.get_meta_information()
bounds = np.array(info['bounds'])

if method == "entropy_search":
    results = entropy_search(f, bounds[:, 0], bounds[:, 1],
                             num_iterations=n_iters, n_init=n_init)
elif method == "gp_mcmc":
    results = bayesian_optimization(f, bounds[:, 0], bounds[:, 1],
                                    num_iterations=n_iters,
                                    acquisition_func=acquisition,
                                    maximizer=maximizer,
                                    n_init=n_init, model_type="gp_mcmc", rng=rng)
elif method == "gp":
    results = bayesian_optimization(f, bounds[:, 0], bounds[:, 1],
                                    num_iterations=n_iters,
                                    acquisition_func=acquisition,
                                    maximizer=maximizer,
                                    n_init=n_init, model_type="gp")
elif method == "rf":
    results = bayesian_optimization(f, bounds[:, 0], bounds[:, 1],
                                    num_iterations=n_iters,
                                    n_init=n_init, model_type="rf")
elif method == "random_search":
    results = random_search(f, bounds[:, 0], bounds[:, 1],
                            num_iterations=n_iters)
elif method == "bohamiann":
    results = bohamiann(f, bounds[:, 0], bounds[:, 1],
                        num_iterations=n_iters,
                        n_init=n_init)

# Offline Evaluation
regret = []
for inc in results["incumbents"]:
    r = f.objective_function(inc)
    regret.append(r["function_value"] - info["f_opt"])

results["method"] = method
results["benchmark"] = benchmark
results["regret"] = regret
results["run_id"] = run_id

print(np.log(regret))

p = os.path.join(output_path, method)
os.makedirs(p, exist_ok=True)

fh = open(os.path.join(p, '%s_run_%d.json' % (benchmark, run_id)), 'w')
json.dump(results, fh)

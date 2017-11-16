import os
import sys
import json
import numpy as np

import hpolib.benchmarks.synthetic_functions as hpobench

from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario

run_id = int(sys.argv[1])
benchmark = sys.argv[2]

n_iters = 200

output_path = "./experiments/RoBO/synthetic_funcs/"

if benchmark == "branin":
    b = hpobench.Branin()
elif benchmark == "hartmann3":
    b = hpobench.Hartmann3()
elif benchmark == "hartmann6":
    b = hpobench.Hartmann6()
elif benchmark == "camelback":
    b = hpobench.Camelback()
elif benchmark == "goldstein_price":
    b = hpobench.GoldsteinPrice()
elif benchmark == "rosenbrock":
    b = hpobench.Rosenbrock()
elif benchmark == "sin_one":
    b = hpobench.SinOne()
elif benchmark == "sin_two":
    b = hpobench.SinTwo()
elif benchmark == "bohachevsky":
    b = hpobench.Bohachevsky()
elif benchmark == "levy":
    b = hpobench.Levy()

info = b.get_meta_information()

scenario = Scenario({"run_obj": "quality",
                     "runcount-limit": n_iters,
                     "cs": b.get_configuration_space(),
                     "deterministic": "true",
                     "initial_incumbent": "RANDOM",
                     "output_dir": ""})

smac = SMAC(scenario=scenario, tae_runner=b)
smac.optimize()

rh = smac.runhistory
incs = []
inc = None
idx = 1
t = smac.get_trajectory()
for i in range(n_iters):

    if idx < len(t) and i == t[idx].ta_runs - 1:
        inc = t[idx].incumbent
        idx += 1
    incs.append(inc)

# Offline Evaluation
regret = []
runtime = []
cum_cost = 0

X, y, _ = smac.get_X_y()
for i, x in enumerate(X):
    y = b.objective_function_test(incs[i])["function_value"]
    regret.append(y - info["f_opt"])

results = dict()
results["method"] = "smac"
results["benchmark"] = benchmark
results["regret"] = regret
results["run_id"] = run_id


p = os.path.join(output_path, "smac")
os.makedirs(p, exist_ok=True)

fh = open(os.path.join(p, '%s_run_%d.json' % (benchmark, run_id)), 'w')
json.dump(results, fh)

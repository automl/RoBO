import os
import sys
import json
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

from robo.fmin import entropy_search

import hpolib.benchmarks.synthetic_functions as hpobench


run_id = int(sys.argv[1])
benchmark = sys.argv[2]
seed = int(sys.argv[3])

rng = np.random.RandomState(seed)

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

num_iterations = 200
output_path = "./experiments/hpolib2/synthetic_functions/results/%s/entropy_search_%d" % (benchmark, run_id)

os.makedirs(output_path, exist_ok=True)

info = f.get_meta_information()
bounds = np.array(info['bounds'])
res = entropy_search(f, bounds[:, 0], bounds[:, 1],
                     num_iterations=num_iterations, n_init=2,
                     rng=rng, output_path=None)

regret = []
current_inc = None
current_inc_val = None

for inc in res["incumbents"]:
    if current_inc != inc:
        y = f.objective_function_test(inc)["function_value"]
        current_inc = inc
        current_inc_val = np.abs(y - info["f_opt"])
    regret.append(current_inc_val)

results = dict()
results["run_id"] = run_id
results["regret"] = regret
results["runtime"] = res["runtime"]

with open(os.path.join(output_path, 'results_%d.json' % run_id), 'w') as fh:
    json.dump(results, fh)

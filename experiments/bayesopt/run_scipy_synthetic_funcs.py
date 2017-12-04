import os
import sys
import json
import numpy as np
import scipy.optimize as spopt

from robo.initial_design import init_random_uniform


import hpolib.benchmarks.synthetic_functions as hpobench


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

X = []
y = []


def wrapper(x):
    X.append(x.tolist())
    y_ = b.objective_function(np.clip(x, lower, upper))['function_value']
    y.append(y_)
    return y_


info = b.get_meta_information()

bounds = np.array(info['bounds'])
lower = bounds[:, 0]
upper = bounds[:, 1]

start_point = init_random_uniform(lower, upper, 1)[0]

res = spopt.minimize(wrapper, start_point, bounds=bounds,
                     method='L-BFGS-B', options={"ftol": 1e-50, "gtol": 1e-10, "maxfun": n_iters, "disp": False})


if len(y) >= n_iters:
    fvals = np.array(y[:n_iters])
else:
    fvals = np.ones([n_iters]) * y[-1]
    fvals[:len(y)] = y

incs = []
incumbent_val = []
curr_inc_val = sys.float_info.max
inc = None
for i, f in enumerate(fvals):
    if curr_inc_val > f:
        curr_inc_val = f
        inc = X[i]
    incumbent_val.append(curr_inc_val)
    incs.append(inc)

regret = []
for inc in incs:
    r = b.objective_function(inc)
    regret.append(r["function_value"] - info["f_opt"])

results = dict()
results["method"] = "scipy"
results["benchmark"] = benchmark
results["regret"] = regret
results["run_id"] = run_id
results["incumbents"] = incs
results["incumbent_values"] = incumbent_val
results["X"] = X
results["y"] = y


p = os.path.join(output_path, "scipy")
os.makedirs(p, exist_ok=True)

fh = open(os.path.join(p, '%s_run_%d.json' % (benchmark, run_id)), 'w')
json.dump(results, fh)

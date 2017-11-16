import os
import sys
import cma
import json
import numpy as np

from robo.initial_design import init_random_uniform

from hpolib.benchmarks.ml.surrogate_svm import SurrogateSVM
from hpolib.benchmarks.ml.surrogate_cnn import SurrogateCNN
from hpolib.benchmarks.ml.surrogate_fcnet import SurrogateFCNet


run_id = int(sys.argv[1])
benchmark = sys.argv[2]

n_iters = 50

output_path = "./experiments/RoBO/surrogates/"

if benchmark == "svm_mnist":
    b = SurrogateSVM(path="/ihome/kleinaa/devel/git/HPOlib/surrogates/")
elif benchmark == "cnn_cifar10":
    b = SurrogateCNN(path="/ihome/kleinaa/devel/git/HPOlib/surrogates/")
elif benchmark == "fcnet_mnist":
    b = SurrogateFCNet(path="/ihome/kleinaa/devel/git/HPOlib/surrogates/")

info = b.get_meta_information()

X = []
y = []


def wrapper(x):
    X.append(x.tolist())
    y_ = b.objective_function(x)['function_value']
    y.append(y_)
    return y_

# Dimension and bounds of the function
bounds = np.array(info['bounds'])

dimensions = len(bounds)
lower = bounds[:, 0]
upper = bounds[:, 1]
start_point = init_random_uniform(lower, upper, 1)[0]

# Evolution Strategy
es = cma.CMAEvolutionStrategy(start_point, 0.6, {'bounds': [lower, upper],
                                                 "maxfevals": n_iters})

es.optimize(wrapper, n_iters)

X = X[:n_iters]
y = y[:n_iters]
fvals = np.array(y)

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

# Offline Evaluation
test_error = []
runtime = []
cum_cost = 0

results = dict()

for i, inc in enumerate(incs):

    y = b.objective_function_test(np.array(inc))["function_value"]
    test_error.append(y)

    # Compute the time it would have taken to evaluate this configuration
    c = b.objective_function(np.array(X[i]))["cost"]
    cum_cost += c
    runtime.append(cum_cost)

# Estimate the runtime as the optimization overhead + estimated cost
results["runtime"] = runtime
results["test_error"] = test_error

results["method"] = "cmaes"
results["benchmark"] = benchmark
results["run_id"] = run_id
results["incumbents"] = incs
results["incumbent_values"] = incumbent_val
results["X"] = X
results["y"] = y

print(test_error)


p = os.path.join(output_path, benchmark, "cmaes")
os.makedirs(p, exist_ok=True)

fh = open(os.path.join(p, '%s_run_%d.json' % (benchmark, run_id)), 'w')
json.dump(results, fh)

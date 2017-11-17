import os
import sys
import json
import numpy as np

from hpolib.benchmarks.ml.surrogate_svm import SurrogateSVM
from hpolib.benchmarks.ml.surrogate_cnn import SurrogateCNN
from hpolib.benchmarks.ml.surrogate_fcnet import SurrogateFCNet
from hpolib.benchmarks.ml.surrogate_paramnet import SurrogateParamNet, PredictiveTerminationCriterion

from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario

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
elif benchmark == "paramnet":
    dataset = sys.argv[3]
    b = SurrogateParamNet(dataset, "/ihome/kleinaa/devel/git/HPOlib/surrogates/")

    benchmark += "_" + dataset

elif benchmark == "paramnet_ptc":
    dataset = sys.argv[3]
    b = PredictiveTerminationCriterion(interval=5,  dataset=dataset,
                                       path="/ihome/kleinaa/devel/git/HPOlib/surrogates/")

    benchmark += "_" + dataset

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
        print(inc)
    incs.append(inc)

# Offline Evaluation
test_error = []
runtime = []
cum_cost = 0

X, y, _ = smac.get_X_y()
for i, x in enumerate(X):
    # Compute the time it would have taken to evaluate this configuration
    c = b.objective_function(x)["cost"]
    cum_cost += c
    runtime.append(cum_cost)
    y = b.objective_function_test(incs[i])["function_value"]
    test_error.append(y)

results = dict()
results["runtime"] = runtime
results["test_error"] = test_error

results["method"] = "smac"
results["benchmark"] = benchmark
results["run_id"] = run_id

p = os.path.join(output_path, benchmark, "smac")
os.makedirs(p, exist_ok=True)

fh = open(os.path.join(p, '%s_run_%d.json' % (benchmark, run_id)), 'w')
json.dump(results, fh)

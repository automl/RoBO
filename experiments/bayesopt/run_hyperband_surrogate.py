import os
import sys
import json
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

from robo.solver.hyperband_datasets_size import HyperBand_DataSubsets
from hpolib.benchmarks.ml.surrogate_svm import SurrogateSVM
from hpolib.benchmarks.ml.surrogate_cnn import SurrogateCNN
from hpolib.benchmarks.ml.surrogate_fcnet import SurrogateFCNet

run_id = int(sys.argv[1])
benchmark = sys.argv[2]

rng = np.random.RandomState(run_id)
dataset = "surrogate"

if benchmark == "svm_mnist":
    f = SurrogateSVM(path="/ihome/kleinaa/devel/git/HPOlib/surrogates/")
elif benchmark == "cnn_cifar10":
    f = SurrogateCNN(path="/ihome/kleinaa/devel/git/HPOlib/surrogates/")
elif benchmark == "fcnet_mnist":
    f = SurrogateFCNet(path="/ihome/kleinaa/devel/git/HPOlib/surrogates/")

output_path = "./experiments/RoBO/surrogates"

eta = 3.
B = -int(np.log(f.s_min)/np.log(3))

opt = HyperBand_DataSubsets(f, eta, eta**(-(B-1)), rng=rng)

opt.run(int(20 / B * 1.5))

results = dict()

test_error = []
runtime = []
cum_cost = 0
for i, c in enumerate(opt.incumbents):
    test_error.append(f.objective_function_test(c)["function_value"])

    cum_cost += opt.time_func_eval_incumbent[i]
    runtime.append(opt.runtime[i] + cum_cost)

results["runtime"] = runtime
results["test_error"] = test_error
results["run_id"] = run_id

p = os.path.join(output_path, benchmark, "hyperband")
os.makedirs(p, exist_ok=True)

fh = open(os.path.join(p, '%s_run_%d.json' % (benchmark, run_id)), 'w')
json.dump(results, fh)
fh.close()

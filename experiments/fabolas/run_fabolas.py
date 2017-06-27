import os
import sys
import json
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

from robo.fmin import fabolas

from hpolib.benchmarks.ml.svm_benchmark import SvmOnMnist, SvmOnVehicle, SvmOnCovertype, SvmOnAdult, SvmOnHiggs
from hpolib.benchmarks.ml.residual_networks import ResidualNeuralNetworkOnCIFAR10
from hpolib.benchmarks.ml.conv_net import ConvolutionalNeuralNetworkOnCIFAR10, ConvolutionalNeuralNetworkOnSVHN


run_id = int(sys.argv[1])
dataset = sys.argv[2]
seed = int(sys.argv[3])

rng = np.random.RandomState(seed)

if dataset == "mnist":
    f = SvmOnMnist(rng=rng)
    num_iterations = 80
    output_path = "./experiments/fabolas/results/svm_%s/fabolas_%d" % (dataset, run_id)
    s_max = f.train.shape[0]
    s_min = 100
    subsets = [128] * 8
    subsets.extend([64] * 4)
    subsets.extend([32] * 2)
    subsets.extend([4] * 1)

elif dataset == "vehicle":
    f = SvmOnVehicle(rng=rng)
    num_iterations = 80
    output_path = "./experiments/fabolas/results/svm_%s/fabolas_%d" % (dataset, run_id)
    s_max = f.train.shape[0]
    s_min = 100
    subsets = [128] * 8
    subsets.extend([64] * 4)
    subsets.extend([32] * 2)
    subsets.extend([4] * 1)

elif dataset == "higgs":
    f = SvmOnHiggs(rng=rng)
    num_iterations = 80
    output_path = "./experiments/fabolas/results/svm_%s/fabolas_%d" % (dataset, run_id)
    s_max = f.train.shape[0]
    s_min = 100
    subsets = [128] * 8
    subsets.extend([64] * 4)
    subsets.extend([32] * 2)
    subsets.extend([4] * 1)

elif dataset == "adult":
    f = SvmOnAdult(rng=rng)
    num_iterations = 80
    output_path = "./experiments/fabolas/results/svm_%s/fabolas_%d" % (dataset, run_id)
    s_max = f.train.shape[0]
    s_min = 100
    subsets = [128] * 8
    subsets.extend([64] * 4)
    subsets.extend([32] * 2)
    subsets.extend([4] * 1)

elif dataset == "covertype":
    f = SvmOnCovertype(rng=rng)
    num_iterations = 150
    output_path = "./experiments/fabolas/results/svm_%s/fabolas_%d" % (dataset, run_id)
    s_max = f.train.shape[0]
    s_min = 100
    subsets = [128] * 8
    subsets.extend([64] * 4)
    subsets.extend([32] * 2)
    subsets.extend([4] * 1)

elif dataset == "cifar10":
    f = ConvolutionalNeuralNetworkOnCIFAR10(rng=rng)
    num_iterations = 50
    output_path = "./experiments/fabolas/results/cnn_%s/fabolas_%d" % (dataset, run_id)
    s_max = f.train.shape[0]
    s_min = 512  # Maximum batch size
    subsets = [64] * 8
    subsets.extend([32] * 4)
    subsets.extend([16] * 2)

elif dataset == "svhn":
    f = ConvolutionalNeuralNetworkOnSVHN(rng=rng)
    num_iterations = 50
    output_path = "./experiments/fabolas/results/cnn_%s/fabolas_%d" % (dataset, run_id)
    s_max = f.train.shape[0]
    s_min = 512  # Maximum batch size
    subsets = [64] * 8
    subsets.extend([32] * 4)
    subsets.extend([16] * 2)

elif dataset == "res_net":
    f = ResidualNeuralNetworkOnCIFAR10(rng=rng)
    num_iterations = 50
    output_path = "./experiments/fabolas/results/%s/fabolas_%d" % (dataset, run_id)
    s_max = f.X_train.shape[0]
    s_min = 128  # Batch size

    subsets = [256] * 8
    subsets.extend([128] * 4)
    subsets.extend([64] * 2)
    subsets.extend([32] * 1)


os.makedirs(output_path, exist_ok=True)


def objective(x, s):
    dataset_fraction = s / s_max

    res = f.objective_function(x, dataset_fraction=dataset_fraction)
    return res["function_value"], res["cost"]

info = f.get_meta_information()
bounds = np.array(info['bounds'])
lower = bounds[:, 0]
upper = bounds[:, 1]
results = fabolas(objective_function=objective, lower=lower, upper=upper,
                  s_min=s_min, s_max=s_max, n_init=len(subsets), num_iterations=num_iterations,
                  n_hypers=30, subsets=subsets,
                  rng=rng, output_path=output_path, inc_estimation="last_seen")

results["run_id"] = run_id
results['X'] = results['X'].tolist()
results['y'] = results['y'].tolist()
results['c'] = results['c'].tolist()

test_error = []
current_inc = None
current_inc_val = None

key = "incumbents"

for inc in results["incumbents"]:
    print(inc)
    if current_inc == inc:
        test_error.append(current_inc_val)
    else:
        y = f.objective_function_test(inc)["function_value"]
        test_error.append(y)

        current_inc = inc
        current_inc_val = y
    print(current_inc_val)

    results["test_error"] = test_error

    with open(os.path.join(output_path, 'results_%d.json' % run_id), 'w') as fh:
        json.dump(results, fh)

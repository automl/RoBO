import os
import logging
import pickle
import numpy as np


def load_configs(path, N):
    configs = []
    learning_curves = []
    i = 0
    while len(configs) < N:
        try:
            res = pickle.load(
                open(os.path.join(path, "config_%d.pkl" % i), "rb"))
        except FileNotFoundError:
            logging.error("Config %d not found!" % i)
            i += 1
            continue

        learning_curves.append(res["learning_curve"])
        configs.append(res["config"].get_array())
        i += 1

    configs = np.array(configs)
    learning_curves = np.array(learning_curves)

    return configs, learning_curves

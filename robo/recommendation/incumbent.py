
import numpy as np


def compute_incumbent(model):
    best = np.argmin(model.Y)
    incumbent = model.X[best]
    return incumbent, best

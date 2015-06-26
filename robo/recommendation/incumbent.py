
import numpy as np


def compute_incumbent(model):
    """
        Determines the incumbent as the best configuration with the lowest observation that has been found so far
    """
    best = np.argmin(model.Y)
    incumbent = model.X[best]

    return incumbent, best

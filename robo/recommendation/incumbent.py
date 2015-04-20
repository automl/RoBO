
import numpy as np


def compute_incumbent(model):
    best = np.argmax(model.Y)
    incumbent = model.X[best]
    return incumbent

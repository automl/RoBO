import numpy as np


def projected_incumbent_estimation(model, X, proj_value=1):
    projection = np.ones([X.shape[0], 1]) * proj_value
    X_projected = np.concatenate((X, projection), axis=1)

    m, _ = model.predict(X_projected)

    best = np.argmin(m)
    incumbent = X_projected[best]
    incumbent_value = m[best]

    return incumbent, incumbent_value

import numpy as np
import logging

logger = logging.getLogger(__name__)


def joint_pmin(m, V, Nf):
    """
    Computes the probability of every given point to be the minimum
    by sampling function and count how often each point has the
    smallest function value.

    Parameters
    ----------
    M: np.ndarray(N, 1)
        Mean value of each of the N points.

    V: np.ndarray(N, N)
        Covariance matrix for all points

    Nf: int 
        Number of function samples that will be drawn at each point

    Returns
    -------
    np.ndarray(N,1)
        pmin distribution
    """
    Nb = m.shape[0]
    noise = 0
    while(True):
        try:
            cV = np.linalg.cholesky(V + noise * np.eye(V.shape[0]))
            break
        except np.linalg.LinAlgError:

            if noise == 0:
                noise = 1e-10
            if noise == 10000:
                raise np.linalg.LinAlgError('Cholesky '
                    'decomposition failed.')
            else:
                noise *= 10

    if noise > 0:
        logger.error("Add %f noise on the diagonal." % noise)
    # Draw new function samples from the innovated GP
    # on the representer points
    F = np.random.multivariate_normal(mean=np.zeros(Nb), cov=np.eye(Nb), size=Nf)
    funcs = np.dot(cV, F.T)
    funcs = funcs[:, :, None]

    m = m[:, None, :]
    funcs = m + funcs

    funcs = funcs.reshape(funcs.shape[0], funcs.shape[1] * funcs.shape[2])

    # Determine the minima for each function sample
    mins = np.argmin(funcs, axis=0)
    c = np.bincount(mins)

    # Count how often each representer point was the minimum
    min_count = np.zeros((Nb,))
    min_count[:len(c)] += c
    pmin = (min_count / funcs.shape[1])
    pmin[np.where(pmin < 1e-70)] = 1e-70

    return pmin

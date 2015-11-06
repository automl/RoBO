"""
This module contains the functions necessary to perform the sampling
in the execution of the entropy search algorithm.
Alternatively the sampling can be carried out using the emcee Python module.
"""
import logging

import numpy as np

logger = logging.getLogger(__name__)


def sample_from_measure(model, xmin, xmax, n_representers, BestGuesses, acquisition_fn):
    """
    This method corresponds to the function SampleBeliefLocations in the original ES code.

    :param model: A GP model containing the currently available information.
    :type model: GPyModel
    :param xmin: Lower bounds of the search space.
    :type xmin: np.ndarray((1.n))
    :param xmax: Upper bounds of the search space.
    :type xmin: np.ndarray((1.n))
    :param n_representers: The desired number of representer points to be returned.
    :type n_representers: int
    :param BestGuesses: An array containing the current best guesses for
                        the objective function's extremum.
    :param acquisition_fn: The acquisition function to be used in
                        the sampling of the representer points.
    :type acquisition_fn: AcquisitionFunction
    :return:
    """

    # If there are no prior observations, do uniform sampling
    if (model.X.size == 0):
        dim = xmax.size
        zb = np.add(np.multiply((xmax - xmin), np.random.uniform(size=(n_representers, dim))), xmin)
        # This is a rather ugly trick to get around the
        # different ways of filling up an array from a sampled
        # distribution Matlab and NumPy use (by columns and rows respectively):
        zb = zb.flatten().reshape((dim, n_representers)).transpose()

        mb = np.dot(-np.log(np.prod(xmax - xmin)), np.ones((n_representers, 1)))
        return zb, mb

    # There are prior observations, i.e. it's not the first ES iteration
    dim = model.X.shape[1]

    # Calculate the step size for the slice sampler
    d0 = np.divide(
        np.linalg.norm((xmax - xmin), ord=2),
        2)

    # zb will contain the sampled values:
    zb = np.zeros((n_representers, dim))
    mb = np.zeros((n_representers, 1))

    # Determine the number of batches for restarts
    numblock = np.floor(n_representers / 10.)
    restarts = np.zeros((numblock, dim))

    restarts[0:(np.minimum(numblock, BestGuesses.shape[0])), ] = \
        BestGuesses[np.maximum(BestGuesses.shape[0] - numblock + 1, 1) - 1:, ]

    restarts[(np.minimum(numblock, BestGuesses.shape[0])):numblock, ] = \
        np.add(xmin,
               np.multiply((xmax - xmin),
                           np.random.uniform(
                               size=(np.arange(np.minimum(
                                   numblock, BestGuesses.shape[0]) + 1, numblock + 1).size, dim)
               )))

    xx = restarts[0, np.newaxis]
    subsample = 20
    num_interrupts = 0
    i = 0
    while i < subsample * n_representers + 1:  # Subasmpling by a factor of 10 improves mixing
        i += 1
        if ((i - 1) % (subsample * 10) == 0) and (i / (subsample * 10.) < numblock):
            xx = restarts[i / (subsample * 10), np.newaxis]
        xx = slice_ShrinkRank_nolog(xx, acquisition_fn, d0, True)
        if i % subsample == 0:
            emb = acquisition_fn(xx)
            mb[(i / subsample) - 1, 0] = np.log(emb)
            zb[(i / subsample) - 1, ] = xx

    # Return values
    return zb, mb


def projNullSpace(J, v):
    """
    Auxiliary function for the multivariate slice sampler
    """

    if J.shape[1] > 0:
        return v - J.dot(J.transpose()).dot(v)
    else:
        return v


def slice_ShrinkRank_nolog(xx, P, s0, transpose):
    """
    Implementation of the shrinking rank slice sampler.

    :param xx: Initial points for the sampler.
    :param P: The function that will be used as sampling density.
    :param s0:
    :param transpose:
    :return:
    """

    if transpose:
        xx = xx.transpose()

    D = xx.shape[0]
    f = P(xx.transpose())

    logf = np.log(f)

    logy = np.log(np.random.uniform()) + logf

    theta = 0.95

    k = 0
    s = np.array([s0])
    c = np.zeros((D, 0))
    J = np.zeros((D, 0))
    while True:
        k += 1
        c = np.append(c, np.array(projNullSpace(J, xx + s[k - 1] * np.random.randn(D, 1))), axis=1)
        sx = np.divide(1., np.sum(np.divide(1., s)))
        mx = np.dot(
            sx,
            np.sum(
                np.multiply(
                    np.divide(1., s),
                    np.subtract(c, xx)
                ),
                1))
        xk = xx + projNullSpace(J, mx.reshape((D, 1)) +
                                np.multiply(sx, np.random.normal(size=(D, 1))))

        # TODO: add the derivative values (we're not considering them yet)
        fk, dfk = P(xk.transpose(), derivative=True)
        logfk = np.log(fk)
        dlogfk = np.divide(dfk, fk)

        if (logfk > logy).all():  # accept these values
            xx = xk.transpose()
            return xx
        else:  # shrink
            g = projNullSpace(J, dlogfk)
            if J.shape[1] < D - 1 and \
               np.dot(g.transpose(), dlogfk) > 0.5 * np.linalg.norm(g) * np.linalg.norm(dlogfk):
                J = np.append(J, np.divide(g, np.linalg.norm(g)), axis=1)
                # s[k] = s[k-1]
                s = np.append(s, s[k - 1])
            else:
                s = np.append(s, np.multiply(theta, s[k - 1]))
                if s[k] < np.spacing(1):
                    logger.error(
                        'bug found: contracted down to zero step size, still not accepted.\n')
                if transpose:
                    xx = xx.transpose()
                    return xx
                else:
                    return xx

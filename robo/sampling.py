import numpy as np

# This method corresponds to the function SampleBeliefLocations in the original ES code
# It is assumed that the GP data structure is a Python dictionary
# This function calls PI, EI etc and samples them (using their values)
def sample_from_measure(model, xmin, xmax, n_representers, BestGuesses, acquisition_fn):

    # If there are no prior observations, do uniform sampling
    if (model.X.size == 0):
        dim = xmax.size
        zb = np.add(np.multiply((xmax - xmin), np.random.uniform(size=(n_representers, dim))), xmin)
        # This is a rather ugly trick to get around the different ways of filling up an array from a sampled
        # distribution Matlab and NumPy use (by columns and rows respectively):
        zb = zb.flatten().reshape((dim, n_representers)).transpose()

        mb = np.dot(-np.log(np.prod(xmax - xmin)), np.ones((n_representers, 1)))
        return zb, mb

    # There are prior observations, i.e. it's not the first ES iteration
    dim = model.X.shape[1]

    # Calculate the step size for the slice sampler
    d0 = np.divide(
        np.linalg.norm((xmax - xmin), ord = 2),
        2)

    # zb will contain the sampled values:
    zb = np.zeros((n_representers, dim))
    mb = np.zeros((n_representers, 1))

    # Determine the number of batches for restarts
    numblock = np.floor(n_representers / 10.)
    restarts = np.zeros((numblock, dim))

    restarts[0:(np.minimum(numblock, BestGuesses.shape[0])), ] = \
        BestGuesses[np.maximum(BestGuesses.shape[0]-numblock+1, 1) - 1:, ]

    restarts[(np.minimum(numblock, BestGuesses.shape[0])):numblock, ] = \
        np.add(xmin,
               np.multiply((xmax-xmin),
                           np.random.uniform(
                               size = (np.arange(np.minimum(numblock, BestGuesses.shape[0]) + 1, numblock + 1).size, dim)
                           )))

    xx = restarts[0,np.newaxis]
    subsample = 20
    for i in range(0, subsample * n_representers + 1): # Subasmpling by a factor of 10 improves mixing
        if (i % (subsample*10) == 0) and (i / (subsample*10.) < numblock):
            xx = restarts[i/(subsample*10), np.newaxis]
        xx = slice_ShrinkRank_nolog(xx, acquisition_fn, d0, True)
        if i % subsample == 0:
            zb[(i / subsample) - 1, ] = xx
            emb = acquisition_fn(xx)
            try:

                mb[(i / subsample) - 1, 0]  = np.log(emb)
            except:
                mb[(i / subsample) - 1, 0]  = -np.inf#sys.float_info.max
                raise

    # Return values
    return zb, mb

def projNullSpace(J, v):
    # Auxiliary function for the multivariate slice sampler
    if J.shape[1] > 0:
        return v - J.dot(J.transpose()).dot(v)
    else:
        return v


def slice_ShrinkRank_nolog(xx, P, s0, transpose):
    # This function is equivalent to the similarly named function in the original ES code
    if transpose:
        xx = xx.transpose()

    # set random seed
    D = xx.shape[0]
    f = P(xx.transpose())


    try:
        logf = np.log(f)
    except:
        #print "~"*90
        logf = -np.inf#sys.float_info.max
    logy = np.log(np.random.uniform()) + logf


    theta = 0.95

    k = 0
    s = np.array([s0])
    c = np.zeros((D,0))
    J = np.zeros((D,0))
    while True:
        k += 1
        c = np.append(c, np.array(projNullSpace(J, xx + s[k-1] * np.random.randn(D,1))), axis = 1)
        sx = np.divide(1., np.sum(np.divide(1., s)))
        mx = np.dot(
            sx,
            np.sum(
                np.multiply(
                    np.divide(1., s),
                    np.subtract(c, xx)
                ),
                1))
        xk = xx + projNullSpace(J, mx.reshape((D, 1)) + np.multiply(sx, np.random.normal(size=(D,1))))

        # TODO: add the derivative values (we're not considering them yet)
        fk, dfk = P(xk.transpose(), derivative = True)

        try:
            logfk  = np.log(fk)
            dlogfk = np.divide(dfk, fk)
        except:
            logfk = - np.inf#sys.float_info.max
            dlogfk = 0

        if (logfk > logy).all(): # accept these values
            xx = xk.transpose()
            return xx
        else: # shrink
            g = projNullSpace(J, dlogfk)
            if J.shape[1] < D - 1 and \
               np.dot(g.transpose(), dlogfk) > 0.5 * np.linalg.norm(g) * np.linalg.norm(dlogfk):
                J = np.append(J, np.divide(g, np.linalg.norm(g)), axis = 1)
                # s[k] = s[k-1]
                s = np.append(s, s[k-1])
            else:
                s = np.append(s, np.multiply(theta, s[k-1]))
                if s[k] < np.spacing(1):
                    print 'bug found: contracted down to zero step size, still not accepted.\n'
                if transpose:
                    xx = xx.transpose()
                    return xx
                else:
                    return xx

def montecarlo_sampler(model, X_lower, X_upper, zb=None, Nx=20, Nf=10):
    # zb are the representer points. If they are not supplied sampling is carried out over a regular grid.
    # Nx is the "grid resolution"
    # Nf is the number of functions to sample from the gaussian process
    if zb is not None:
        xs = zb
    else:
        dim = X_lower.size
        n_bins = Nx*np.ones(dim)
        bounds = np.empty((dim, 2))
        bounds[:,0] = X_lower
        bounds[:,1] = X_upper

        xs = np.mgrid[[slice(row[0], row[1], n*1j) for row, n in zip(bounds, n_bins)]]
        xs = xs.reshape(dim,-1).T

    ys = model.sample(xs, size=Nf)

    mins = np.argmin(ys, axis=0)

    min_count = np.zeros(ys.shape)
    min_count[mins, np.arange(0, Nf)] = 1
    pmin = np.sum(min_count, axis=1) * (1. / Nf)

    return xs, pmin
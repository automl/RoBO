from scipy.stats import norm
import numpy as np
        
class PI(object):
    def __init__(self, model):
        self.model = model
    def __call__(self, X, Z=None, **kwargs):
        mean, var = self.model.predict(X, Z)
        Y_star = self.model.getCurrentBest()
        u = 1 - norm.cdf((mean - Y_star) / var)
        return u
    def model_changed(self):
        pass

class UCB(object):
    def __init__(self, model):
        self.model = model
    def __call__(self, X, Z=None, **kwargs):
        mean, var = self.model.predict(X, Z)
        return -mean + var
    def model_changed(self):
        pass

class Entropy(object):
    # This function calls PI, EI etc and samples them (using their values)
    def __init__(self, model):
        self.model = model
    def __call__(self, X, Z=None, **kwargs):
        raise NotImplementedError
    def model_changed(self):
        raise NotImplementedError

    # This method corresponds to the function SampleBeliefLocations in the original ES code
    # It is assumed that the GP data structure is a Python dictionary
    def sample_from_measure(self, gp, xmin, xmax, n_representers, BestGuesses, acquisition_fn):

        # If there are no prior observations, do uniform sampling
        if (gp['x'].size == 0):
            dim = xmax.size
            zb = np.add(np.multiply((xmax - xmin), np.random.uniform(size=(n_representers, dim))), xmin)
            mb = np.dot(-np.log(np.prod(xmax - xmin)), np.ones((n_representers, 1)))
            return zb, mb

        # There are prior observations, i.e. it's not the first ES iteration
        dim = gp['x'].shape[1]
        # print '\n' + '*'*30
        # print str(gp['x'].size)
        EI = lambda x: x

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

        # print "numblock:" + str(numblock)
        # print "restarts:\n" + str(restarts)
        #
        # print '*'*30
        #
        # print "left side: "
        # print str(restarts[0:(np.minimum(numblock, BestGuesses.shape[0])), ])
        #
        # print "right side: "
        # print str(BestGuesses[np.maximum(BestGuesses.shape[0]-numblock+1, 1) - 1:, ])
        # print str(np.maximum(BestGuesses.shape[0]-numblock+1, 1) - 1)


        ### I don't really understand what the idea behind the following two assignments is...
        restarts[0:(np.minimum(numblock, BestGuesses.shape[0])), ] = \
            BestGuesses[np.maximum(BestGuesses.shape[0]-numblock+1, 1) - 1:, ]

        restarts[(np.minimum(numblock, BestGuesses.shape[0])):numblock, ] = \
            np.add(xmin,
                   np.multiply((xmax-xmin),
                               np.random.uniform(
                                   size = (np.arange(np.minimum(numblock, BestGuesses.shape[0]) + 1, numblock + 1).size, dim)
                               )))

        xx = restarts[0, ]
        subsample = 20 # why this value?
        # print str(range(0, subsample * n_representers + 1))
        # print
        for i in range(0, subsample * n_representers + 1): # Subasmpling by a factor of 10 improves mixing (?)
            # print i,
            if (i % (subsample*10) == 0) & (i / (subsample*10.) < numblock):
                xx = restarts[i/(subsample*10), ]
                # print str(xx)
            # TODO: implement the Slice_ShrinkRank_nolog function
            # xx = Slice_ShrinkRank_nolog(xx,EI,d0,true)
            if i % subsample == 0:
                zb[(i / subsample) - 1, ] = xx
                # TODO: emb = EI(xx)
                emb = 1
                mb[(i / subsample) - 1]  = np.log(emb)

        # Return values
        return zb, mb

    def projNullSpace(self, J, v):
        # Auxiliary function for the multivariate slice sampler
        if J.shape[1] > 0:
            return v - J.dot(J.transpose()).dot(v)
        else:
            return v

    def slice_shrinkRank_nolog(self, xx, P, s0, transpose):
        # This function is equivalent to the similarly named function in the original ES code
        if transpose: # Obs: the function is only called with transpose = True in the original code
                      # also it appears that the case transpose = False leads to a runtime error
            return xx.transpose()
        else:
            raise NotImplementedError
        # D = xx.shape[0]
        # f = P(xx.transpose())[0]
        # logf = np.log(f)
        # logy = np.log(np.random.uniform()) + logf
        #
        # theta = 0.95
        #
        # k = 0
        # s = s0
        # # the following two should be empty arrays, since they appears to grow dynamically I'll just use a list for now
        # c = []
        # J = []
        #
        # while True:
        #     k += 1





    
class EI(object):
    def __init__(self, model):
        self.model = model
    def __call__(self, X, Z=None, **kwargs):
        raise NotImplementedError
    def model_changed(self):
        pass

'''
Created on 13.07.2015

@author: Aaron Klein
'''
import scipy
import emcee
import numpy as np

from robo.maximizers.base_maximizer import BaseMaximizer


class StochasticLocalSearch(BaseMaximizer):

    def __init__(self, objective_function, X_lower, X_upper,
                 Ne=20, starts=None, rng=None):

        """
        Stochastic local search.

        Parameters
        ----------
        objective_function: acquisition function
            The acquisition function which will be maximized
        X_lower: np.ndarray (D)
            Lower bounds of the input space
        X_upper: np.ndarray (D)
            Upper bounds of the input space
        Ne: int
            Determines how often the local search is repeated
        """

        if rng is None:
            self.rng = np.random.RandomState(42)
        else:
            self.rng = rng
        self.Ne = Ne
        self.starts = starts
        super(StochasticLocalSearch, self).__init__(objective_function,
                                                    X_lower, X_upper)

    def _scipy_optimizer_fkt_wrapper(self, acq_f, derivative=True):
        def _l(x, *args, **kwargs):
            x = np.array([x])
            if np.any(np.isnan(x)):
                # raise Exception("oO")

                if derivative:
                    return np.inf, np.zero_like(x)
                else:
                    return np.inf
            a = acq_f(x, derivative=derivative, *args, **kwargs)

            if derivative:
                # print -a[0][0], -a[1][0][0, :]
                return -a[0][0], -a[1][0][0, :]

            else:
                return -a[0]
        return _l

    def maximize(self):
        """
        Maximizes the given acquisition function.

        Returns
        -------
        np.ndarray(N,D)
            Point with highest acquisition value.
        """
        if hasattr(self.objective_func, "_get_most_probable_minimum"):
            xx = self.objective_func._get_most_probable_minimum()
        else:
            xx = np.add(
                np.multiply(
                    (self.X_lower - self.X_upper),
                    self.rng.uniform(
                        size=(
                            1,
                            self.X_lower.shape[0]))),
                self.X_lower)

        def fun_p(x):
            acq_v = self.objective_func(np.array([x]))[0]
            log_acq_v = np.log(acq_v) if acq_v > 0 else -np.inf

            return log_acq_v
        sc_fun = self._scipy_optimizer_fkt_wrapper(self.objective_func, False)
        S0 = 0.5 * np.linalg.norm(self.X_upper - self.X_lower)
        D = self.X_lower.shape[0]
        Xstart = np.zeros((self.Ne, D))

        restarts = np.zeros((self.Ne, D))
        if self.starts is None and hasattr(self.objective_func, "BestGuesses"):
            self.starts = self.objective_func.BestGuesses
        if self.starts is not None and self.Ne > self.starts.shape[0]:
            restarts[self.starts.shape[0]:self.Ne, ] = self.X_lower + \
                (self.X_upper - self.X_lower) * self.rng.uniform(
                    size=(self.Ne - self.starts.shape[0], D))
        elif self.starts is not None:
            restarts[0:self.Ne] = self.starts[0:self.Ne]
        else:
            restarts = self.X_lower + (self.X_upper - self.X_lower) * \
                self.rng.uniform(size=(self.Ne, D))

        sampler = emcee.EnsembleSampler(self.Ne, D, fun_p)
        Xstart, logYstart, _ = sampler.run_mcmc(restarts, 20)
        search_cons = []
        for i in range(0, self.X_lower.shape[0]):
            xmin = self.X_lower[i]
            xmax = self.X_upper[i]
            search_cons.append({'type': 'ineq',
                                'fun': lambda x: x - xmin})
            search_cons.append({'type': 'ineq',
                                'fun': lambda x: xmax - x})
        search_cons = tuple(search_cons)
        minima = []
        jacobian = False
        i = 0
        while i < self.Ne:
            # try:
            minima.append(scipy.optimize.minimize(fun=sc_fun,
                                                  x0=Xstart[i, np.newaxis],
                                                  jac=jacobian,
                                                  method='L-BFGS-B',
                                                  constraints=search_cons,
                                                  options={'ftol': np.spacing(1),
                                                           'maxiter': 20}))
            i += 1
            # no derivatives
            # except BayesianOptimizationError, e:
            #    if e.errno == BayesianOptimizationError.NO_DERIVATIVE:
            #        jacobian = False
            #        sc_fun = self._scipy_optimizer_fkt_wrapper(self.objective_func, False)
            #    else:
            #        raise e
        # X points:
        Xend = np.array([res.x for res in minima])
        # Objective function values:
        Xdh = np.array([res.fun for res in minima])
        new_x = Xend[np.nanargmin(Xdh)]
        if len(new_x.shape):
            new_x = np.array([new_x])
            return new_x

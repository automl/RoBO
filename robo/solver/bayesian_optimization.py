
import os
import time
import errno
import logging
import numpy as np
import shutil

from robo.models.GPyModelMCMC import GPyModelMCMC
from robo.models.hmc_gp import HMCGP
from robo.recommendation.optimize_posterior import optimize_posterior_mean_and_std

try:
    import cPickle as pickle
except:
    import pickle

from argparse import ArgumentError


here = os.path.abspath(os.path.dirname(__file__))


class BayesianOptimization(object):
    """
    Class implementing general Bayesian optimization.
    """
    def __init__(self, acquisition_fkt=None, model=None,
                 maximize_fkt=None, task=None, save_dir=None,
                 initialization=None, recommendation_strategy=None, num_save=1):
        """
        Initializes the Bayesian optimization.
        Either acquisition function, model, maximization function, bounds, dimensions and objective function are
        specified or an existing run can be continued by specifying only save_dir.

        :param acquisition_fkt: Any acquisition function
        :param model: A model
        :param maximize_fkt: The function for maximizing the acquisition function
        :param initialization: The initialization strategy that to find some starting points in order to train the model
        :param task: The task (derived from BaseTask) that should be optimized
        :param recommendation_strategy: A function that recommends which configuration should be return at the end
        :param save_dir: The directory to save the iterations to (or to load an existing run from)
        :param num_save: A number specifying the n-th iteration to be saved
        """

        logging.basicConfig(level=logging.DEBUG)

        self.enough_arguments = reduce(lambda a, b: a and b is not None, [True, acquisition_fkt, model, maximize_fkt, task])
        if self.enough_arguments:
            self.task = task
            self.acquisition_fkt = acquisition_fkt
            self.model = model
            self.maximize_fkt = maximize_fkt

            self.initialization = initialization

            self.X = None
            self.Y = None
            self.time_func_eval = None
            self.time_optimization_overhead = None

            self.save_dir = save_dir
            self.num_save = num_save
            if save_dir is not None:
                self.create_save_dir()

            self.model_untrained = True
            self.recommendation_strategy = recommendation_strategy
            self.incumbent = None

        elif save_dir is not None:
            self.save_dir = save_dir
            try:
                os.mkdir(self.save_dir)
            except:
                pass
        else:
            raise ArgumentError()

    def init_last_iteration(self):
        """
        Loads the last iteration from a previously stored run
        :return: the previous observations
        """
        max_iteration = self._get_last_iteration_number()

        iteration_folder = os.path.join(self.save_dir, "%03d" % (max_iteration, ))

        that = pickle.load(open(os.path.join(iteration_folder, "bayesian_opt.pickle"), "rb"))
        self.task = that.task
        self.acquisition_fkt = that.acquisition_fkt
        self.model = that.model
        self.maximize_fkt = that.maximize_fkt
        return pickle.load(open(iteration_folder + "/observations.pickle", "rb"))

    #FIXME: Update to new interface
    @classmethod
    def from_iteration(cls, save_dir, i):
        """
        Loads the data from a previous run
        :param save_dir: directory for the data
        :param i: index of iteration
        :return:
        """
        iteration_folder = save_dir + "/%03d" % (i, )
        that = pickle.load(open(iteration_folder + "/bayesian_opt.pickle", "rb"))
        if not isinstance(that, cls):
            print  "not a robo instance"
            exit()
        new_x, X, Y, best_guess = pickle.load(open(iteration_folder + "/observations.pickle", "rb"))
        return that, new_x, X, Y, best_guess

    def create_save_dir(self):
        """
        Creates the save directory to store the runs
        """
        if self.save_dir is not None:
            try:
                os.makedirs(self.save_dir)
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise

    def initialize(self):
        """
        Draws a random configuration and initializes the first data point
        """
        start_time = time.time()
        if self.initialization is None:
            # Draw one random configuration
            self.X = np.array([np.random.uniform(self.task.X_lower, self.task.X_upper, self.task.n_dims)])
            logging.info("Evaluate randomly chosen candidate %s" % (str(self.X[0])))
        else:
            logging.info("Initialize ...")
            self.X = self.initialization()
        self.time_optimization_overhead = np.array([time.time() - start_time])

        start_time = time.time()
        self.Y = self.task.evaluate(self.X)
        self.time_func_eval = np.array([time.time() - start_time])
        logging.info("Configuration achieved a performance of %f " % (self.Y[0]))
        logging.info("Evaluation of this configuration took %f seconds" % (self.time_func_eval[0]))

    def get_observations(self):
        return self.X, self.Y

    def get_model(self):
        if self.model is None:
            logging.info("No model trained yet!")
        return self.model

    def run(self, num_iterations=10, X=None, Y=None, overwrite=False):
        """
        The main Bayesian optimization loop

        :param num_iterations: number of iterations to perform
        :param X: (optional) Initial observations. If a run continues these observations will be overwritten by the load
        :param Y: (optional) Initial observations. If a run continues these observations will be overwritten by the load
        :param overwrite: data present in save_dir will be deleted and overwritten, otherwise the run will be continued.
        :return: the incumbent
        """
        # Save the time where we start the Bayesian optimization procedure
        self.time_start = time.time()

        def _onerror(dirs, path, info):
            if info[1].errno != errno.ENOENT:
                raise

        if overwrite and self.save_dir:
            shutil.rmtree(self.save_dir, onerror=_onerror)
            self.create_save_dir()
        else:
            self.create_save_dir()

        if X is None and Y is None:
            self.initialize()
            #num_iterations = num_iterations - 1
            self.incumbent = self.X[0]
            self.incumbent_value = self.Y[0]
            if self.save_dir is not None and (0) % self.num_save == 0:
                self.save_iteration(0)
        else:
            self.X = X
            self.Y = Y
            self.time_func_eval = np.zeros([self.X.shape[0]])
            self.time_optimization_overhead = np.zeros([self.X.shape[0]])

        for it in range(1, num_iterations):
            logging.info("Start iteration %d ... ", it)
            start_time = time.time()
            new_x = self.choose_next(self.X, self.Y)
            time_optimization_overhead = time.time() - start_time
            self.time_optimization_overhead = np.append(self.time_optimization_overhead, np.array([time_optimization_overhead]))
            logging.info("Optimization overhead was %f seconds" % (self.time_optimization_overhead[-1]))
            logging.info("Evaluate candidate %s" % (str(new_x)))
            start_time = time.time()
            new_y = self.task.evaluate(new_x)
            time_func_eval = time.time() - start_time
            self.time_func_eval = np.append(self.time_func_eval, np.array([time_func_eval]))
            logging.info("Configuration achieved a performance of %f " % (new_y[0, 0]))
            logging.info("Evaluation of this configuration took %f seconds" % (self.time_func_eval[-1]))

            if self.X is None:
                self.X = new_x
            else:
                self.X = np.append(self.X, new_x, axis=0)
            if self.Y is None:
                self.Y = new_y
            else:
                self.Y = np.append(self.Y, new_y, axis=0)

            if self.save_dir is not None and (it) % self.num_save == 0:
                self.save_iteration(it)

        # Recompute the incumbent before we return it
        if self.recommendation_strategy is None:
            best_idx = np.argmin(self.Y)
            self.incumbent = self.X[best_idx]
            self.incumbent_value = self.Y[best_idx]
        else:
            best_idx = np.argmin(self.Y)
            startpoint = self.X[best_idx]
            self.incumbent, self.incumbent_value = self.recommendation_strategy(self.model, self.task.X_lower, self.task.X_upper, inc=startpoint)

        logging.info("Return %s as incumbent with performance %f" % (str(self.incumbent), self.incumbent_value))
        return self.incumbent, self.incumbent_value

    def choose_next(self, X=None, Y=None):
        """
        Chooses the next configuration by optimizing the acquisition function.

        :param X: The point that have been where the objective function has been evaluated
        :param Y: The function values of the evaluated points
        :return: The next promising configuration
        """
        if X is not None and Y is not None:
            try:
                logging.info("Train model...")
                t = time.time()
                self.model.train(X, Y)
                logging.info("Time to train the model: %f", (time.time() - t))
            except Exception, e:
                logging.info("Model could not be trained", X, Y)
                raise
            self.model_untrained = False
            self.acquisition_fkt.update(self.model)

            logging.info("Determine new incumbent")

            if self.recommendation_strategy is None:
                best_idx = np.argmin(Y)
                self.incumbent = X[best_idx]
                self.incumbent_value = Y[best_idx]
            elif self.recommendation_strategy is optimize_posterior_mean_and_std:
                best_idx = np.argmin(Y)
                startpoint = X[best_idx]

                # If we do MCMC sampling over the GP hyperparameter, we optimize each model individually and return the best found point
                # TODO: Maybe we should optimize based on the average over the GPs' predictions instead of optimizing each GP individually. that would
                # prevent that we suffer return the incumbent based on a GP that is to certain in its predictions
                if isinstance(self.model, GPyModelMCMC):
                    incs = np.zeros([len(self.model.models), self.task.n_dims])
                    inc_vals = np.zeros([len(self.model.models)])
                    for i, model in enumerate(self.model.models):
                        incs[i], inc_vals[i] = self.recommendation_strategy(model, self.task.X_lower, self.task.X_upper, inc=startpoint)

                        best = np.argmin(inc_vals)
                        self.incumbent = incs[best]
                        self.incumbent_value = inc_vals[best]
                elif isinstance(self.model, HMCGP):
                    #TODO: Not clear how to compute gradients with HMCGP
                    self.incumbent, self.incumbent_value = self.recommendation_strategy(self.model, self.task.X_lower, self.task.X_upper, inc=startpoint, with_gradients=True)
                else:
                    #TODO: incumbent_value is the predicted value of the incumbent not its real value (if we optimize the posterior)
                    self.incumbent, self.incumbent_value = self.recommendation_strategy(self.model, self.task.X_lower, self.task.X_upper, inc=startpoint, with_gradients=True)
            else:
                best_idx = np.argmin(Y)
                startpoint = X[best_idx]
                self.incumbent, self.incumbent_value = self.recommendation_strategy(self.model, self.task.X_lower, self.task.X_upper, inc=startpoint)
            logging.info("New incumbent is %s", str(self.incumbent))

            logging.info("Maximize acquistion function...")
            t = time.time()
            x = self.maximize_fkt.maximize()
            logging.info("Time to maximze the acquisition function: %f", (time.time() - t))
        else:
            self.initialize()
            x = self.X
        return x

    def save_iteration(self, it):
        """
            Saves an iteration.
        """
        file_name = "iteration_%03d.pkl" % (it)

        file_name = os.path.join(self.save_dir, file_name)
        #os.makedirs(iteration_folder)
        #FIXME What does Entropy return as incumbent?
        #if hasattr(self.acquisition_fkt, "_get_most_probable_minimum") and not self.model_untrained:
        #    pickle.dump([self.X, self.Y, self.acquisition_fkt._get_most_probable_minimum()[0], self.time_func_eval, self.time_optimization_overhead], open(file_name, "w"))
        #else:
        d = dict()
        d['X'] = self.X
        d['Y'] = self.Y
        d['incumbent'] = self.incumbent
        d['incumbent_value'] = self.incumbent_value
        d['time_function_eval'] = self.time_func_eval
        d['time_optimization_overhead'] = self.time_optimization_overhead
        d['all_time'] = time.time() - self.time_start
        pickle.dump(d, open(file_name, "w"))

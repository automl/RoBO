
import os
import time
import errno
import numpy as np

import shutil
try:
    import cPickle as pickle
except:
    import pickle
from robo.util.exc import BayesianOptimizationError
from argparse import ArgumentError

here = os.path.abspath(os.path.dirname(__file__))


class BayesianOptimization(object):
    """
    Class implementing general Bayesian optimization.
    """
    def __init__(self, acquisition_fkt=None, model=None,
                 maximize_fkt=None, task=None, save_dir=None, initialization=None, num_save=1):
        """
        Initializes the Bayesian optimization.
        Either acquisition function, model, maximization function, bounds, dimensions and objective function are
        specified or an existing run can be continued by specifying only save_dir.

        :param acquisition_fkt: Any acquisition function
        :param model: A model
        :param maximize_fkt: The function for maximizing the acquisition function
        :param initialization: The initialization strategy that to find some starting points in order to train the model
        :param task: The task (derived from BaseTask) that should be optimized
        :param objective_fkt: The objective function to execute in each step
        :param save_dir: The directory to save the iterations to (or to load an existing run from)
        :param num_save: A number specifying the n-th iteration to be saved
        """

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
            self.recommendation_strategy = None
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
            raise BayesianOptimizationError(BayesianOptimizationError.LOAD_ERROR, "not a robo instance")
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
            self.X = np.array([np.random.uniform(self.task.X_lower, self.task.X_upper, self.task.dims)])
            print "Evaluate randomly chosen candidate %s" % (str(self.X[0]))
        else:
            print "Initialize ..."
            self.X = self.initialization()
        self.time_optimization_overhead = np.array([time.time() - start_time])

        start_time = time.time()
        self.Y = self.objective_fkt(self.X)
        self.time_func_eval = np.array([time.time() - start_time])
        print "Configuration achieved a performance of %f " % (self.Y[0])

    def get_observations(self):
        return self.X, self.Y

    def get_model(self):
        if self.model is None:
            print "No model trained yet!"
        return self.model

#     def iterate(self, save_it=False):
#         """
#         Performs one iteration
#         :param save_it: If true, the iteration is saved (only if the save_dir is configured)
#         """
#         print "Choose a new configuration"
#         new_x = self.choose_next(self.X, self.Y)
#         print "Evaluate candidate %s" % (str(new_x))
#         new_y = self.objective_fkt(new_x)
#         print "Configuration achieved a performance of %d " % (new_y[0, 0])
#         if self.X is None:
#             self.X = new_x
#         else:
#             self.X = np.append(self.X, new_x, axis=0)
#         if self.Y is None:
#             self.Y = new_y
#         else:
#             self.Y = np.append(self.Y, new_y, axis=0)
# 
#         if self.save_dir is not None and save_it:
#             self.save_iteration(self.X, self.Y, new_x)

    def run(self, num_iterations=10, X=None, Y=None, overwrite=False):
        """
        Runs iterations.

        :param num_iterations: number of iterations to perform
        :param X: (optional) Initial observations. If a run continues these observations will be overwritten by the load
        :param Y: (optional) Initial observations. If a run continues these observations will be overwritten by the load
        :param overwrite: data present in save_dir will be deleted and overwritten, otherwise the run will be continued.
        :return: the incumbent
        """
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
            num_iterations = num_iterations - 1
        else:
            self.X = X
            self.Y = Y
            self.time_func_eval = np.zeros([self.X.shape[0]])
            self.time_optimization_overhead = np.zeros([self.X.shape[0]])

        for it in range(num_iterations):
            print "Choose a new configuration"
            start_time = time.time()
            new_x = self.choose_next(self.X, self.Y)
            time_optimization_overhead = time.time() - start_time
            self.time_optimization_overhead = np.append(self.time_func_eval, np.array([time_optimization_overhead]))

            print "Evaluate candidate %s" % (str(new_x))
            start_time = time.time()
            new_y = self.objective_fkt(new_x)
            time_func_eval = time.time() - start_time
            self.time_func_eval = np.append(self.time_func_eval, np.array([time_func_eval]))
            print "Configuration achieved a performance of %d " % (new_y[0, 0])

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
            self.incumbent, self.incumbent_value = self.recommendation_strategy(self.model, self.acquisition_fkt)

        print "Return %s as incumbent with performance %f" % (str(self.incumbent), self.incumbent_value)
        return self.incumbent, self.incumbent_value

    def choose_next(self, X=None, Y=None):
        """
        Chooses the next configuration by optimizing the acquisition function.

        :param X: The X for the model
        :param Y: The Y for the model
        :return: The next desired configuration
        """
        if X is not None and Y is not None:
            try:
                self.model.train(X, Y)
            except Exception, e:
                print "Model could not be trained", X, Y
                raise
            self.model_untrained = False
            self.acquisition_fkt.update(self.model)

            if self.recommendation_strategy is None:
                best_idx = np.argmin(Y)
                self.incumbent = X[best_idx]
                self.incumbent_value = Y[best_idx]
            else:
                self.incumbent, self.incumbent_value = self.recommendation_strategy(self.model, self.acquisition_fkt)
            x = self.maximize_fkt(self.acquisition_fkt, self.task.X_lower, self.task.X_upper)
        else:
            self.initialize()
            x = self.X
        return x

#     def _get_last_iteration_number(self):
#         max_iteration = 0
#         for i in os.listdir(self.save_dir):
#             try:
#                 it_num = int(i)
#                 if it_num > max_iteration:
#                     max_iteration = it_num
#             except Exception, e:
#                 print e
#         return max_iteration

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
        pickle.dump([self.X, self.Y, self.incumbent, self.incumbent_value, self.time_func_eval, self.time_optimization_overhead], open(file_name, "w"))

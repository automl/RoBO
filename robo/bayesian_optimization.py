
import os
import errno
import numpy as np

import shutil
try:
    import cpickle as pickle
except:
    import pickle
from robo.util.exc import BayesianOptimizationError

here = os.path.abspath(os.path.dirname(__file__))


class BayesianOptimization(object):
    """
        save_dir:
            save to save_dir after each iteration
    """
    def __init__(self, acquisition_fkt=None, model=None,
                 maximize_fkt=None, X_lower=None, X_upper=None, dims=None,
                 objective_fkt=None, save_dir=None, num_save=1, initialization=None):

        self.enough_arguments = reduce(lambda a, b: a and b is not None, [True, acquisition_fkt, model, maximize_fkt, X_lower, X_upper, dims])
        if self.enough_arguments:
            self.objective_fkt = objective_fkt
            self.acquisition_fkt = acquisition_fkt
            self.model = model
            self.maximize_fkt = maximize_fkt
            self.X_lower = X_lower
            self.X_upper = X_upper
            self.dims = dims

            self.initialization = initialization

            self.X = None
            self.Y = None

            self.save_dir = save_dir
            self.num_save = num_save
            if save_dir is not None:
                self.create_save_dir()

            self.model_untrained = True
            self.recommendation_strategy = None
            self.incumbent = None

        elif save_dir is not None:
            self.save_dir = save_dir
        else:
            raise ArgumentError()

    def init_last_iteration(self):
        max_iteration = self._get_last_iteration_number()

        iteration_folder = os.path.join(self.save_dir, "%03d" % (max_iteration, ))

        that = pickle.load(open(os.path.join(iteration_folder, "bayesian_opt.pickle"), "rb"))
        self.objective_fkt = that.objective_fkt
        self.acquisition_fkt = that.acquisition_fkt
        self.model = that.model
        self.maximize_fkt = that.maximize_fkt
        self.X_lower = that.X_lower
        self.X_upper = that.X_upper
        self.dims = that.dims
        return pickle.load(open(iteration_folder + "/observations.pickle", "rb"))

    @classmethod
    def from_iteration(cls, save_dir, i):
        iteration_folder = save_dir + "/%03d" % (i, )
        that = pickle.load(open(iteration_folder + "/bayesian_opt.pickle", "rb"))
        if not isinstance(that, cls):
            raise BayesianOptimizationError(BayesianOptimizationError.LOAD_ERROR, "not a robo instance")
        new_x, X, Y, buest_guess = pickle.load(open(iteration_folder + "/observations.pickle", "rb"))
        return that, new_x, X, Y, buest_guess

    def create_save_dir(self):
        if self.save_dir is not None:
            try:
                os.makedirs(self.save_dir)
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise

    def initialize(self):
        if self.initialization is None:
            # Draw one random configuration
            self.X = np.array([np.random.uniform(self.X_lower, self.X_upper, self.dims)])
            print "Evaluate randomly chosen candidate %s" % (str(self.X[0]))
        else:
            print "Initialize ..."
            self.X = self.initialization()

        self.Y = self.objective_fkt(self.X)

    def get_observations(self):
        return self.X, self.Y

    def get_model(self):
        if self.model is None:
            print "No model trained yet!"
        return self.model

    def run(self, num_iterations=10, X=None, Y=None, overwrite=False):
        """
        overwrite:
            True: data present in save_dir will be deleted.
            False: data present will be loaded an the run will continue
        X, Y:
            Initial observations. They are optional. If a run continues
            these observations will be overwritten by the load
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
            # TODO: allow different initialization strategies here
            self.initialize()
            num_iterations = num_iterations - 1
        else:
            self.X = X
            self.Y = Y

        for it in range(num_iterations):
            print "Choose a new configuration"
            new_x = self.choose_next(self.X, self.Y)
            print "Evaluate candidate %s" % (str(new_x))
            new_y = self.objective_fkt(new_x)
            print "Configuration achieved a performance of %d " % (new_y[0, 0])
            self.X = np.append(self.X, new_x, axis=0)
            self.Y = np.append(self.Y, new_y, axis=0)

            if self.save_dir is not None and (it) % self.num_save == 0:
                self.save_iteration(self.X, self.Y, new_x)

        # Recompute the incumbent before we return it
        if self.recommendation_strategy is None:
            best_idx = np.argmin(self.Y)
            self.incumbent = self.X[best_idx]
        else:
            self.incumbent = self.recommendation_strategy(self.model, self.acquisition_fkt)

        print "Return %s as incumbent" % (str(self.incumbent))
        return self.incumbent

    def choose_next(self, X, Y):
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
        else:
            self.incumbent = self.recommendation_strategy(self.model, self.acquisition_fkt)

        x = self.maximize_fkt(self.acquisition_fkt, self.X_lower, self.X_upper)
        return x

    def _get_last_iteration_number(self):
        max_iteration = 0
        for i in os.listdir(self.save_dir):
            try:
                it_num = int(i)
                if it_num > max_iteration:
                    max_iteration = it_num
            except Exception, e:
                print e
        return max_iteration

    def save_iteration(self, X, Y, new_x):
        max_iteration = self._get_last_iteration_number()
        iteration_folder = self.save_dir + "/%03d" % (max_iteration + 1, )
        #pickle.dump(self, open(iteration_folder+"/bayesian_opt.pickle", "w"))
        os.makedirs(iteration_folder)
        if hasattr(self.acquisition_fkt, "_get_most_probable_minimum") and not self.model_untrained:
            pickle.dump([new_x, X, Y, self.acquisition_fkt._get_most_probable_minimum()[0]], open(iteration_folder + "/observations.pickle", "w"))
        else:
            pickle.dump([new_x, X, Y, self.model.getCurrentBestX()], open(iteration_folder + "/observations.pickle", "w"))

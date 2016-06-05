'''
Created on Aug 21, 2015

@author: Aaron Klein
'''

import os
import csv
import time
import errno
import logging
import json

logger = logging.getLogger(__name__)


class BaseSolver(object):

    def __init__(self, acquisition_func=None, model=None,
                 maximize_func=None, task=None, save_dir=None):
        """
        Base class which specifies the interface for solvers. Derive from
        this class if you implement your own solver.

        Parameters
        ----------
        acquisition_func: BaseAcquisitionFunction Object
            The acquisition function which will be maximized.
        model: ModelObject
            Model (i.e. GaussianProcess, RandomForest) that models our current
            believe of the objective function.
        task: TaskObject
            Task object that contains the objective function and additional
            meta information such as the lower and upper bound of the search
            space.
        maximize_func: MaximizerObject
            Optimization method that is used to maximize the acquisition
            function
        save_dir: String
            Output path
        """

        self.model = model
        self.acquisition_func = acquisition_func
        self.maximize_func = maximize_func
        self.task = task
        self.save_dir = save_dir
        if self.save_dir is not None:
            self.create_save_dir()

    def create_save_dir(self):
        """
        Creates the save directory to store the runs
        """
        try:
            os.makedirs(self.save_dir)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
        self.output_file = open(os.path.join(self.save_dir, 'results.csv'), 'w')
        self.output_file_json = open(os.path.join(self.save_dir,'results.json'),'w')
        self.csv_writer = None
        self.json_writer = None

    def get_observations(self):
        return self.X, self.Y

    def get_model(self):
        if self.model is None:
            logger.info("No model trained yet!")
        return self.model

    def run(self, num_iterations=10, X=None, Y=None, overwrite=False):
        """
        The main optimization loop

        Parameters
        ----------
        num_iterations: int
            The number of iterations
        X: np.ndarray(N,D)
            Initial points that are already evaluated
        Y: np.ndarray(N,1)
            Function values of the already evaluated points

        Returns
        -------
        np.ndarray(1,D)
            Incumbent
        np.ndarray(1,1)
            (Estimated) function value of the incumbent
        """
        pass

    def choose_next(self, X=None, Y=None):
        """
        Suggests a new point to evaluate.

        Parameters
        ----------
        num_iterations: int
            The number of iterations
        X: np.ndarray(N,D)
            Initial points that are already evaluated
        Y: np.ndarray(N,1)
            Function values of the already evaluated points

        Returns
        -------
        np.ndarray(1,D)
            Suggested point
        """
        pass

    def save_iteration(self, it, **kwargs):
        """
        Saves the meta information of an iteration.
        """
        if self.csv_writer is None:
            self.fieldnames = ['iteration', 'config', 'fval',
                               'incumbent', 'incumbent_val',
                               'time_func_eval', 'time_overhead', 'runtime']

            for key in kwargs:
                self.fieldnames.append(key)
            self.csv_writer = csv.DictWriter(self.output_file,
                                             fieldnames=self.fieldnames)
            self.csv_writer.writeheader()

        output = dict()
        output["iteration"] = it
        output['config'] = self.X[it]
        output['fval'] = self.Y[it]
        output['incumbent'] = self.incumbent
        output['incumbent_val'] = self.incumbent_value
        output['time_func_eval'] = self.time_func_eval[it]
        output['time_overhead'] = self.time_overhead[it]
        output['runtime'] = time.time() - self.time_start

        if kwargs is not None:
            for key, value in kwargs.items():
                output[key] = str(value)

        self.csv_writer.writerow(output)
        self.output_file.flush()


    def get_json_data(self, it):
        """
        Json getter function

        :return: dict() object
        """

        jsonData = dict()
        jsonData = {"optimization_overhead":self.time_overhead[it], "runtime": time.time() - self.time_start,
                    "incumbent": self.incumbent.tolist(), "incumbent_fval":self.incumbent_value.tolist(),"time_func_eval": self.time_func_eval[it],
                    "iteration":it}
        return jsonData

    def save_json(self, it, **kwargs):
        """
        Saves meta information of an iteration in a Json file.
        """
        base_solver_data =self.get_json_data(it)
        base_model_data = self.model.get_json_data()
        base_task_data = self.task.get_json_data()
        base_acquisition_data = self.acquisition_func.get_json_data()

        data = [base_solver_data,
                base_model_data,
                base_task_data,
                base_acquisition_data
                ]

        for j in data:
            json.dump(j, self.output_file_json)
            self.output_file_json.write('\n')
        self.output_file_json.write('\n')  #Json more readable. Drop it?
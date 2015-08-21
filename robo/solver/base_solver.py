'''
Created on Aug 21, 2015

@author: Aaron Klein
'''

import os
import csv
import time
import errno
import logging


class BaseSolver(object):
    '''
    classdocs
    '''

    def __init__(self, acquisition_fkt=None, model=None,
                 maximize_fkt=None, task=None, save_dir=None):
        '''
        Constructor
        '''
        self.model = model
        self.acquisition_fkt = acquisition_fkt
        self.maximize_fkt = maximize_fkt
        self.task = task

        if save_dir is not None:
            self.save_dir = save_dir
            try:
                os.mkdir(self.save_dir)
            except:
                pass
            self.output_file = open(os.path.join(save_dir, 'results.csv'), 'wb')
            self.csv_writer = None

    def init_last_iteration(self):
        """
        Loads the last iteration from a previously stored run
        :return: the previous observations
        """
        raise("Not yet implemented")

    def from_iteration(self, save_dir, i):
        """
        Loads the data from a previous run
        :param save_dir: directory for the data
        :param i: index of iteration
        :return:
        """
        raise("Not yet implemented")

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
        pass

    def choose_next(self, X=None, Y=None):
        """
        Chooses the next configuration by optimizing the acquisition function.

        :param X: The point that have been where the objective function has been evaluated
        :param Y: The function values of the evaluated points
        :return: The next promising configuration
        """
        pass

    def save_iteration(self, it, **kwargs):
        """
            Saves an iteration.
        """

        if self.csv_writer is None:
            self.fieldnames = ["iteration", "config", "fval", "incumbent", "incumbent_val", "time_func_eval", "time_overhead", "runtime"]
            for key in kwargs:
                    self.fieldnames.append(key)
            self.csv_writer = csv.DictWriter(self.output_file, fieldnames=self.fieldnames)
            self.csv_writer.writeheader()

        output = dict()
        output["iteration"] = str(it)
        output['config'] = str(self.X[-1])
        output['fval'] = str(self.Y[-1])
        output['incumbent'] = str(self.incumbent)
        output['incumbent_val'] = str(self.incumbent_value)
        output['time_func_eval'] = str(self.time_func_eval[-1])
        output['time_overhead'] = str(self.time_optimization_overhead[-1])
        output['runtime'] = str(time.time() - self.time_start)

        if kwargs is not None:
            for key, value in kwargs.iteritems():
                output[key] = str(value)

        self.csv_writer.writerow(output)
        self.output_file.flush()

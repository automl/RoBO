# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 17:00:02 2015

@author: aaron
"""
import csv
import numpy as np


class OutputReader(object):

    def __init__(self):
        pass

    def read_results_file(self,filename,
                          fieldnames=['iteration', 'config', 'fval',
                                        'incumbent', 'incumbent_val',
                                        'time_func_eval', 'time_overhead',
                                        'runtime']):

        csv_file = open(filename, "r")
        reader = csv.DictReader(csv_file)

        output = dict()
        for field in reader.fieldnames:
            output[field] = []
        for row in reader:
            for field in row:
                try:
                    output[field].append(float(row[field]))
                except ValueError:
                    output[field].append(np.fromstring(row[field].strip('[').strip(']'), sep=' '))

        return output

    def read_experiment(self, experiment_dir):
        """
            Read mutliple runs of an experiment.
            Assumes that experiment_dir contains multiple subdirectories where each represents
            one single run of the experiment.
        """
        pass

    def read_multiple_experiments(self, experiment_dirs):
        pass

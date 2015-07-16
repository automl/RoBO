'''
Created on 13.07.2015

@author: Aaron Klein
'''


class BaseMaximizer(object):
    '''
    classdocs
    '''

    def __init__(self, objective_function, X_lower, X_upper):
        '''
        Constructor
        '''
        self.X_lower = X_lower
        self.X_upper = X_upper
        self.objective_func = objective_function

    def maximize(self):
        pass
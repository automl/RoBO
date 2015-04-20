class BayesianOptimizationError(Exception):
    LOAD_ERROR = 1
    SINGLE_INPUT_ONLY = 2
    NO_DERIVATIVE = 3
    def __init__(self, errno, *args, **kwargs):
        self.errno = errno
        Exception.__init__(self,*args, **kwargs)
        
class History(object):
    def __init__(self, observations = None, models = None, acquisition_fkt = None):
        self.observations = observations
        self.models = models
        self.acquisition_fkt = acquisition_fkt
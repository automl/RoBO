import time
import numpy as np
from sklearn import svm

from robo.task.base_task import BaseTask


class SupportVectorMachineTask(BaseTask):

    def __init__(self, train, train_targets,
                     valid, valid_targets,
                     test, test_targets,
                 multi_task=False, with_costs=False, fabolas_task=False):
        """
        Hyperparameter optimization task to optimize the regularization
        parameter C and the kernel parameter gamma of a support vector machine.
        Both hyperparameters are optimized on a log scale [-10, 10].
        
        The test dataset is only used for a final offline evaluation of 
        a configuration. For that the validation and training data is
        concatenated to form the whole training dataset.
        
        MultiTaskBO: 1/4 of the training data is used for the auxillary task
        Fabolas: The dataset size s is also optimized on a
                log scale [s_min, s_max], where s_min is 10 * the number of 
                classes and s_max is the number of datapoints of the whole
                training dataset.
        
        Parameters
        ----------
        train : np.ndarray(N, D)
            Training data
        train_targets : np.ndarray(N, 1)
            Labels of the training data
        valid : np.ndarray(N, D)
            Validation data that is used to optimize the hyperparameters
        valid targets: np.ndarray(N, 1)
            Labels of the validation data
        test : np.ndarray(N, D)
            Test data that is used for an offline evaluation of a configuration
        test targets: np.ndarray(N, 1)
            Labels of the test data
        multi_task: bool
            True means that we consider the multitask BO case with one
            auxillary task and one primary task
        fabolas_task: bool
            If true than we optimize across different subsets. See the paper
            for more explaination.
        with_costs: bool
            If true, than also the the time (seconds) that was needed for
            evaluation is returned.

        """
        X_lower = np.array([-10, -10])
        X_upper = np.array([10, 10])

        self.multi_task = multi_task
        self.with_costs = with_costs
        self.fabolas_task = fabolas_task
        self.train = train
        self.train_targets = train_targets
        self.valid = valid
        self.valid_targets = valid_targets
        self.test = test
        self.test_targets = test_targets        
        
        if self.fabolas_task:            
            #Use 10 time the number of classes as lower bound
            self.n_classes = np.unique(self.train_targets).shape[0]
    
            self.s_min = np.log(10 * self.n_classes)
            self.s_max = np.log(self.train_targets.shape[0])            
            X_lower = np.concatenate((X_lower,
                                  np.array([self.s_min])))
            X_upper = np.concatenate((X_upper,
                                  np.array([self.s_max])))
            self.is_env = np.array([0, 0, 1])

        if self.multi_task:
            # Add dimension for the tasks
            X_lower = np.concatenate((X_lower,
                                  np.array([0])))
            X_upper = np.concatenate((X_upper,
                                  np.array([1])))
            # The auxillary task consists of 1/4 of the original data set
            self.auxillary_task_size = int(self.train.shape[0] * 0.25)
            self.is_env = np.array([0, 0, 1])
        super(SupportVectorMachineTask, self).__init__(X_lower, X_upper)

    def objective_function(self, x):
        start_time = time.time()
        if self.multi_task:
            # Evaluate the config on the auxillary task
            if np.round(x[0, -1]) == 0:

                train = self.train[:self.auxillary_task_size]
                train_targets = self.train_targets[:self.auxillary_task_size]

                err = self._train_and_validate(x[:, :2], train, train_targets,
                                  self.valid, self.valid_targets)
            # Evaluate on whole data set
            elif np.round(x[0, -1]) == 1:
                err = self._train_and_validate(x[:, :2], self.train, self.train_targets,
                                  self.valid, self.valid_targets)

        elif self.fabolas_task:
            
            size = int(np.exp(x[0, -1]))
            shuffle = np.random.permutation(np.arange(int(np.exp(self.s_max))))

            train = self.train[shuffle[:size]]
            train_targets = self.train_targets[shuffle[:size]]

            i = 0
            # Check if we have a sample of each class in the subset
            while True:
                if (np.unique(train_targets).shape[0] == self.n_classes):
                    break
                shuffle = np.random.permutation(np.arange(int(np.exp(self.s_max))))
                train = self.train[shuffle[:size]]
                train_targets = self.train_targets[shuffle[:size]]
                i += 1
                # Sanity check if we can actually find a valid shuffled split
                if i == 20:
                    raise("Couldn't find a valid split that contains a \
                    sample from each class after 20 iterations. \
                    Maybe increase your bounds!")

            err = self._train_and_validate(x[:, :2], train, train_targets,
                                  self.valid, self.valid_targets)            
                                  
        else:
            err = self._train_and_validate(x, self.train, self.train_targets,
                                  self.valid, self.valid_targets)
                                  
        if self.with_costs:
            t = time.time() - start_time
            return err, np.array([[t]])
        else:
            return err

    def _train_and_validate(self, x, train, train_targets, valid, valid_targets):
        C = np.exp(float(x[0, 0]))
        gamma = np.exp(float(x[0, 1]))

        clf = svm.SVC(gamma=gamma, C=C)

        clf.fit(train, train_targets)
        y = 1 - clf.score(valid, valid_targets)
        y = np.log(y)
        return np.array([[y]])

    def objective_function_test(self, x):
        if self.multi_task:
            x_ = x[:, :2]
        elif self.fabolas_task:
            x_ = x[:, :2]
        else:
            x_ = x            
        
        # Concatenate training and validation data
        train = np.concatenate((self.train, self.valid), axis=0)
        train_targets = np.concatenate((self.train_targets,
                                        self.valid_targets), axis=0)
        return self._train_and_validate(x_, train, train_targets,
                                        self.test, self.test_targets)


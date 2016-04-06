import time
import lasagne
import numpy as np
import theano
import theano.tensor as T

from robo.task.base_task import BaseTask
from model_zoo import batch_norm


class ConvNetTask2D(BaseTask):

    def __init__(self, train, train_targets,
                 valid, valid_targets,
                 test, test_targets,
                 do_enves=False,
                 multitask=False,
                 with_costs=False):

        self.do_enves = do_enves
        self.multitask = multitask
        self.with_costs = with_costs

        self.conv_net_task = ConvNetTask(train, train_targets,
                                         valid, valid_targets,
                                         test, test_targets,
                                         do_enves, multitask,
                                         with_costs)

        self.params = [0.150525, 0.150525, 0.660429259851807,
               0.103845915430237,  0.856603994754596, 0.610192096865752,
               0.15057682025474, 0.402569279425545, 0.207173507177861,
               0.70727051556196, 0.632370812824289, 0.989960548851386,
               0.786357776683791, 0.469491216180133, 0.844823856839956,
               0.282132332872129, 0.48647866537331, 0.714407065743167,
               0.241739474196816, 0.825379160472778, 0.368644452393994]

        X_lower = np.array([-0.3, -3])
        X_upper = np.array([3, 1])

        if do_enves:
            self.s_min = np.log(1000)
            self.s_max = np.log(train.shape[0])

            super(ConvNetTask2D, self).__init__(
                            X_lower=np.append(X_lower, np.array([self.s_min])),
                            X_upper=np.append(X_upper, np.array([self.s_max])))
            self.is_env = np.array([0, 0, 1])
        elif multitask:
            self.s_min = np.log(train.shape[0] * 0.25)  # 1/4 s_max
            self.s_max = np.log(train.shape[0])
            super(ConvNetTask2D, self).__init__(
                            X_lower=np.append(X_lower, np.array([self.s_min])),
                            X_upper=np.append(X_upper, np.array([self.s_max])))
            self.is_env = np.array([0, 0, 1])
        else:
            super(ConvNetTask2D, self).__init__(X_lower=X_lower,
                                      X_upper=X_upper)

    def objective_function(self, x):

        LR_start = x[0, 0]
        betaval = x[0, 1]

        alpha = self.params[5]
        epsilon = self.params[6]
        pdrop1 = self.params[7]
        pdrop2 = self.params[8]
        pdrop3 = self.params[9]
        pdrop4 = self.params[19]
        nfilters1 = self.params[10]
        nfilters2 = self.params[11]
        nfilters3 = self.params[12]
        nfilters4 = self.params[20]
        LR_fin = self.params[14]
        epsval = self.params[16]
        beta2val = self.params[17]
        num_epochs_adapt = self.params[18]

        if self.do_enves or self.multitask:
            size = x[0, 2]
            x_ = np.array([[LR_start, LR_fin, betaval, beta2val,
                        epsval, num_epochs_adapt,
                        alpha, epsilon,
                        pdrop1, pdrop2, pdrop3, pdrop4,
                        nfilters1, nfilters2, nfilters3, nfilters4,
                        size]])

        else:
            x_ = np.array([[LR_start, LR_fin, betaval, beta2val,
                        epsval, num_epochs_adapt,
                        alpha, epsilon,
                        pdrop1, pdrop2, pdrop3, pdrop4,
                        nfilters1, nfilters2, nfilters3, nfilters4]])

        res = self.conv_net_task.objective_function(x_)
        self.lcurve = self.conv_net_task.lcurve
        return res

    def objective_function_test(self, x):
        self.objective_function(x)
        return self.conv_net_task.test_error


class ConvNetTask6D(BaseTask):

    def __init__(self, train, train_targets,
                 valid, valid_targets,
                 test, test_targets,
                 do_enves=False,
                 multitask=False,
                 with_costs=False):

        self.do_enves = do_enves
        self.multitask = multitask
        self.with_costs = with_costs

        self.conv_net_task = ConvNetTask(train, train_targets,
                                         valid, valid_targets,
                                         test, test_targets,
                                         do_enves, multitask,
                                         with_costs)

        self.params = [0.150525, 0.150525, 0.660429259851807,
               0.103845915430237,  0.856603994754596, 0.610192096865752,
               0.15057682025474, 0.402569279425545, 0.207173507177861,
               0.70727051556196, 0.632370812824289, 0.989960548851386,
               0.786357776683791, 0.469491216180133, 0.844823856839956,
               0.282132332872129, 0.48647866537331, 0.714407065743167,
               0.241739474196816, 0.825379160472778, 0.368644452393994]

        #LR_start, LR_fin, nfilters1, nfilters2, nfilters3, nfilters4
        X_lower = np.array([0, 0, 0, 0, 0, 0])
        X_upper = np.array([2, 3, 1, 1, 1, 1])

        if do_enves:
            self.s_min = np.log(1000)
            self.s_max = np.log(train.shape[0])

            super(ConvNetTask6D, self).__init__(
                            X_lower=np.append(X_lower, np.array([self.s_min])),
                            X_upper=np.append(X_upper, np.array([self.s_max])))
            self.is_env = np.array([0, 0, 0, 0, 0, 0, 1])
        elif multitask:
            self.s_min = np.log(train.shape[0] * 0.25)  # 1/4 s_max
            self.s_max = np.log(train.shape[0])
            super(ConvNetTask6D, self).__init__(
                            X_lower=np.append(X_lower, np.array([self.s_min])),
                            X_upper=np.append(X_upper, np.array([self.s_max])))
            self.is_env = np.array([0, 0, 0, 0, 0, 0, 1])
        else:
            super(ConvNetTask6D, self).__init__(X_lower=X_lower,
                                      X_upper=X_upper)

    def objective_function(self, x):

        LR_start = x[0, 0]
        LR_fin = x[0, 1]
        nfilters1 = x[0, 2]
        nfilters2 = x[0, 3]
        nfilters3 = x[0, 4]
        nfilters4 = x[0, 5]

        alpha = self.params[5]
        epsilon = self.params[6]

        beta2val = self.params[17]
        betaval = self.params[15]

        pdrop1 = self.params[7]
        pdrop2 = self.params[8]
        pdrop3 = self.params[9]
        pdrop4 = self.params[19]
#         nfilters1 = self.params[10]
#         nfilters2 = self.params[11]
#         nfilters3 = self.params[12]
#         nfilters4 = self.params[20]
        epsval = self.params[16]
        num_epochs_adapt = self.params[18]

        if self.do_enves or self.multitask:
            size = x[0, -1]
            x_ = np.array([[LR_start, LR_fin, betaval, beta2val,
                        epsval, num_epochs_adapt,
                        alpha, epsilon,
                        pdrop1, pdrop2, pdrop3, pdrop4,
                        nfilters1, nfilters2, nfilters3, nfilters4,
                        size]])

        else:
            x_ = np.array([[LR_start, LR_fin, betaval, beta2val,
                        epsval, num_epochs_adapt,
                        alpha, epsilon,
                        pdrop1, pdrop2, pdrop3, pdrop4,
                        nfilters1, nfilters2, nfilters3, nfilters4]])

        res = self.conv_net_task.objective_function(x_)
        self.lcurve = self.conv_net_task.lcurve
        return res

    def objective_function_test(self, x):
        self.objective_function(x)
        return self.conv_net_task.test_error


class ConvNetTask(BaseTask):

    def __init__(self, train, train_targets,
                 valid, valid_targets,
                 test, test_targets,
                 do_enves=False,
                 multitask=False,
                 with_costs=False):

        self.X_train = train
        self.y_train = train_targets
        self.X_valid = valid
        self.y_valid = valid_targets
        self.X_test = test
        self.y_test = test_targets

        self.input_var = T.tensor4('inputs')
        self.target_var = T.ivector('targets')

        self.do_enves = do_enves
        self.multitask = multitask
        self.with_costs = with_costs

        self.n_classes = np.unique(self.y_train).shape[0]

        self.input_shape = (3, 32, 32)
        self.num_epochs = 20
        self.batch_size = 500

        X_lower = np.zeros([16])
        X_upper = np.ones([16])

        if do_enves:
            self.s_min = np.log(1000)
            self.s_max = np.log(self.X_train.shape[0])

            super(ConvNetTask, self).__init__(
                            X_lower=np.append(X_lower, np.array([self.s_min])),
                            X_upper=np.append(X_upper, np.array([self.s_max])))
            self.is_env = np.append(X_lower, np.array([1]))
        elif multitask:
            self.s_min = np.log(11250)  # 1/4 s_max
            self.s_max = np.log(self.X_train.shape[0])
            super(ConvNetTask, self).__init__(
                            X_lower=np.append(X_lower, np.array([self.s_min])),
                            X_upper=np.append(X_upper, np.array([self.s_max])))
            self.is_env = np.append(X_lower, np.array([1]))
        else:
            super(ConvNetTask, self).__init__(X_lower=X_lower,
                                      X_upper=X_upper)

    def _build_cnn(self, alpha=.15, epsilon=1e-4,
               nf1=128, nf2=128, nf3=128, nf4=128,
               pdrop1=0.2, pdrop2=0.2, pdrop3=0.2, pdrop4=0.2):

        cnn = lasagne.layers.InputLayer(shape=(None,) + self.input_shape,
                                        input_var=self.input_var)

        cnn = lasagne.layers.Conv2DLayer(cnn, num_filters=nf1,
                                 filter_size=(3, 3),
                                 pad='same',
                                 nonlinearity=lasagne.nonlinearities.identity)

        cnn = batch_norm.BatchNormLayer(cnn, epsilon=epsilon, alpha=alpha,
                                nonlinearity=lasagne.nonlinearities.rectify)

        cnn = lasagne.layers.Conv2DLayer(cnn, num_filters=nf1,
                                 filter_size=(3, 3),
                                 pad='same',
                                 nonlinearity=lasagne.nonlinearities.identity)

        cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))

        cnn = batch_norm.BatchNormLayer(cnn, epsilon=epsilon, alpha=alpha,
                                nonlinearity=lasagne.nonlinearities.rectify)

        cnn = lasagne.layers.dropout(cnn, p=pdrop1)

        cnn = lasagne.layers.Conv2DLayer(cnn, num_filters=nf2,
                                 filter_size=(3, 3),
                                 pad='same',
                                 nonlinearity=lasagne.nonlinearities.identity)

        cnn = batch_norm.BatchNormLayer(cnn, epsilon=epsilon, alpha=alpha,
                                 nonlinearity=lasagne.nonlinearities.rectify)

        cnn = lasagne.layers.Conv2DLayer(cnn, num_filters=nf2,
                                filter_size=(3, 3),
                                pad='same',
                                nonlinearity=lasagne.nonlinearities.identity)

        cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))

        cnn = batch_norm.BatchNormLayer(cnn, epsilon=epsilon, alpha=alpha,
                                nonlinearity=lasagne.nonlinearities.rectify)

        cnn = lasagne.layers.dropout(cnn, p=pdrop2)

        # 512C3-512C3-P2
        cnn = lasagne.layers.Conv2DLayer(cnn, num_filters=nf3,
                                 filter_size=(3, 3),
                                 pad='same',
                                 nonlinearity=lasagne.nonlinearities.identity)
        cnn = batch_norm.BatchNormLayer(cnn, epsilon=epsilon, alpha=alpha,
                                 nonlinearity=lasagne.nonlinearities.rectify)

        cnn = lasagne.layers.Conv2DLayer(cnn, num_filters=nf3,
                                 filter_size=(3, 3),
                                 pad='same',
                                 nonlinearity=lasagne.nonlinearities.identity)

        cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))

        cnn = batch_norm.BatchNormLayer(cnn, epsilon=epsilon, alpha=alpha,
                                nonlinearity=lasagne.nonlinearities.rectify)
        cnn = lasagne.layers.dropout(cnn, p=pdrop3)

        cnn = lasagne.layers.Conv2DLayer(cnn, num_filters=nf4,
                                 filter_size=(3, 3),
                                 pad='same',
                                 nonlinearity=lasagne.nonlinearities.identity)
        cnn = batch_norm.BatchNormLayer(cnn, epsilon=epsilon, alpha=alpha,
                                 nonlinearity=lasagne.nonlinearities.rectify)

        cnn = lasagne.layers.Conv2DLayer(cnn, num_filters=nf4,
                                 filter_size=(3, 3),
                                 pad='same',
                                 nonlinearity=lasagne.nonlinearities.identity)

        cnn = batch_norm.BatchNormLayer(cnn, epsilon=epsilon, alpha=alpha,
                                nonlinearity=lasagne.nonlinearities.rectify)

        cnn = lasagne.layers.dropout(cnn, p=pdrop4)

        cnn = lasagne.layers.DenseLayer(cnn, num_units=10,
                                nonlinearity=lasagne.nonlinearities.softmax)

        return cnn

    def objective_function_test(self, x):
        self.objective_function(x)
        return self.test_error

    def objective_function(self, x):

        self.start_time = time.time()

        # Learning rate and momentum
        LR_start = np.power(10.0, -1 - 3.0 * x[0, 0])
        LR_fin = np.power(10.0, -3 - 3.0 * x[0, 1])
        betaval = 0.8 + 0.199 * x[0, 2]
        beta2val = 1.0 - np.power(10.0, -2.0 - 2.0 * x[0, 3])
        epsval = np.power(10.0, -5.0 - 5.0 * x[0, 4])
        num_epochs_adapt = int(20 + 200 * x[0, 5])

        # Batch normalization
        alpha = 0.01 + 0.2 * x[0, 6]
        epsilon = np.power(10.0, -8.0 + 5.0 * x[0, 7])

        # Dropout
        pdrop1 = 0.8 * x[0, 8]
        pdrop2 = 0.8 * x[0, 9]
        pdrop3 = 0.8 * x[0, 10]
        pdrop4 = 0.8 * x[0, 11]

        # Number of units in each layer
        nfilters1 = int(np.power(2.0, 3.0 + 5.0 * x[0, 12]))
        nfilters2 = int(np.power(2.0, 3.0 + 5.0 * x[0, 13]))
        nfilters3 = int(np.power(2.0, 4.0 + 5.0 * x[0, 14]))
        nfilters4 = int(np.power(2.0, 4.0 + 5.0 * x[0, 15]))

        print("Config:")
        print("LR_start=%f" % (LR_start))
        print("LR_fin=%f" % (LR_fin))
        print("nfilters1=%f" % (nfilters1))
        print("nfilters2=%f" % (nfilters2))
        print("nfilters3=%f" % (nfilters3))
        print("nfilters4=%f" % (nfilters4))

        # Build network
        self.network = self._build_cnn(float(alpha), float(epsilon),
                            int(nfilters1), int(nfilters2),
                            int(nfilters3), int(nfilters4),
                            float(pdrop1), float(pdrop2),
                            float(pdrop3), float(pdrop4))

        if self.do_enves or self.multitask:
            size = int(np.exp(x[0, -1]))
            print "Size: %f" % size
            shuffle = np.random.permutation(np.arange(int(np.exp(self.s_max))))

            X_train = self.X_train[shuffle[:size]]
            y_train = self.y_train[shuffle[:size]]

            i = 0
            # Check if we have a sample of each class in the subset
            while True:
                if (np.unique(y_train).shape[0] == self.n_classes):
                    break

                shuffle = np.random.permutation(np.arange(int(np.exp(self.s_max))))
                X_train = self.X_train[shuffle[:size]]
                y_train = self.y_train[shuffle[:size]]

                i += 1
                # Sanity check if we can actually find a valid shuffled split
                if i == 20:
                    ValueError("Couldn't find a valid split that contains a \
                    sample from each class after 20 iterations. \
                    Maybe increase your bounds!")

            valid_error, lcurve, test_error = self.train(X_train, y_train,
                                 self.X_valid, self.y_valid,
                                 self.X_test, self.y_test,
                                 self.num_epochs, num_epochs_adapt,
                                 LR_start, LR_fin, betaval, epsval,
                                 beta2val, self.batch_size)
        else:
            valid_error, lcurve, test_error = self.train(self.X_train, self.y_train,
                                             self.X_valid, self.y_valid,
                                             self.X_test, self.y_test,
                                             self.num_epochs, num_epochs_adapt,
                                             LR_start, LR_fin, betaval, epsval,
                                             beta2val, self.batch_size)
        self.lcurve = lcurve
        self.test_error = np.array([[test_error]])
        cost = time.time() - self.start_time

        valid_error = np.log(valid_error)

        if self.with_costs:
            return np.array([[valid_error]]), np.array([[cost]])
        else:
            return np.array([[valid_error]])

    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]

    def train(self, X_train, y_train, X_val, y_val, X_test, y_test,
        num_epochs=50, num_epochs_adapt=50,
        LR_start=0.001, LR_fin=0.001,
        betaval=0.9, epsval=1e-08,  beta2val=0.999, bsmax=500):

        prediction = lasagne.layers.get_output(self.network)

        loss = lasagne.objectives.categorical_crossentropy(prediction,
                                                           self.target_var)
        loss = loss.mean()

        # Define lr decay schedule
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        LR_decay = (LR_fin / LR_start) ** (1. / num_epochs_adapt)
        learning_rate = LR_start
        #LR = theano.shared(learning_rate, dtype=theano.config.floatX)
        LR = theano.shared(np.cast['float32'](learning_rate))

        # Configure solver Adam
        updates = lasagne.updates.adam(loss, params, learning_rate=LR,
                                       beta1=float(betaval),
                                       beta2=float(beta2val),
                                       epsilon=float(epsval))

        # Create a loss expression for validation/testing.
        test_prediction = lasagne.layers.get_output(self.network,
                                                    deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                                self.target_var)
        test_loss = test_loss.mean()
        # As a bonus, also create an expression for
        # the classification accuracy:
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), self.target_var),
                          dtype=theano.config.floatX)

        # Compile a function performing a training step on a mini-batch
        # (by giving the updates dictionary) and returning
        # the corresponding training loss:
        train_fn = theano.function([self.input_var, self.target_var], loss,
                                   updates=updates)

        # Compile a second function computing the validation loss and accuracy:
        val_fn = theano.function([self.input_var, self.target_var],
                                 [test_loss, test_acc])

        print("Starting training...")

        best_validation_error = 1e+10
        best_predicted_test_error = 1e+10

        learning_curve = np.zeros([num_epochs])

        # We iterate over epochs:
        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            # a full pass over the training data:
            train_err = 0
            train_acc = 0
            train_batches = 0

            bs_begin = 16
            bs_end = 16
            fac_begin = 100
            fac_end = 100
            adapt_type = 0
            mult_bs = np.exp(np.log(float(bs_end) / float(bs_begin)) / num_epochs_adapt)
            mult_fac = np.exp(np.log(fac_end / fac_begin) / num_epochs_adapt)
            fac = fac_begin * np.power(mult_fac, epoch)
            if (adapt_type == 0):   # linear
                bs = bs_begin + (bs_end - bs_begin) * (float(epoch) / float(num_epochs_adapt - 1))
            if (adapt_type == 1):   # exponential
                bs = bs_begin * np.power(mult_bs, epoch)
            if (epoch > num_epochs_adapt):
                bs = bs_end
                fac = fac_end
            bs = int(np.floor(bs))

            for batch in self.iterate_minibatches(X_train, y_train, bs, shuffle=True):
                inputs, targets = batch
                s = time.time()
                batch_err = train_fn(inputs, targets)

            bsmax = 500

            for batch in self.iterate_minibatches(X_train, y_train,
                                             bsmax,
                                             shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                train_err += err
                train_acc += acc
                train_batches += 1

            cur_train_error = train_err / train_batches

            # Full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in self.iterate_minibatches(X_val, y_val,
                                            bsmax, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

            cur_valid_error = 100 - val_acc / val_batches * 100

            learning_curve[epoch] = cur_valid_error

            if (cur_valid_error < best_validation_error):
                best_validation_error = cur_valid_error
                learning_curve[epoch]
                test_err = 0
                test_acc = 0
                test_batches = 0
                for batch in self.iterate_minibatches(X_test, y_test,
                                                 bsmax,
                                                 shuffle=False):
                    inputs, targets = batch
                    err, acc = val_fn(inputs, targets)
                    test_err += err
                    test_acc += acc
                    test_batches += 1
                best_predicted_test_error = 100 - test_acc / test_batches * 100

            print("Epoch {} of {}".format(epoch + 1, num_epochs))
            curtime = time.time()
            epoch_time = curtime - epoch_start_time
            total_time = curtime - self.start_time
            print("Epoch time {:.3f}s, "
                  "total time {:.3f}s".format(epoch_time, total_time))

            print("Training loss:\t\t{:.5g}".format(train_err / train_batches))
            print("Train error:\t\t{:.3f} %".format(cur_train_error))
            print("Validation loss:\t\t{:.7f}".format(val_err / val_batches))
            print("Validation error:\t\t{:.3f} % , "
                  "Test error:\t\t{:.3f} % ".format(cur_valid_error,
                                                    best_predicted_test_error))

            # Adapt the learning rate
            print("Decay lr=%f by factor %f" % (LR.get_value(), LR_decay))
            LR.set_value(np.float32(LR.get_value() * LR_decay))

        return best_validation_error / 100., learning_curve, best_predicted_test_error / 100.

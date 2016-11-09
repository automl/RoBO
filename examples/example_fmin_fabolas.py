import os
import sys
import time
import numpy as np

from sklearn import svm

from robo.fmin import fabolas_fmin


# Example script to optimize the C and gamma parameter of a
# support vector machine on MNIST with Fabolas.
# Have a look into the paper " Fast Bayesian Optimization of Machine Learning
# Hyperparameters on Large Datasets" (http://arxiv.org/abs/1605.07079)
# to see how it works. Note in order run this example you need scikit-learn
# you can install by: pip install sklearn


def load_dataset():
    # This function loads the MNIST data, its copied from the Lasagne tutorial
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    X_train = X_train.reshape(X_train.shape[0], 28 * 28)
    X_val = X_val.reshape(X_val.shape[0], 28 * 28)
    X_test = X_test.reshape(X_test.shape[0], 28 * 28)
    
    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test


# The optimization function that we want to optimize.
# It gets a numpy array x with shape (1,D) where D are the number of parameters
# and s which is the ratio of the training data that is used to
# evaluate this configuration
def objective_function(x, s):

    # Start the clock to determine the cost of this function evaluation
    start_time = time.time()

    # Shuffle the data and split up the request subset of the training data    
    size = int(np.exp(s))
    s_max = y_train.shape[0]
    shuffle = np.random.permutation(np.arange(s_max))
    train_subset = X_train[shuffle[:size]]
    train_targets_subset = y_train[shuffle[:size]]

    # Train the SVM on the subset set
    C = np.exp(float(x[0, 0]))
    gamma = np.exp(float(x[0, 1]))
    clf = svm.SVC(gamma=gamma, C=C)
    clf.fit(train_subset, train_targets_subset)
    
    # Validate this hyperparameter configuration on the full validation data
    y = 1 - clf.score(X_val, y_val)

    c = time.time() - start_time

    return np.array([[np.log(y)]]), np.array([[c]])

# Load the data
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()


# We optimize s on a log scale, as we expect that the performance varies
# logarithmically across s
s_min = np.log(100)
s_max = np.log(X_train.shape[0])

# Defining the bounds and dimensions of the
# input space (configuration space + environment space)
# We also optimize the hyperparameters of the svm on a log scale
X_lower = np.array([-10, -10, s_min])
X_upper = np.array([10, 10, s_max])

# Start Fabolas to optimize the objective function
res = fabolas_fmin(objective_function, X_lower, X_upper, num_iterations=100)

x_best = res["x_opt"]
print(x_best)
print(objective_function(x_best[:, :-1], s=x_best[:, None, -1]))


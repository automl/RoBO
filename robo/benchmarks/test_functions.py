import math
import numpy as np

class ObjectiveFkt(object):
    def __init__(self, dims, X_lower, X_upper, fkt, X_stars=None, Y_star=None):
        self.dims = dims 
        self.X_lower = X_lower
        self.X_upper = X_upper
        self.fkt = fkt
        self.X_stars = X_stars
        self.Y_star = Y_star

    def __call__(self, x, **kwargs):
        return self.fkt(x, **kwargs)


def _branin(x):
    one = np.ones((x.shape[0], 1))
    x1 = x[:, 0][:,None]
    x2 = x[:, 1][:,None]

    a = 1
    b = 5.1 / (4 * math.pi ** 2)
    c = 5 / math.pi
    r = 6
    s = 10
    t = 1 / (8 * math.pi)
    f = (x2 - b * x1 * x1 + c * x1 - r)
    y = one * a * f * f + s * (1 - t) * np.cos(x1) + s

    return np.array(y)

branin = ObjectiveFkt(2, 
                      X_lower=np.array([-5, 0]), 
                      X_upper=np.array([10, 15]), 
                      fkt=_branin,
                      X_stars=np.array(((-np.pi, 12.275), (np.pi, 2.275), (9.42478, 2.475))),
                      Y_star=0.397887)



def _one_dim_test(x):
    #return 4 * (3 * (x**2+x-2) * np.cos(3 * x)+(2 * x+1) * np.sin(3 *x))

    return  np.sin(3*x) * 4*(x-1)* (x+2)


one_dim_test= ObjectiveFkt(1, 
                      X_lower=np.array([-2.1]), 
                      X_upper=np.array([2.1]), 
                      fkt=_one_dim_test,
                      X_stars=np.array(((1.73633274819831,),)),
                      Y_star=-9.67539878)

def _hartmann6(x, zero_mean=False):
    alpha = [1.00, 1.20, 3.00, 3.20]
    A = np.array([[10.00, 3.00, 17.00, 3.50, 1.70, 8.00],
                  [ 0.05, 10.00, 17.00, 0.10, 8.00, 14.00],
                  [ 3.00, 3.50, 1.70, 10.00, 17.00, 8.00],
                  [17.00, 8.00, 0.05, 10.00, 0.10, 14.00]])
    P = 0.0001 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                           [2329, 4135, 8307, 3736, 1004, 9991],
                           [2348, 1451, 3522, 2883, 3047, 6650],
                           [4047, 8828, 8732, 5743, 1091, 381]])
    prefactor = -1.0
    if zero_mean:
        prefactor = -1.0 / 1.94
    external_sum = np.zeros((x.shape[0], 1))
    if zero_mean:
        external_sum.fill(2.58)
    for i in range(4):
        internal_sum = 0
        for j in range(6):
            A_ij = np.empty((x.shape[0], 1))
            A_ij.fill(A[i, j])
            P_ij = np.empty((x.shape[0], 1))
            P_ij.fill(P[i, j])
            internal_sum = internal_sum + A_ij * (x[:, j][:,None] - P_ij) * (x[:, j][:,None] - P_ij)  
        external_sum = external_sum + alpha[i] * np.exp(-internal_sum)
    return np.array(prefactor * external_sum)

hartmann6 = ObjectiveFkt(6, 
                         X_lower=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 
                         X_upper=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 
                         fkt=_hartmann6,
                         X_stars=np.array(((0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573),)),
                         Y_star=-3.32237)
def _hartmann3(x):
    alpha = [1.0, 1.2, 3.0, 3.2]
    A = np.array([[ 3.0, 10.0, 30.0],
                  [ 0.1, 10.0, 35.0],
                  [ 3.0, 10.0, 30.0],
                  [ 0.1, 10.0, 35.0]])
    P = 0.0001 * np.array([[3689, 1170, 2673],
                           [4699, 4387, 7470],
                           [1091, 8732, 5547],
                           [ 381, 5743, 8828]])
    external_sum = 0
    for i in range(4):
        internal_sum = 0
        for j in range(3):
            A_ij = np.empty((x.shape[0], 1))
            A_ij.fill(A[i, j])
            P_ij = np.empty((x.shape[0], 1))
            P_ij.fill(P[i, j])
            internal_sum = internal_sum + A_ij * (x[:, j][:,None] - P_ij) * (x[:, j][:,None] - P_ij)  
        external_sum = external_sum + alpha[i] * np.exp(-internal_sum)
    return np.array(-external_sum)

hartmann3 = ObjectiveFkt(3, 
                         X_lower=np.array([0.0, 0.0, 0.0]), 
                         X_upper=np.array([1.0, 1.0, 1.0]), 
                         fkt=_hartmann3,
                         X_stars=np.array(((0.114614, 0.555649, 0.852547),)),
                         Y_star=-3.86278)

def _goldstein_price_fkt(x):
    x1 = x[:, 0][:,None]
    x2 = x[:, 1][:,None]
    one = np.ones((x.shape[0], 1))
    factor_1 = 1 + (x1 + x2 + 1) * (x1 + x2 + 1) * (19 - 14 * x1 + 3 * x1 * x1 - 14 * x2 + 6 * x1 * x2 + 3 * x2 * x2)
    factor_2 = 30 + (2 * x1 - 3 * x2) * (2 * x1 - 3 * x2) * (18 - 32 * x1 + 12 * x1 * x1 + 48 * x2 - 36 * x1 * x2 + 27 * x2 * x2)
    return np.array(one * factor_1 * factor_2)

goldstein_price_fkt = ObjectiveFkt(2, 
                                   X_lower=np.array([-2.0, -2.0]), 
                                   X_upper=np.array([2.0, 2.0]), 
                                   fkt=_goldstein_price_fkt,
                                   X_stars=np.array(((0, -1),)),
                                   Y_star=3)
import math
import numpy as np
def branin(x):
    x1 = x[:,0]
    x2 = x[:,1]
    a = 1
    b = 5.1/(4*math.pi**2)
    c = 5/math.pi
    r = 6;
    s = 10;
    t = 1/(8*math.pi);
    f = (x2-b*x1*x1+c*x1-r)
    y = a*f*f+s*(1-t)*np.cos(x1)+s;
    return y

def hartmann6(x, zero_mean=False):
    alpha = [1.00, 1.20,  3.00,  3.20]
    A = np.array([[10.00,  3.00, 17.00,  3.50,  1.70,  8.00],
                  [ 0.05, 10.00, 17.00,  0.10,  8.00, 14.00],
                  [ 3.00,  3.50,  1.70, 10.00, 17.00,  8.00],
                  [17.00,  8.00,  0.05, 10.00,  0.10, 14.00]])
                 
    P = 0.0001 *  np.array([[1312, 1696, 5569,  124, 8283, 5886],
                            [2329, 4135, 8307, 3736, 1004, 9991],
                            [2348, 1451, 3522, 2883, 3047, 6650],
                            [4047, 8828, 8732, 5743, 1091,  381]])
    a = -1.0
    if zero_mean:
        a = -1.0/1.94
    b = 0
    if zero_mean:
        b = 2.58
    for i in range(4):
        c = 0
        for j in range(6):
            A_ij = np.empty((x.shape[0],1))
            A_ij.fill(A[i,j])
            P_ij = np.empty((x.shape[0],1))
            P_ij.fill(P[i,j])
            c += A_ij * (x[:, j] - P_ij) * (x[:, j] - P_ij)  
        b += alpha[i] * np.exp(-c)
    return np.array(a*b)
    
def hartmann3(x):
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
            A_ij = np.empty((x.shape[0],1))
            A_ij.fill(A[i,j])
            P_ij = np.empty((x.shape[0],1))
            P_ij.fill(P[i,j])
            internal_sum += A_ij * (x[:, j] - P_ij) * (x[:, j] - P_ij)  
        external_sum += alpha[i] * np.exp(-internal_sum)
    return np.array(-external_sum)

def goldstein_price_fkt(x):
    x1 = x[:,0]
    x2 = x[:,1]
    one = np.ones((x.shape[0],1))
    factor_1 = 1 + (x1+x2+1)*(x1+x2+1)*(19-14*x1+3*x1*x1-14*x2+6*x1*x2+3*x2*x2)
    factor_2 = 30 + (2*x1-3*x2)*(2*x1-3*x2)*(18-32*x1+12*x1*x1+48*x2-36*x1*x2+27*x2*x2)
    return factor_1*factor_2

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

if  __name__ == "__main__":
    import random
    X_lower = np.array([0.0,0.0,0.0,0.0,0.0,0.0]);
    X_upper = np.array([1.0,1.0,1.0,1.0,1.0,1.0]);
    x_star =  np.array(((0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573),))
    
    assert -3.32237 - hartmann6(x_star) < 0.0001
    print  "f(0.31225309,  0.1687257 ,  0.4116617 ,  0.27186383,  0.29280941, 0.6497946) = " + str(hartmann6(np.array([[0.31225309,  0.1687257 ,  0.4116617 ,  0.27186383,  0.29280941,
        0.6497946 ]])))
    num_t = 2000
    dims = 6
    X = np.empty((num_t, dims))
    Y = np.empty((num_t, 1))
    for j in range(num_t):
        for i in range(dims):
            X[j,i] = random.random() * (X_upper[i] - X_lower[i]) + X_lower[i];
    Y = hartmann6(X)
    print  Y.min()
    

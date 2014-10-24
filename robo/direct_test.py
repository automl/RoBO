#!/usr/bin/python
"""
Solve the six-hump camelback function.
"""

from __future__ import division
from DIRECT import solve
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm


def obj(x, user_data):
    """Six-hump camelback function"""
    
    x1 = x[0]
    x2 = x[1]
    
    f = (4 - 2.1*(x1*x1) + (x1*x1*x1*x1)/3.0)*(x1*x1) + x1*x2 + (-4 + 4*(x2*x2))*(x2*x2)
    return f, 0


if __name__ == '__main__':

    l = [-3, -2]
    u = [3, 2]

    x, fmin, ierror = solve(
                        obj,
                        l,
                        u
                        )

    print 'Optimal point:', x
    print 'Optimal value:', fmin
    print 'Exit status:', ierror
    
    #
    # Plot the results.
    #
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X, Y = np.mgrid[x[0]-1:x[0]+1:50j, x[1]-1:x[1]+1:50j]
    Z = np.zeros_like(X)

    for i in range(X.size):
        Z.ravel()[i] = obj([X.flatten()[i], Y.flatten()[i]], None)[0]
        
    ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet)
    ax.scatter(x[0], x[1], fmin, c='r', marker='o')
    ax.set_title('Six-hump Camelback Function')
    plt.show()
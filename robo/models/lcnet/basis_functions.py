import numpy as np


def vapor_pressure(x, a, b, c):
    a_ = -(a + 1) / 2
    b_ = -(b + 1) / 2
    c_ = -(c + 1) / 2

    b_ /= 10.
    c_ /= 10.

    return np.exp(a_ + b_ / (x + 1e-15) + c_ * np.log(x + 1e-15)) - np.exp(a_ + b_)


def hill_3(x, a, b, c):
    a_ = (a + 1) / 2
    b_ = (b + 1) / 2
    c_ = (c + 1) / 2

    b_ *= 0.1

    return a_ * (1. / ((c_ / x) ** b_ + 1.) - 1. / (c_ ** b_ + 1.))


def pow_func(x, a, b):

    a_ = (a + 1) / 2
    b_ = (b + 1) / 2
    return a_ * (x ** b_ - 1)


def log_power(x, a, b, c):
    a_ = (a + 1) / 2
    b_ = (b + 1) / 2
    c_ = (c + 1) / 2

    c_ *= 0.1
    return 2 * a_ * (1 / (1. + np.exp(-c_ * b_)) - 1 / (1. + (x / np.exp(b_)) ** c_))


def exponential(x, a, b):

    a_ = (a + 1) / 2
    b_ = (b + 1) / 2
    return b_ * (-np.exp(-a_ * x) + np.exp(-a_))

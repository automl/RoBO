import numpy as np


def vapor_pressure(t, a, b, c):

    a_ = a
    b_ = b / 10.
    c_ = c / 10.

    return np.exp(-a_ - b_ / t - c_ * np.log(t)) - np.exp(-a_ - b_)


def pow_func(t, a, b):
    return a * (t ** b - 1)


def log_power(t, a, b, c):

    b_ = b / 10

    return 2 * a * (1 / (1 + np.exp(-b_ * c)) - 1 / (1 + np.exp(c * np.log(t) - b_)))


def exponential(t, a, b):
    b_ = 10 * b
    return a * (np.exp(-b_) - np.exp(-b_ * t))


def hill_3(t, a, b, c):
    return a * (1 / (c ** b * t ** (-b) + 1) - a / (c ** b + 1))

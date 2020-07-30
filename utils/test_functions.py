import numpy as np
import math


def sphere(X):
    X = np.asarray_chkfinite(X)
    return np.sum(X ** 2)


def michalewicz(X, m=5):
    X = np.asarray_chkfinite(X)
    d = len(X)
    i = np.arange(1, d + 1)
    return -np.sum(np.sin(X) * np.sin(i * X ** 2 / np.pi) ** (2 * m))


def easom(X):
    X = np.asarray_chkfinite(X)
    return -math.cos(X[0]) * math.cos(X[1]) * math.exp(-(X[0] - math.pi) ** 2 - (X[1] - math.pi) ** 2)


def shubert(X):
    X = np.asarray_chkfinite(X)
    i = np.arange(1, 6)
    return np.sum(i * np.cos((i + 1) * X[0] + i)) * np.sum(i * np.cos((i + 1) * X[1] + i))


def rosenbrock(X):
    X = np.asarray_chkfinite(X)
    X0 = X[:-1]
    X1 = X[1:]
    return np.sum((1 - X0) ** 2) + 50 * np.sum((X1 - X0 ** 2) ** 2)


def rastrigin(X):
    X = np.asarray_chkfinite(X)
    d = len(X)
    return 5 * d + np.sum(X ** 2 - 5 * np.cos(2 * np.pi * X))


def schwefel(X):
    X = np.asarray_chkfinite(X)
    d = len(X)
    return 418.9829 * d - np.sum(X * np.sin(np.sqrt(np.abs(X))))


def griewank(X):
    X = np.asarray_chkfinite(X)
    d = len(X)
    i = np.arange(1, d + 1)
    s = np.sum(X**2)
    p = np.prod(np.cos(X / np.sqrt(i)))
    return s / 4000 - p + 1


def ackley(X, a=20, b=0.2, c=2 * np.pi):
    X = np.asarray_chkfinite(X)
    d = len(X)
    s1 = np.sum(X**2)
    s2 = np.sum(np.cos(c * X))
    return -a * np.exp(-b * np.sqrt(s1 / d)) - np.exp(s2 / d) + a + np.exp(1)


# Implemented only for d=2 and m=5.
def langermann(X, m=5, c=(1, 2, 5, 2, 3)):
    X = np.asarray_chkfinite(X)
    c = np.asarray_chkfinite(c)
    A = np.array([[3, 5], [5, 2], [2, 1], [1, 4], [7, 9]])
    res = 0
    for i in range(m):
        s = np.sum((X - A[i]) ** 2)
        res += c[i] * np.exp(-1 / np.pi * s) * np.cos(np.pi * s)
    return res


def dixonprice(X):
    X = np.asarray_chkfinite(X)
    d = len(X)
    j = np.arange(2, d + 1)
    X2 = 2 * X ** 2
    return sum(j * (X2[1:] - X[:-1]) ** 2 ) + (X[0] - 1) ** 2


def levy(X):
    X = np.asarray_chkfinite(X)
    z = 1 + (X - 1) / 4
    return (np.sin(np.pi * z[0]) ** 2
        + np.sum((z[:-1] - 1) ** 2 * (1 + 5 * np.sin(np.pi * z[:-1] + 1 ) ** 2))
        + (z[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * z[-1]) ** 2))


def perm(X, b=.5):
    X = np.asarray_chkfinite(X)
    d = len(X)
    j = np.arange(1., d + 1)
    xbyj = np.fabs(X) / j
    return np.mean([np.mean( (j ** k + b) * (xbyj ** k - 1)) ** 2 for k in j / d ])

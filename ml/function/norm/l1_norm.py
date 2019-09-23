import numpy as np

def f(x):
    return np.sum(np.abs(x))

def grad_f(x):
    out = x.copy()
    out[out < 0] = -1
    out[out >= 0] = 1
    return out

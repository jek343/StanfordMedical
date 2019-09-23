import numpy as np

def gradient(X, y, theta, f, grad_f):
    out = np.zeros(theta.shape)
    for i in range(0, len(y)):
        out -= grad_f(X[i], theta) / f(X[i], theta)
    return out

def cost(X, y, theta, f):
    out = 0
    for i in range(0, len(y)):
        out -= np.log(f(X[i], theta) / y[i])
    return out

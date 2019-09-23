import ml.function.differentiable.sigmoid as sigmoid
import numpy as np

def f(x, theta):
    return sigmoid.f(np.dot(x, theta))

def grad_theta_f(x, theta):
    f_x = f(x, theta)
    return f_x * (1 - f_x) * x

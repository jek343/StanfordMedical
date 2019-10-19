import numpy as np

'''
grad_f(x) is the gradient of the objective to be minimized at x. Must have
same shape as x_0
x_0 is the initial iterate for gradient descent.

Will modifiy x_0 in-place
'''
def minimize(grad_f, x_0, learn_rate, n_iters, convergence_grad_magnitude):
    x = x_0
    for iter in range(n_iters):
        grad_f_x = grad_f(x)
        x -= learn_rate * grad_f_x
        if np.linalg.norm(grad_f_x) < convergence_grad_magnitude:
            return x
    return x

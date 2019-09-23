import numpy as np
'''
Numerically stable sigmoid from https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
'''
def f(x):
    if x >= 0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        # if x is less than zero then z will be small, denom can't be
        # zero because it's 1+z.
        z = np.exp(x)
        return z / (1.0 + z)

def derivative_f(x):
    f_x = f(x)
    return f_x * (1 - f_x)

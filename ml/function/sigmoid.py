import numpy as np

'''
https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
(explanation for numerical stability trick)
'''
def sigmoid(x):

    if not isinstance(x, np.ndarray):
        x = np.full(1, x)

    out = np.zeros(x.shape, dtype = np.float64)
    where_x_geq_0 = np.where(x >= 0)
    where_x_lt_0 = np.where(x < 0)
    z_geq_0 = np.exp(-x[where_x_geq_0])
    z_lt_0 = np.exp(x[where_x_lt_0])
    out[where_x_geq_0] = 1.0 / (1.0 + z_geq_0)
    out[where_x_lt_0] = z_lt_0 / (1.0 + z_lt_0)
    return out

def sigmoid_derivative_in_terms_of_output(sigmoid_outs):
    return sigmoid_outs * (1.0 - sigmoid_outs)

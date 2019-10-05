import numpy as np

'''
Returns: the probability of a randomly sampled person being randomly
sampled as a value by the approximate distribution that differs from that
sampled from the true distribution.

true_probs: true_probs[i,j] is the (True) probability that x[i] is of class j.
approx_probs: approx_probs[i,j] is the (approximated by model) probability that x[i] is of class j.
point_probs: point_probs[i] is the probability of a given point in p(k, x). For example,
    if x[i] consists of county data upon which the model conditions, then x[i] would
    be the probability that a random person lives in that county.
'''
def gini_penalty(point_probs, true_probs, approx_probs):
    #can make this faster by forming all pair-wise inequal probabilities, then using np.sum on the rows.
    out = 0
    for i in range(len(point_probs)):
        out += point_probs[i] * __point_gini(true_probs[i], approx_probs[i])
    return out

def __point_gini(true_prob, approx_prob):
    return true_prob * (1 - approx_prob) + (1 - true_prob) * approx_prob

def grad_gini(point_probs, grad_fs, X, y):
    out = None
    for i in range(len(point_probs)):
        add_grad = point_probs[i] * (1.0 - 2.0*y[i]) * grad_fs[i]
        out = add_grad if out is None else out + add_grad
    return out

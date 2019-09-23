import ml.penalty.KL_divergence as KL_divergence

'''
Returns: the gradient of C = KL(y || f(X)) + lambda g(theta),
    where theta are the parameters of the model, f, X is the input
    dataset, y is the output dataset, and grad_f, grad_g are the
    gradients of f and g with respect to theta (where grad_f is
    a function of an input, x, and theta, and grad_g is a function
    of only theta).

'''
def gradient(X, y, theta, lambd, f, grad_f, grad_g):
    return lambd * grad_g(theta) + KL_divergence.gradient(X, y, theta, f, grad_f)

def cost(X, y, theta, lambd, f, g):
    return KL_divergence.cost(X, y, theta, f) + lambd * g(theta)

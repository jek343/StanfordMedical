import numpy as np
import ml.function.sigmoid as sigmoid

#minimizes expected log likelihood, taking classes to be sampled at random
#according to the probability to be estimated.
class ClassSampledErrorPenalizedLogisticRegression:

    def __init__(self, regularizer = None):
        if regularizer is None:
            self.__regularizer_gradient_func = self.__no_regularizer_gradient
        elif regularizer is "l2":
            self.__regularizer_gradient_func = self.__l2_regularizer_gradient
        elif regularizer is "l1":
            self.__regularizer_gradient_func = self.__l1_regularizer_gradient

    def __l1_regularizer_gradient(self):
        out = np.zeros(self.__theta.shape[0])
        out[np.where(self.__theta <= 0)] = -1
        out[np.where(self.__theta > 0)] = 1
        #last elem is for bias, so doesn't regularize it
        out[-1] = 0
        return out

    def __l2_regularizer_gradient(self):
        #last element is for the bias, so it doesn't regularize it
        out = 2 * self.__theta
        out[-1] = 0
        return out

    def __no_regularizer_gradient(self):
        return np.zeros(self.__theta.shape)

    def f(self, X, has_bias = False):
        if not has_bias:
            X_bias = np.ones((X.shape[0], X.shape[1] + 1), dtype = X.dtype)
            X_bias[:X.shape[0],:X.shape[1]] = X
            X = X_bias
        return sigmoid.sigmoid(np.dot(X, self.__theta))


    def __grad_f(self, X, f_X, one_minus_f_X):
        #without regularization for now
        return X * (f_X * one_minus_f_X)[:, np.newaxis]

    def __get_error(self, X, y):
        f_X = self.f(X, has_bias = True)
        return np.sum(np.log(y*f_X + (1-y)*(1-f_X)))

    def __error_gradient(self, X, y, one_minus_y, regularizer_strength):
        f_X = self.f(X, has_bias = True)
        one_minus_f_X = 1 - f_X
        grad_f_X = self.__grad_f(X, f_X, one_minus_f_X)
        out = ((y/f_X) - (one_minus_y/one_minus_f_X))[:,np.newaxis] * grad_f_X
        return np.sum(out, axis = 0) - regularizer_strength * self.__regularizer_gradient_func()

    def __step_theta(self, grad, learn_rate):
        self.__theta += (learn_rate) * grad


    def get_params(self):
        return self.__theta.copy()

    def train(self, X, y, learn_rate, regularizer_strength, n_iters, n_print_iters):
        X_bias = np.ones((X.shape[0], X.shape[1] + 1), dtype = X.dtype)
        X_bias[:X.shape[0],:X.shape[1]] = X
        X = X_bias
        one_minus_y = 1 - y
        self.__theta = np.zeros(X.shape[1], dtype = np.float64)
        for iter in range(n_iters):
            if iter % n_print_iters == 0:
                f_X = self.f(X, has_bias = True)
                print("error(" + str(iter) + ")")
                print("\tmean L1: ", np.average(np.abs(f_X - y)))
                print("\tmean output: ", np.average(f_X))
                print("\tregularized expected log likelihood: ", self.__get_error(X, y))
            grad = self.__error_gradient(X, y, one_minus_y, regularizer_strength)
            self.__step_theta(grad, learn_rate)

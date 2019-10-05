import numpy as np
import ml.function.sigmoid as sigmoid

class SquareErrorPenalizedLogisticRegression:


    def f(self, X):
        logits = np.dot(X, self.__theta)
        return sigmoid.sigmoid(logits)



    def __grad_f(self, X, f_X):
        #without regularization for now
        return X * sigmoid.sigmoid_derivative_in_terms_of_output(f_X)[:, np.newaxis]

    def __expected_square_error(self, point_probs, X, y):
        return np.sum(point_probs * np.square(y - self.f(X)))

    def __grad_expected_square_error(self, point_probs, X, y):
        #out = np.zeros(self.__theta.shape, dtype = np.float64)
        f_X = self.f(X)
        grad_f_X = self.__grad_f(X, f_X)
        #for i in range(X.shape[0]):
        #    out += point_probs[i] * (y[i] - f_X[i]) * grad_f_X[i]
        #return out
        return -np.sum((point_probs * (y - f_X))[:,np.newaxis] * grad_f_X)

    def train(self, point_probs, X, y, learn_rate, n_iters, n_print_iters):
        if point_probs is None:
            point_probs = np.full(X.shape[0], 1.0 / float(X.shape[0]))
        self.__theta = np.zeros(X.shape[1], dtype = np.float64)
        for iter in range(n_iters):
            if iter % n_print_iters == 0:
                print("error: ", self.__expected_square_error(point_probs, X, y))
            grad = self.__grad_expected_square_error(point_probs, X, y)
            self.__theta -= learn_rate * grad

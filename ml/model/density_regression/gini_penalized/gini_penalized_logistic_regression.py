import ml.model.density_regression.gini_penalized.binary_gini_penalized_model as gini_penalty
import numpy as np
import ml.function.sigmoid as sigmoid

#Works, but gives unuseful solutions.
#didn't realize the linearity of the cost fixing each y[i]
class GiniPenalizedLogisticRegression:


    def f(self, X):
        logits = np.dot(X, self.__theta)
        return sigmoid.sigmoid(logits)

    def __grad_f(self, X, f_X):
        #without regularization for now
        return X * sigmoid.sigmoid_derivative_in_terms_of_output(f_X)[:, np.newaxis]

    def __grad_gini(self, point_probs, X, y):
        f_X = self.f(X)
        grad_f = self.__grad_f(X, f_X)
        return gini_penalty.grad_gini(point_probs, grad_f, X, y)

    def train(self, point_probs, X, y, learn_rate, n_iters, n_print_iters):
        self.__theta = np.zeros(X.shape[1], dtype = np.float64)
        for iter in range(n_iters):
            if iter % n_print_iters == 0:
                f_X = self.f(X)

                print("mean L1: ", np.sum(np.abs(y - f_X))/float(y.shape[0]))
                print("mean L2: ", np.sum(np.square(y-f_X))/float(y.shape[0]))
                #print("theta: ", np.abs(self.__theta).max())
                print("GINI ( " + str(iter) + "): ", gini_penalty.gini_penalty(point_probs, f_X, y))
            grad_theta = self.__grad_gini(point_probs, X, y)
            learn_rate_scaled_grad_theta = learn_rate * grad_theta
            self.__theta -= learn_rate_scaled_grad_theta

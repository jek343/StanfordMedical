from abc import ABC, abstractmethod
import ml.optimization.gradient_descent_optimizer as gradient_descent_optimizer
import numpy as np

class BoostedRegressor(ABC):

    def __init__(self, pointwise_loss, num_learners, learner_regularizer = 1):
        BoostedRegressor.set_params(self, pointwise_loss, num_learners, learner_regularizer)

    def set_params(self, pointwise_loss, num_learners, learner_regularizer):
        self._pointwise_loss = pointwise_loss
        self._num_learners = num_learners
        self._learner_regularizer = learner_regularizer

    def predict(self, X):
        out = np.zeros(X.shape[0], dtype = np.float64)
        for m in range(len(self.__h)):
            if self.__h[m] is None:
                return out
            out += self.__gamma[m] * self.__h[m](X)
        return out



    '''
    Returns a function that takes in X, a numpy array
    of datapoints where X[i] is the ith datapoint, and returns
    a vector h, where h[i] is the evaluation of the trained
    weak learner on X[i]
    '''
    @abstractmethod
    def _fit_weak_learner(self, X, y):
        pass

    @abstractmethod
    def _solve_for_gamma_m(self, X, y, current_model_preds, h_m):
        pass

    def __get_initial_weak_learner(self, y):
        y_avg = np.average(y)
        return lambda X : np.full(X.shape[0], y_avg)

    def get_weak_learner_coefficients(self):
        return self.__gamma

    def train(self, X, y):
        self.__h = [None for i in range(0, self._num_learners)]
        self.__gamma = np.zeros(self._num_learners, dtype = np.float64)
        self.__h[0] = (self.__get_initial_weak_learner(y))
        self.__gamma[0] = 1.0
        current_model_preds = self.predict(X)
        for m in range(1, self._num_learners):
            pseudo_residuals = -self._pointwise_loss.loss_derivatives(current_model_preds, y)
            h_m = self._fit_weak_learner(X, pseudo_residuals)
            self.__h[m] = h_m
            self.__gamma[m] = self._learner_regularizer * self._solve_for_gamma_m(X, y, current_model_preds, h_m)
            current_model_preds += self.__gamma[m] * h_m(X)
            #print("learner (" + str(m) + ") mean error: " + str(np.average(self._pointwise_loss.losses(current_model_preds, y))))

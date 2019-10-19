from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
import ml.optimization.gradient_descent_optimizer as gradient_descent_optimizer
import numpy as np
from ml.model.regression.loss.pointwise_square_error_loss import PointwiseSquareErrorLoss

class BoostedRegressor(ABC, BaseEstimator):

    def __init__(self, pointwise_loss, num_learners, learner_regularizer = 1, remaining_params = {}):
        self._params = {"pointwise_loss": pointwise_loss,\
            "num_learners": num_learners,\
            "learner_regularizer": learner_regularizer}
        for k in remaining_params:
            self._params[k] = remaining_params[k]


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

    def __get_pointwise_loss(self):
        loss_str = self._params["pointwise_loss"]
        if loss_str == "PointwiseSquareErrorLoss":
            return PointwiseSquareErrorLoss()
        else:
            raise ValueError("Pointwise loss string: " + loss_str + " not implemented")

    def train(self, X, y):
        pointwise_loss = self.__get_pointwise_loss()
        num_learners = self._params["num_learners"]
        learner_regularizer = self._params["learner_regularizer"]

        self.__h = [None for i in range(0, num_learners)]
        self.__gamma = np.zeros(num_learners, dtype = np.float64)
        self.__h[0] = (self.__get_initial_weak_learner(y))
        self.__gamma[0] = 1.0
        current_model_preds = self.predict(X)
        for m in range(1, num_learners):
            pseudo_residuals = -pointwise_loss.loss_derivatives(current_model_preds, y)
            h_m = self._fit_weak_learner(X, pseudo_residuals)
            self.__h[m] = h_m
            self.__gamma[m] = learner_regularizer * self._solve_for_gamma_m(X, y, current_model_preds, h_m)
            current_model_preds += self.__gamma[m] * h_m(X)
            print("learner (" + str(m) + ") mean error: " + str(np.average(pointwise_loss.losses(current_model_preds, y))))

from abc import ABC, abstractmethod

class PointwiseRegressorLoss(ABC):

    @abstractmethod
    def losses(self, y_pred, y):
        pass

    @abstractmethod
    def loss_derivatives(self, y_pred, y):
        pass

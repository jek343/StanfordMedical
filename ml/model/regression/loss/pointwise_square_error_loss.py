from ml.model.regression.loss.pointwise_regressor_loss import PointwiseRegressorLoss
import numpy as np

class PointwiseSquareErrorLoss(PointwiseRegressorLoss):

    def losses(self, y_pred, y):
        return 0.5 * np.square(y_pred - y)


    def loss_derivatives(self, y_pred, y):
        return y_pred - y

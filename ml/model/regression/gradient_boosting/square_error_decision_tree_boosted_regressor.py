from ml.model.regression.gradient_boosting.decision_tree_boosted_regressor import DecisionTreeBoostedRegressor
from ml.model.regression.loss.pointwise_square_error_loss import PointwiseSquareErrorLoss
import numpy as np

class SquareErrorDecisionTreeBoostedRegressor(DecisionTreeBoostedRegressor):

    def __init__(self, depth_range, num_learners, learner_regularizer, num_features_range, weak_learner_point_percent_range):
        DecisionTreeBoostedRegressor.__init__(self, PointwiseSquareErrorLoss(), num_learners, learner_regularizer, depth_range, num_features_range, weak_learner_point_percent_range)

    def set_params(self, num_learners, learner_regularizer, depth_range, num_features_range, weak_learner_point_percent_range):
        DecisionTreeBoostedRegressor.set_params(self, PointwiseSquareErrorLoss(), num_learners, learner_regularizer, depth_range, num_features_range, weak_learner_point_percent_range)

    '''
    gamma_m has a closed-form solution for square pointwise error, and it is
    significantly faster to use over some optimization method
    '''
    def _solve_for_gamma_m(self, X, y, current_model_preds, h_m):
        #each gamma is 1 because in the pointwise square error, the pseudo-residuals
        #are just y[i] - pred(y)[i]. This can be found logistically, or by closed-form
        #based solution by setting partial derivative to zero.
        return 1
        '''
        h_m_preds = h_m(X)
        gamma_numerator = np.sum(h_m_preds * (y - current_model_preds))
        gamma_denominator = np.sum(np.square(h_m_preds))
        return gamma_numerator/gamma_denominator
        '''

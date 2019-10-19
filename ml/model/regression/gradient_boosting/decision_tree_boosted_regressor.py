from ml.model.regression.gradient_boosting.boosted_regressor import BoostedRegressor
from sklearn import tree
import numpy as np
import ml.optimization.gradient_descent_optimizer as gradient_descent_optimizer




class DecisionTreeBoostedRegressor(BoostedRegressor):

    '''
    depth_range: a tuple of (m, n) where m is the minimum depth of a fit weak learner, and
    n is the maximum depth of a fit weak learner.


    '''
    def __init__(self,\
            pointwise_loss = "PointwiseSquareErrorLoss",\
            num_learners = 100,\
            learner_regularizer = 1,\
            depth_range = (3,7),\
            percent_features_range = (.2,.5),
            weak_learner_point_percent_range = (1.0,1.0)):
        BoostedRegressor.__init__(self, pointwise_loss, num_learners, learner_regularizer, \
            {"depth_range": depth_range, "percent_features_range": percent_features_range, \
            "weak_learner_point_percent_range": weak_learner_point_percent_range})
        #self._params["depth_range"] = depth_range
        #self._params["percent_features_range"] = percent_features_range
        #self._params["weak_learner_point_percent_range"] = weak_learner_point_percent_range
        self.__weak_decision_trees = []
        self.__features = []

    def _fit_weak_learner(self, X, y):
        depth = np.random.randint(self._params["depth_range"][0], self._params["depth_range"][1] + 1)
        n_points = np.random.randint(int(self._params["weak_learner_point_percent_range"][0] * X.shape[0]),\
            int(self._params["weak_learner_point_percent_range"][1] * X.shape[0]) + 1)
        num
        num_features_range = (int(self._params["percent_features_range"][0] * X.shape[1]),\
            int(self._params["percent_features_range"][1] * X.shape[1]))
        n_features = np.random.randint(num_features_range[0], num_features_range[1] + 1)
        self.__features.append(np.random.choice(np.arange(0, X.shape[1]), n_features))

        point_subset = np.random.choice(np.arange(0, X.shape[0]), n_points)
        feature_ind = len(self.__features) - 1
        X_features = X[:, self.__features[feature_ind]]
        out = tree.DecisionTreeRegressor(max_depth = depth)
        out = out.fit(X_features[point_subset], y[point_subset])
        self.__weak_decision_trees.append(out)
        return lambda X_all_features : out.predict(X_all_features[:, self.__features[feature_ind]])


    def get_weak_decision_trees(self):
        return self.__weak_decision_trees

    def get_weak_learner_features(self):
        return self.__features

    def _solve_for_gamma_m(self, X, y, current_model_preds, h_m):
        learn_rate = 0.00001
        n_iters = 10000
        convergence_grad_mag = 10**(-5)
        h_m_preds = h_m(X)
        def gamma_grad(gamma):
            loss_derivs = self._pointwise_loss.loss_derivatives(current_model_preds + gamma * h_m_preds, y)
            return np.dot(h_m_preds, loss_derivs)
        return gradient_descent_optimizer.minimize(gamma_grad, 0, learn_rate, n_iters, convergence_grad_mag)

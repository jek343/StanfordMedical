
import numpy as np
'''
X,y = datasets.load_breast_cancer(return_X_y = True)
X = X[:,0:5]

print("x: ", X)
X_new = np.ones((X.shape[0], X.shape[1] + 1))
X_new[:, :X_new.shape[1] - 1] = X

X = X_new
X = X.astype(np.float32)
y = y.astype(np.float32)
y *= 0.9
y += .05
'''
'''
model = logistic_regression_function
X = np.random.rand(1000,3)
y = np.zeros(X.shape[0])
theta_gt = np.random.rand(3)
print("theta_gt: ", theta_gt)
for i in range(len(X)):
    y[i] = model.f(X[i], theta_gt)

theta = np.zeros(X.shape[1])
lambd = 0#1.0
learn_rate = .001

for iter in range(0, 10000):
    grad_theta = regularized_KL_divergence.gradient(\
        X,\
        y,\
        theta,\
        lambd,\
        logistic_regression_function.f,\
        logistic_regression_function.grad_theta_f,\
        l1_norm.grad_f)
    #print("grad theta: ", grad_theta)
    if iter % 100 == 0:
        cost = regularized_KL_divergence.cost(X, y, theta, lambd, logistic_regression_function.f, l1_norm.f)
        print("cost: ", cost)
        theta -= learn_rate * grad_theta
        print("theta - theta_gt: ", np.linalg.norm(theta - theta_gt))
'''
if __name__ == "__main__":
    from ml.model.regression.gradient_boosting.decision_tree_boosted_regressor import DecisionTreeBoostedRegressor
    from sklearn.utils.estimator_checks import check_estimator

    print("is estimator? ", check_estimator(DecisionTreeBoostedRegressor))

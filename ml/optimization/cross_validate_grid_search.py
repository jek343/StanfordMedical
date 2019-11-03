import ml.data.k_fold as k_fold
import ml.optimization.grid_search as grid_search

'''
'''
def cross_validate_grid_search(X, y, k, ranges, step_sizes, param_setter_func, model_fitter_func, model_output_func, error_func):
    folds = k_fold.k_fold(X, y, k)
    def to_maximize(x):
        '''out = 0
        param_setter_func(x)
        for X_train, y_train, X_test, y_test in folds:
            model_fitter_func(X_train, y_train)
            test_outputs = model_output_func(X_test)
            test_error = error_func(test_outputs, y_test)
            out += test_error
        return - out / k'''
        param_setter_func(x)
        return - cross_validated_error(folds, model_fitter_func, model_output_func, error_func)

    return grid_search.find_max(to_maximize, ranges, step_sizes)


def cross_validated_error(folds, model_fitter_func, model_output_func, error_func):
    out = 0
    for X_train, y_train, X_test, y_test in folds:
        model_fitter_func(X_train, y_train)
        test_outputs = model_output_func(X_test)
        test_error = error_func(test_outputs, y_test)
        out += test_error
    return out / len(folds)

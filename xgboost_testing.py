from sklearn.datasets import load_boston
from ml.model.regression.gradient_boosting.decision_tree_boosted_regressor import DecisionTreeBoostedRegressor
from ml.model.regression.gradient_boosting.square_error_decision_tree_boosted_regressor import SquareErrorDecisionTreeBoostedRegressor
from ml.model.regression.loss.pointwise_square_error_loss import PointwiseSquareErrorLoss
from ml.model.density_regression.gini_penalized.gini_penalized_logistic_regression import GiniPenalizedLogisticRegression
from ml.model.density_regression.square_error_penalized.square_error_penalized_logistic_regression import SquareErrorPenalizedLogisticRegression
from ml.model.density_regression.class_sampled_error_penalized.class_sampled_error_penalized_logistic_regression import ClassSampledErrorPenalizedLogisticRegression
import ml.function.sigmoid as sigmoid
import numpy as np
import csv
from sklearn.model_selection import train_test_split


def open_csv(path):
    data_csv_file = open(path)
    return csv.reader(data_csv_file, delimiter=',')

def read_fields(csv, field_row):
    row_num = 0
    for row in csv:
        if row_num == field_row:
            out = []
            for x in row:
                out.append(x)
            return out
        row_num += 1
    return None

def get_data_as_dicts(csv, ignore_rows, field_names):
    row_num = 0
    D = []
    for row in csv:
        if not row_num in ignore_rows:
            i = 0
            d = {}
            for x in row:
                d[field_names[i]] = x
                i += 1
            D.append(d)
        row_num += 1
    return D

def is_numeric_string(s):
    return s.replace('.', '', 1).isdigit()

def trim_features(data_dict, remove_features):
    print("remove features: ", remove_features)

    for remove_feature in remove_features:
        for i in range(len(data_dict)):
            del data_dict[i][remove_feature]
    return data_dict


def data_dict_to_dataset(data_dict, label_field_name):
    X_field_order = []
    for d_feature in data_dict[0]:
        if d_feature != label_field_name:
            X_field_order.append(d_feature)

    if len(X_field_order) == len(data_dict[0]):
        raise ValueError("label field name never appears")

    X = np.zeros((len(data_dict), len(data_dict[0]) - 1), dtype = np.float64)

    y = np.zeros(len(data_dict), dtype = np.float64)
    for i in range(len(data_dict)):
        d = data_dict[i]
        #print("len X_field_order: ", len(X_field_order))
        #print("len(data_dict[i]): ", len(data_dict[i]))
        for j in range(len(X_field_order)):
            X[i,j] = float(d[X_field_order[j]])
        y[i] = float(d[label_field_name])


    return X, y, X_field_order

def normalize_features(X):
    X -= np.min(X, axis = 1)[:,np.newaxis]
    X /= np.max(X, axis = 1)[:,np.newaxis]


def normalize_rates(y):
    y -= y.min()
    y /= y.max()


def get_remove_fields(data_dict, field_names):
    remove_features = []
    for d in data_dict:
        for feature in d:
            if not feature in remove_features and not is_numeric_string(d[feature]):
                remove_features.append(feature)

    for field_name in field_names:
        if field_name not in remove_features and \
            ("numerator" in field_name \
            or "denominator" in field_name\
            or "FIPS" in field_name\
            or "Year" in field_name\
            or "CI" in field_name):

            remove_features.append(field_name)
    return remove_features




DATA_PATH = "C:/Users/peter/OneDrive/Desktop/ML/CDS/stanford_medical/datasets/super_clean_analytic_data2019.csv"
FIELD_NAMES = read_fields(open_csv(DATA_PATH), 0)
DATA_DICT = get_data_as_dicts(open_csv(DATA_PATH), [0,1], FIELD_NAMES)
REMOVE_FIELDS = get_remove_fields(DATA_DICT, FIELD_NAMES)
DATA_DICT = trim_features(DATA_DICT, REMOVE_FIELDS)
X, y, X_field_order = data_dict_to_dataset(DATA_DICT, "Premature age-adjusted mortality raw value")

normalize_features(X)
normalize_rates(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)

model = SquareErrorDecisionTreeBoostedRegressor((2, 4), (X.shape[1], X.shape[1]), (0.8,1.0))
model.train(X_train, y_train, 1000)
print("test error: ", np.average(np.square(model.predict(X_test) - y_test)))

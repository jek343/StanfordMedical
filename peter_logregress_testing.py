from ml.model.density_regression.gini_penalized.gini_penalized_logistic_regression import GiniPenalizedLogisticRegression
from ml.model.density_regression.square_error_penalized.square_error_penalized_logistic_regression import SquareErrorPenalizedLogisticRegression
from ml.model.density_regression.class_sampled_error_penalized.class_sampled_error_penalized_logistic_regression import ClassSampledErrorPenalizedLogisticRegression
import ml.function.sigmoid as sigmoid
import numpy as np
import csv


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


    for remove_feature in remove_features:
        for i in range(len(data_dict)):
            del data_dict[i][remove_feature]
    return data_dict


def data_dict_to_dataset(data_dict, label_field_name):
    X_field_order = []
    for d_feature in data_dict[0]:
        if d_feature != label_field_name:
            X_field_order.append(d_feature)


    X = np.zeros((len(data_dict), len(data_dict[0]) - 1), dtype = np.float64)

    y = np.zeros(len(data_dict), dtype = np.float64)
    for i in range(len(data_dict)):
        d = data_dict[i]
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
        if "numerator" in field_name \
            or "denominator" in field_name\
            or "FIPS" in field_name\
            or "Year" in field_name\
            or "CI" in field_name:

            remove_features.append(field_name)
    return remove_features




DATA_PATH = "C:/Users/peter/OneDrive/Desktop/ML/CDS/stanford_medical/datasets/clean_analytic_data2019.csv"
FIELD_NAMES = read_fields(open_csv(DATA_PATH), 0)
DATA_DICT = get_data_as_dicts(open_csv(DATA_PATH), [0,1], FIELD_NAMES)
REMOVE_FIELDS = get_remove_fields(DATA_DICT, FIELD_NAMES)
DATA_DICT = trim_features(DATA_DICT, REMOVE_FIELDS)
X, y, X_field_order = data_dict_to_dataset(DATA_DICT, "Mortality Rati")

normalize_features(X)
normalize_rates(y)
model = ClassSampledErrorPenalizedLogisticRegression("l1")

print("X: ", X.shape)
model.train(X, y, .001, .01, 35000, 100)

f_X = model.f(X)
for i in range(X.shape[0]):
    print("y[i]: " + str(y[i]) + ", f(x[i]): " + str(f_X[i]))
print("mean L1: ", np.sum(np.abs(f_X - y))/float(X.shape[0]))

weight_dict = []
for i in range(len(X_field_order)):
    weight_dict.append((X_field_order[i], model.get_params()[i]))
weight_dict.sort(key = lambda x: abs(x[1]), reverse = True)

for (feature, weight) in weight_dict:
    print(feature + ": " + str(weight))
print("bias: ", model.get_params()[-1])

'''
X_arr = []
y = []
populations = []
data_csv_file = open(DATA_PATH)
data_csv = csv.reader(data_csv_file, delimiter=',')

row_num = 0
SKIP_FIRST_ROWS = 2
POPULATION_COLUMN = 54-1
for row in data_csv:
    if row_num >= SKIP_FIRST_ROWS:
        x = []
        for row_elem in row:

            if row_elem.replace('.', '', 1).isdigit():
                x.append(float(row_elem))

        populations.append(float(row[POPULATION_COLUMN]))
        x.append(1.0)
        X_arr.append(x)
        y.append(float(row[-1]))
    row_num += 1


X = np.zeros((len(X_arr), len(X_arr[0])))
for i in range(X.shape[0]):
    X[i] = np.asarray(X_arr[i])
X = X.astype(np.float64)

X -= np.min(X, axis = 1)[:,np.newaxis]
X /= np.max(X, axis = 1)[:,np.newaxis]
print("X column sums: ", np.max(X, axis = 1))
X[:,-1] = 1.0

y = np.asarray(y).astype(np.float64)
y -= y.min()
y /= y.max()
point_probs = np.asarray(populations).astype(np.float64)
point_probs /= point_probs.sum()


model = ClassSampledErrorPenalizedLogisticRegression("l1")

print("X: ", X.shape)
model.train(X, y, .00005, 1.0, 200000, 100)

f_X = model.f(X)
for i in range(X.shape[0]):
    print("y[i]: " + str(y[i]) + ", f(x[i]): " + str(f_X[i]))
print("mean L1: ", np.sum(np.abs(f_X - y))/float(X.shape[0]))
'''

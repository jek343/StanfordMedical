import csv
import os
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import pandas as pd


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


def get_remove_fields(data_dict, field_names):
    remove_features = []
    for d in data_dict:
        for feature in d:
            if feature not in remove_features and not is_numeric_string(d[feature]):
                remove_features.append(feature)

    for field_name in field_names:
        if field_name not in remove_features and \
            ("numerator" in field_name
                or "denominato" in field_name
                or "FIPS" in field_name
                or "Year" in field_name
                or "CI" in field_name):
            remove_features.append(field_name)

    return remove_features


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

    X = np.zeros((len(data_dict), len(data_dict[0]) - 1), dtype=np.float64)

    y = np.zeros(len(data_dict), dtype=np.float64)
    for i in range(len(data_dict)):
        d = data_dict[i]
        for j in range(len(X_field_order)):
            X[i, j] = float(d[X_field_order[j]])
        y[i] = float(d[label_field_name])

    return X, y, X_field_order


def get_remove_rows(csv, field_names):
    row_num = 0
    remove_rows = [0]
    for row in csv:
        if row[1] == '000':
            remove_rows.append(row_num)
        row_num += 1
    return remove_rows


DATA_PATH = os.path.join(os.getcwd(),  '..', 'datasets', 'super_clean_analytic_data2019.csv')
FIELD_NAMES = read_fields(open_csv(DATA_PATH), 0)
REMOVE_ROWS = get_remove_rows(open_csv(DATA_PATH), FIELD_NAMES)
DATA_DICT = get_data_as_dicts(open_csv(DATA_PATH), REMOVE_ROWS, FIELD_NAMES)
REMOVE_FIELDS = get_remove_fields(DATA_DICT, FIELD_NAMES)
DATA_DICT = trim_features(DATA_DICT, REMOVE_FIELDS)
X, y, X_field_order = data_dict_to_dataset(DATA_DICT, "Premature age-adjusted mortality raw value")

# print(preprocessing.scale(X).std(axis=0))
# print(X.std(axis=0))
X = pd.DataFrame(data=preprocessing.scale(X), columns=X_field_order)
y = pd.DataFrame(data=y, columns=["Mortality Ratio"])
# print(X)

X -= np.min(X, axis=1)[:, np.newaxis]
X /= np.max(X, axis=1)[:, np.newaxis]

y -= y.min()
y /= y.max()
# y *= 10


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
y_train = y_train.iloc[:, 0]
y_test = y_test.iloc[:, 0]
# clf = LogisticRegression(penalty = 'l2', C = 1000.0).fit(X_train, y_train)
# clf = Lasso(alpha=1.0, fit_intercept=True)  # l1
# clf = Ridge(alpha=1.0, fit_intercept=True)  # l2
clf = LinearRegression()
clf = clf.fit(X_train, y_train)

# for i in range(len(X_field_order)):
#     if abs(clf.coef_[i]) == 0:
#         # print((X_field_order[i], clf.coef_[i]))
#         print(X_field_order[i])

pred_y = clf.predict(X_test)  # [:,0]
print('mean absolute error', mean_absolute_error(y_test, pred_y))
print('r2', r2_score(y_test, pred_y))
#  clf.score(X_test, y_test)
# print(('prediction', 'mortality ratio'))
# for i in range(20):
#     print((np.array(pred_y)[i], np.array(y_test)[i]))

print("bias", clf.intercept_)

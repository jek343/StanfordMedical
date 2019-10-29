import csv
import os
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


include_features_paper = ["% Rural raw value", "Population raw value",
                    "% Females raw value", "% below 18 years of age raw value",
                    "% 65 and older raw value",
                    "% Non-Hispanic African American raw value",
                    "% Hispanic raw value", "% Asian raw value",
                    "% American Indian and Alaskan Native raw value",
                    "% Native Hawaiian/Other Pacific Islander raw value",
                    "Median household income raw value",
                    "Some college raw value",
                    "Food insecurity raw value",
                    "Unemployment raw value",
                    "Severe housing problems raw value",
                    "Uninsured raw value",
                    "Primary care physicians raw value",
                    "Access to exercise opportunities raw value",
                    "Food environment index raw value",
                    "Premature age-adjusted mortality raw value"]

include_features_brfs = ["Premature age-adjusted mortality raw value",
"Poor physical health days raw value", "Poor mental health days raw value",
"Adult smoking raw value","Adult obesity raw value", "Physical inactivity raw value",
"Access to exercise opportunities raw value", "Excessive drinking raw value",
"Sexually transmitted infections raw value", "Teen births raw value", "Diabetes prevalence raw value",
"Insufficient sleep raw value", "Social associations raw value"]

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
                or "CI" in field_name
                or field_name.lower() == "premature death raw value"
                or field_name.lower() == "life expectancy raw value"
                or field_name.lower() == "injury deaths raw value"
                # or field_name == "Premature age-adjusted mortality raw value"
                or field_name == "County Ranked (Yes=1/No=0)"):
        # if field_name not in remove_features and field_name not in include_features_paper:
        #     # if "Rural" in field_name or "rural" in field_name:
        #     #     print(field_name)
            remove_features.append(field_name)

    return remove_features


def trim_features(data_dict, remove_features):
    for remove_feature in remove_features:
        for i in range(len(data_dict)):
            # print(data_dict[i])
            del data_dict[i][remove_feature]

    return data_dict


def data_dict_to_dataset(data_dict, label_field_name):
    X_field_order = []
    for d_feature in data_dict[0]:
        if d_feature != label_field_name:
            X_field_order.append(d_feature)

    print(X_field_order)

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
X, y, X_field_order = data_dict_to_dataset(DATA_DICT, "Alcohol-impaired driving deaths raw value")

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

#calculate correlation between X and y and print in decreasing magnitude
Xy = pd.concat([X, y], axis=1)
correlation = Xy.corr()[y.columns[0]][:]
order = correlation.map(lambda x : abs(x)).sort_values(ascending = False)
# for i in order.index.values[1:]:
#     print(X_field_order.index(i), i, correlation[i])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
y_train = y_train.iloc[:, 0]
y_test = y_test.iloc[:, 0]

clf = LinearRegression()
clf1 = Lasso(alpha=0.0001, fit_intercept=True)  # l1
clf2 = Ridge(alpha=0.1, fit_intercept=True)  # l2

clf = clf.fit(X_train, y_train)
clf1 = clf1.fit(X_train, y_train)
clf2 = clf2.fit(X_train, y_train)

pred_y = clf.predict(X_test)  # [:,0]
pred_y1 = clf1.predict(X_test)
pred_y2 = clf2.predict(X_test)


#  clf.score(X_test, y_test)
# print(('prediction', 'mortality ratio'))
# for i in range(20):
#     print((np.array(pred_y)[i], np.array(y_test)[i]))
print('mean absolute error', mean_absolute_error(y_test, pred_y))
print('r2', r2_score(y_test, pred_y))
print("bias", clf.intercept_)

print('mean absolute error 1', mean_absolute_error(y_test, pred_y1))
print('r2 1', r2_score(y_test, pred_y1))
print("bias 1", clf1.intercept_)

print('mean absolute error 2', mean_absolute_error(y_test, pred_y2))
print('r2 2', r2_score(y_test, pred_y2))
print("bias 2", clf2.intercept_)

fig, ax = plt.subplots()
ax.scatter(range(len(clf.coef_)), clf.coef_, s = 5)
# print if weight >.1
for i, txt in enumerate(clf.coef_):
    if abs(txt) > 0.1:
        ax.annotate(i, (i+0.5,txt), fontsize=7)
        print(i, X_field_order[i], txt)
# print all weights
# for i, txt in enumerate(clf.coef_):
#     # ax.annotate(i, (i+0.5,txt), fontsize=7)
#     print(i, X_field_order[i], txt)


plt.xlabel("Index of feature")
plt.ylabel("Weight Value")
plt.title("Unregularized Linear Regression Weight Value vs Index of feature")
plt.legend(["R^2: " + str(round(r2_score(y_test, pred_y),2))], loc="lower right")
plt.savefig('weights.png')
plt.clf()

fig, ax = plt.subplots()
ax.scatter(range(len(clf1.coef_)),clf1.coef_, s = 5, color="orange")
for i, txt in enumerate(clf1.coef_):
    if abs(txt) > 0.1:
        ax.annotate(i, (i+0.5,txt), fontsize=7)
        print(i, X_field_order[i], txt)

plt.xlabel("Index of feature")
plt.ylabel("Weight Value")
plt.title("L1 Linear Regression Weight Value vs Index of feature")
plt.legend(["R^2: " + str(round(r2_score(y_test, pred_y1),2))], loc="lower right")
plt.savefig('weights1.png')
plt.clf()

fig, ax = plt.subplots()
ax.scatter(range(len(clf2.coef_)),clf2.coef_, s = 5, color="green")
for i, txt in enumerate(clf2.coef_):
    if abs(txt) > 0.1:
        ax.annotate(i, (i+0.5,txt), fontsize=7)
        print(i, X_field_order[i], txt)

plt.legend(["R^2: " + str(round(r2_score(y_test, pred_y2),2))], loc="lower right")
plt.xlabel("Index of feature")
plt.ylabel("Weight Value")
plt.title("L2 Linear Regression Weight Value vs Index of feature")
plt.savefig('weights2.png')
plt.clf()

plt.bar(range(len(clf1.coef_)), correlation[:-1])
plt.xlabel("Index of feature")
plt.ylabel("Correlation between feature and mortality")
plt.title("Correlation between feature and mortality vs Index of feature")
plt.savefig("corr.png")
plt.clf()

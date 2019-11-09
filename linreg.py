import csv
import os
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_YEAR = 2019

predict = "Premature age-adjusted mortality raw value"

include_features_paper = [predict, "% Rural raw value", "Population raw value",
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
                    "Food environment index raw value", "% Missing entries"]

include_features_brfs = [predict,
                    "Poor physical health days raw value",
                    "Poor mental health days raw value",
                    "Adult smoking raw value","Adult obesity raw value",
                    "Physical inactivity raw value",
                    "Access to exercise opportunities raw value",
                    "Excessive drinking raw value",
                    "Sexually transmitted infections raw value",
                    "Teen births raw value", "Diabetes prevalence raw value",
                    "Insufficient sleep raw value",
                    "Social associations raw value", "% Missing entries"]

#if fields is [], will use all the usable & not obviously correlated features
fields = include_features_paper
coef = "paper"

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
                field = field_names[i]
                d[field] = x
                i += 1
            D.append(d)
        row_num += 1
    return D


def is_numeric_string(s):
    s.replace('.', '', 1).isdigit()
    s.replace('E', '', 1).isdigit()
    s.replace('-', '', 1).isdigit()
    return s


def get_remove_fields(data_dict, field_names, field_subset):
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
                or field_name == "County Ranked (Yes=1/No=0)"
                or field_name == "State Abbreviation"
                or field_name == "Name"
                or field_name not in field_subset):
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
    '''Remove state and country level data.
    Remove rows that the prediction is blank (i.e. '0')'''
    pidx = field_names.index(predict)
    row_num = 0
    remove_rows = [0]
    for row in csv:
        if row[1] == '000' or row[pidx] == '0':
            remove_rows.append(row_num)
        row_num += 1
    return remove_rows


DATA_PATH = os.path.join(os.getcwd(),  '..', 'datasets', 'super_clean_analytic_data_missing' + str(DATA_YEAR) + '.csv')
FIELD_NAMES = read_fields(open_csv(DATA_PATH), 0)

possible_y = ["Premature death raw value", "Life expectancy raw value",
"Injury deaths raw value", "Premature age-adjusted mortality raw value",
"Alcohol-impaired driving deaths raw value"]

if len(fields) == 0:
    fields = [FIELD_NAMES[i] for i in range(len(FIELD_NAMES))]
    for y in possible_y:
        if y != predict:
            try:
                fields.remove(y)
            except ValueError:
                pass


REMOVE_ROWS = get_remove_rows(open_csv(DATA_PATH), FIELD_NAMES)
DATA_DICT = get_data_as_dicts(open_csv(DATA_PATH), REMOVE_ROWS, FIELD_NAMES)

REMOVE_FIELDS = get_remove_fields(DATA_DICT, FIELD_NAMES, fields)

DATA_DICT = trim_features(DATA_DICT, REMOVE_FIELDS)

X, y, X_field_order = data_dict_to_dataset(DATA_DICT, predict)

def get_clean_data():
    return X, y, X_field_order

X = pd.DataFrame(data=preprocessing.scale(X), columns=X_field_order)
y = pd.DataFrame(data=y, columns=[predict])
Xy = pd.concat([X, y], axis=1)

X -= np.min(X, axis=1)[:, np.newaxis]
X /= np.max(X, axis=1)[:, np.newaxis]

y -= y.min()
y /= y.max()

#calculate correlation between X and y and print in decreasing magnitude
def corr_coef_Xy_dec_mag(Xy):
    correlation = Xy.corr()[y.columns[0]][:]
    order = correlation.map(lambda x : abs(x)).sort_values(ascending = False)
    #printing the correlation coefficient matrix
    for i in order.index.values[1:]:
        print(X_field_order.index(i), i, correlation[i])
    #create bar graph of coefficients
    plt.bar(range(len(clf1.coef_)), correlation[:-1])
    plt.xlabel("Index of feature")
    plt.ylabel("Correlation between feature and mortality")
    plt.title("Correlation between feature and mortality vs Index of feature")
    plt.savefig("corr.png")
    plt.clf()

def create_coef_map(shrink, filename, label_font_size, coef_font_size, fig_size, title_size):
    cols = list(Xy.columns)
    stdsc = StandardScaler()
    X_std = stdsc.fit_transform(Xy[cols].iloc[:,range(len(cols))].values)
    cov_mat =np.cov(X_std.T)
    plt.figure(figsize=(fig_size,fig_size))
    sns.set(font_scale=label_font_size)
    hm = sns.heatmap(cov_mat,
                     cbar=True,
                     cbar_kws={"shrink": shrink},
                     annot=True,
                     square=True,
                     fmt='.2f',
                     annot_kws={'size': coef_font_size},
                     cmap='coolwarm',
                     yticklabels=[x[:-10] for x in cols],
                     xticklabels=[x[:-10] for x in cols])
    plt.title('Covariance matrix showing correlation coefficients', size = title_size)
    plt.tight_layout()
    plt.savefig(filename + ".png")
    plt.cla()
    g = sns.pairplot(Xy[cols], size=2.0)
    g.set(xticklabels=[], yticklabels = [])
    plt.savefig(filename + "_pair.png")
    plt.cla()

brfs = {"shrink":0.71, "filename": "coef_brfs", "label_font_size":1.5, "coef_font_size": 12, "fig_size":10, "title_size":18}
paper = {"shrink":0.82, "filename": "coef_paper", "label_font_size":0.95, "coef_font_size": 8, "fig_size":10, "title_size":18}
all = {"shrink":0.82, "filename": "coef_all", "label_font_size":1.0, "coef_font_size": 10, "fig_size":30, "title_size":30}

dict = all
if coef=="paper":
    dict = paper
elif coef=="brfs":
    dict = brfs

create_coef_map(dict["shrink"], dict["filename"], dict["label_font_size"], dict["coef_font_size"], dict["fig_size"], dict["title_size"])

#splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
y_train = y_train.iloc[:, 0]
y_test = y_test.iloc[:, 0]

#creating the models
clf = LinearRegression()
clf1 = Lasso(alpha=0.0001, fit_intercept=True)  # l1
clf2 = Ridge(alpha=0.1, fit_intercept=True)  # l2

#fitting the models
clf = clf.fit(X_train, y_train)
clf1 = clf1.fit(X_train, y_train)
clf2 = clf2.fit(X_train, y_train)

#predicting the outputs
pred_y = clf.predict(X_test)  # [:,0]
pred_y1 = clf1.predict(X_test)
pred_y2 = clf2.predict(X_test)


#  clf.score(X_test, y_test)
# print(('prediction', 'mortality ratio'))
# for i in range(20):
#     print((np.array(pred_y)[i], np.array(y_test)[i]))

#analyzing performance of models
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

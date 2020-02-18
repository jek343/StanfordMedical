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

DATA_YEAR = 2018
PREDICT_YEAR = 2019

create_map = True

predict = "Premature age-adjusted mortality raw value"

possible_y = ["Premature death raw value", "Life expectancy raw value",
"Injury deaths raw value", "Premature age-adjusted mortality raw value",
"Alcohol-impaired driving deaths raw value"]

assert predict in possible_y
assert DATA_YEAR <= PREDICT_YEAR
assert DATA_YEAR >= 2013 and DATA_YEAR <= 2019
assert PREDICT_YEAR >= 2013 and PREDICT_YEAR <= 2019

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

def open_csv(path):
    data_csv_file = open(path)
    return csv.reader(data_csv_file, delimiter=',')


DATA_PATH = os.path.join(os.getcwd(),  '..', 'datasets', 'super_clean_analytic_data' + str(DATA_YEAR) + '.csv')
P_DATA_PATH = os.path.join(os.getcwd(),  '..', 'datasets', 'super_clean_analytic_data' + str(PREDICT_YEAR) + '.csv')

prev_year = pd.read_csv(DATA_PATH)
curr_year = pd.read_csv(P_DATA_PATH)

def remove_rows_cols(data):
    #rename row indexes to fips code
    data.rename(index = dict(zip(data.index, data['5-digit FIPS Code'])), inplace = True)
    #drop rows that do not have mortality
    data.dropna(axis='index', subset = [predict], inplace = True)
    #drop rows and columns that have missing values
    data.dropna(inplace = True)
    #only include numeric columns
    data.select_dtypes(include=[np.number])
    cols = data.columns
    cols_to_remove = filter(lambda col: "numerator" in col or "denominato" in col or \
                            "FIPS" in col or "CI" in col or "Year" in col or \
                            col in ["County Ranked (Yes=1/No=0)", "State Abbreviation", "Name"] or \
                            (col in possible_y and col != predict), cols)
    data.drop(columns=list(cols_to_remove), inplace = True)
    return data

curr_year = remove_rows_cols(curr_year)
prev_year = remove_rows_cols(prev_year)

def data_both_years(prev_year, curr_year):
    '''Keeps county info (rows) that are in both years' datasets using 5-digit FIPS code as unique identifier'''
    #only keep counties that are in both years
    curr_rows = set(curr_year.index)
    prev_rows = set(prev_year.index)
    indexes = curr_rows.intersection(prev_rows)
    drop_curr = curr_rows - indexes
    drop_prev = prev_rows - indexes
    prev_year.drop(index=list(drop_prev), inplace = True)
    curr_year.drop(index=list(drop_curr), inplace = True)

    #only keep columns that are in both years
    curr_cols = set(curr_year.columns)
    prev_cols = set(prev_year.columns)
    cols = curr_cols.intersection(prev_cols)
    drop_curr = curr_cols - cols
    drop_prev = prev_cols - cols
    prev_year.drop(columns=list(drop_prev), inplace = True)
    curr_year.drop(columns=list(drop_curr), inplace = True)
    return prev_year, curr_year

prev_year, curr_year = data_both_years(prev_year, curr_year)

def get_x(year):
    '''Gets all the normalized "x" data (socioeconomic & demographic, not the mortality) for year
    Note: columns are still labeled with FIPS'''
    x = year.drop(columns=[predict]) #note need inplace false (default)
    x -= np.min(x, axis=0)
    x /= np.max(x, axis=0)
    return x

def get_y(year):
    '''Gets the normalized mortality for the given year
    Note: columns are still labeled with FIPS'''
    y = year[predict]
    y -= np.min(y, axis=0)
    y /= np.max(y, axis=0)
    return y

prev_x = get_x(prev_year)
curr_y = get_y(curr_year)
x_train, x_test, y_train, y_test = train_test_split(prev_x, curr_y, test_size=0.2, random_state=0)

def deltas(prev_year, curr_year):
    prev_y = prev_year[predict]
    curr_y = curr_year[predict]
    curr_y -= np.min(curr_y, axis=0)
    curr_y /= np.max(curr_y, axis=0)
    prev_year.drop(columns=[predict], inplace = True)
    curr_year.drop(columns=[predict], inplace = True)
    delta = curr_year - prev_year
    delta = pd.concat([delta, prev_y], axis=1)
    delta -= np.min(delta, axis=0)
    delta /= np.max(delta, axis=0)
    delta.dropna(axis='columns', how='all', inplace = True)
    delta.fillna(value=0.0, inplace=True)
    delta_X = delta
    return delta_X, curr_y


delta_X, delta_Y = deltas(prev_year, curr_year)
x_train_d, x_test_d, y_train_d, y_test_d= train_test_split(delta_X, delta_Y, test_size=0.2, random_state=0)

#creating the models
clf = LinearRegression()
clf1 = Lasso(alpha=0.0001, fit_intercept=True)  # l1
clf2 = Ridge(alpha=0.1, fit_intercept=True)  # l2

#fitting the models
clf = clf.fit(x_train, y_train)
clf1 = clf1.fit(x_train, y_train)
clf2 = clf2.fit(x_train, y_train)

#predicting the outputs
pred_y = clf.predict(x_test)
pred_y1 = clf1.predict(x_test)
pred_y2 = clf2.predict(x_test)

#analyzing performance of models
def print_performance(title, actual, prediction, clf, train):
    print("\n" + title)
    print('mean absolute error', mean_absolute_error(actual, prediction))
    print('r2', r2_score(actual, prediction))
    print("bias", clf.intercept_)

    fig, ax = plt.subplots()
    ax.scatter(range(len(clf.coef_)), clf.coef_, s = 5)
    # print if weight >.05
    for i, txt in enumerate(clf.coef_):
        if abs(txt) > 0.05:
            ax.annotate(i, (i+0.5,txt), fontsize=7)
            print(i, train.columns[i], txt)


    plt.xlabel("Index of feature")
    plt.ylabel("Weight Value")
    plt.title(title + " Linear Regression Weight Value vs Index of feature")
    plt.legend(["R^2: " + str(round(r2_score(actual, prediction),2))], loc="lower right")
    plt.savefig('weights.png')
    plt.clf()

print_performance("Unregularized", y_test, pred_y, clf, x_train)
print_performance("L1", y_test, pred_y1, clf1, x_train)
print_performance("L2", y_test, pred_y2, clf2, x_train)

if PREDICT_YEAR != DATA_YEAR:
    clf_d = LinearRegression()
    clf1_d = Lasso(alpha=0.0001, fit_intercept=True)  # l1
    clf2_d = Ridge(alpha=0.1, fit_intercept=True)  # l2

    clf_d = clf_d.fit(x_train_d, y_train_d)
    clf1_d = clf1_d.fit(x_train_d, y_train_d)
    clf2_d = clf2_d.fit(x_train_d, y_train_d)

    pred_y_d = clf_d.predict(x_test_d)
    pred_y1_d = clf1_d.predict(x_test_d)
    pred_y2_d = clf2_d.predict(x_test_d)

    print_performance("Delta Unregularized", y_test_d, pred_y_d, clf_d, x_train_d)
    print_performance("Delta L1", y_test_d, pred_y1_d, clf1_d, x_train_d)
    print_performance("Delta L2", y_test_d, pred_y2_d, clf2_d, x_train_d)


def create_coef_map():
    '''Creates a figure that shows the correlation coefficients between all the features and the feature being predicted'''
    xy = pd.concat([prev_x, curr_y], axis=1)
    cols = list(xy.columns)
    stdsc = StandardScaler()
    X_std = stdsc.fit_transform(xy.to_numpy())
    cov_mat =np.cov(X_std.T)
    cov_mat_row = cov_mat[:,-1]
    indexes = np.argsort(-cov_mat_row)
    mid = int(len(indexes) / 2)
    cols = np.array(cols)[indexes]
    cov_mat_row = cov_mat_row[indexes]
    fig, (ax,ax2) = plt.subplots(ncols=2, figsize=(10, 5))
    fig.subplots_adjust(wspace=0.01)
    sns.heatmap(np.expand_dims(cov_mat_row[:mid], axis=1),
                    ax = ax,
                     cbar=False,
                     annot=True,
                     fmt='.2f',
                     cmap='Reds',
                     yticklabels=[x[:-10] for x in cols[:mid]],
                     xticklabels=False)
    sns.heatmap(np.expand_dims(cov_mat_row[mid:], axis=1),
                    ax = ax2,
                     cbar=False,
                     annot=True,
                     fmt='.2f',
                     cmap='Blues',
                     yticklabels=[x[:-10] for x in cols[mid:]],
                     xticklabels=False)

    ax2.yaxis.tick_right()
    ax2.tick_params(rotation=0)
    fig.suptitle('Correlation coefficients between features and ' + predict[:-10])#, size = title_size)
    fig.canvas.draw()
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("coef.png")
    plt.cla()

if create_map:
    create_coef_map()

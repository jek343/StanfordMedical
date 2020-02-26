import csv
import os
import sys
from xgboost import XGBRegressor
'''
Note: to install xgboost on mac
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost; cp make/minimum.mk ./config.mk; make -j4
cd python-package; python setup.py develop --user
'''
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV,  LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_YEAR = 2018
PREDICT_YEAR = 2019
X_DELTA = False
XY_DELTA = False
CV = True

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
    '''Returns the csv reader for the file at the specified path'''
    data_csv_file = open(path)
    return csv.reader(data_csv_file, delimiter=',')


def remove_rows_cols(data):
    '''Renames row indexes to fips code unique identifier, drops counties that
    do not report mortality then drops rows and columns with empty cells,
    non-numeric columns, numerators, denominators, confidence intervals,
    county ranked, state abbreviation, name, and obviously related features'''
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


def data_both_years(prev_year, curr_year):
    '''Keeps county info (rows) that are in both years' datasets using 5-digit
    FIPS code as unique identifier'''
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


def x_deltas(prev_year, curr_year):
    '''Returns the normalized deltas between curr_year and prev_year socioeconomic &
    demographic data concatenated with prev_year mortality to be used as the
    x dataset and the normalized curr_year mortality to be used as the y dataset'''
    prev_y = prev_year[predict]
    curr_y = curr_year[predict]
    curr_y -= np.min(curr_y, axis=0)
    curr_y /= np.max(curr_y, axis=0)
    p_year = prev_year.drop(columns=[predict])
    c_year = curr_year.drop(columns=[predict])
    x = c_year - p_year
    x = pd.concat([x, prev_y], axis=1)
    x -= np.min(x, axis=0)
    x /= np.max(x, axis=0)
    x.dropna(axis='columns', how='all', inplace = True)
    x.fillna(value=0.0, inplace=True)
    return x, curr_y

def xy_deltas(prev_year, curr_year):
    '''Returns the normalized deltas between curr_year and prev_year socioeconomic &
    demographic data to be used as the x dataset and the deltas between curr_year
    and prev_year mortality to be used as the y dataset'''
    prev_y = prev_year[predict]
    curr_y = curr_year[predict]
    y = curr_y - prev_y
    y -= np.min(y, axis=0)
    y /= np.max(y, axis=0)
    p_year = prev_year.drop(columns=[predict])
    c_year = curr_year.drop(columns=[predict])
    x = c_year - p_year
    x -= np.min(x, axis=0)
    x /= np.max(x, axis=0)
    x.dropna(axis='columns', how='all', inplace = True)
    x.fillna(value=0.0, inplace=True)
    return x, y

def model(x, y, x_delta, xy_delta):
    '''Evaluates linear regression, random forest, and gradient boosted random
    forest for unregularized, lasso, and ridge regression.
    Calls print_performance to print out each model's performance.'''
    clf = LinearRegression()
    clf1 = Lasso(alpha=0.0001, fit_intercept=True)  # l1
    clf2 = Ridge(alpha=0.1, fit_intercept=True)  # l2

    if CV:
        cv_results = cross_validate(clf, x, y, cv=5, return_estimator = True)
        cv_results1 = cross_validate(clf1, x, y, cv=5, return_estimator = True)
        cv_results2 = cross_validate(clf2, x, y, cv=5, return_estimator = True)

        scores = cv_results['test_score']
        scores1 = cv_results1['test_score']
        scores2 = cv_results2['test_score']

        print("\nCROSS VALIDATION RESULTS")
        print_performance_cv("Unregularized", scores, cv_results, x.columns)
        print_performance_cv("L1", scores1, cv_results1, x.columns)
        print_performance_cv("L2", scores2, cv_results2, x.columns)

        kfold = KFold(n_splits=5)
        regr = RandomForestRegressor(n_estimators=100, random_state = 0)
        regr_scores = cross_val_score(regr, x, y, cv=kfold)
        print("\nRandom Forest: %0.5f (+/- %0.5f)" % (regr_scores.mean(), regr_scores.std() * 2))

        gb_regr = GradientBoostingRegressor(loss = "huber", learning_rate = 0.1, n_estimators=100, random_state = 0, max_depth = 3)
        gb_regr_scores = cross_val_score(gb_regr, x, y, cv=kfold)
        print("\nGradient Boosted Random Forest: %0.5f (+/- %0.5f)" % (gb_regr_scores.mean(), gb_regr_scores.std() * 2))

        xgb = XGBRegressor(loss = "huber", n_estimators=100, learning_rate=0.1, max_depth=3)
        xgb_scores = cross_val_score(xgb, x, y, cv=kfold)
        print("\nExtreme Gradient Boosted Regressor: %0.5f (+/- %0.5f)" % (xgb_scores.mean(), xgb_scores.std() * 2))


        return scores1.mean(), scores1.std() * 2
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

        regr = RandomForestRegressor(n_estimators = 100, random_state=0)
        gb_regr = GradientBoostingRegressor(n_estimators = 100, random_state=0,  max_depth = 5)
        xgb_regr = XGBRegressor(loss = "huber", n_estimators=100, learning_rate=0.1, max_depth=3)

        #fitting the models
        clf = clf.fit(x_train, y_train)
        clf1 = clf1.fit(x_train, y_train)
        clf2 = clf2.fit(x_train, y_train)
        clf3 = regr.fit(x_train, y_train)
        clf4 = gb_regr.fit(x_train, y_train)
        clf5 = xgb_regr.fit(x_train, y_train)

        #predicting the outputs
        pred_y = clf.predict(x_test)
        pred_y1 = clf1.predict(x_test)
        pred_y2 = clf2.predict(x_test)
        pred_y3 = regr.predict(x_test)
        pred_y4 = gb_regr.predict(x_test)
        pred_y5 = xgb_regr.predict(x_test)

        print_performance("Unregularized", y_test, pred_y, clf, x_train, x_delta, xy_delta)
        print_performance("L1", y_test, pred_y1, clf1, x_train, x_delta, xy_delta)
        print_performance("L2", y_test, pred_y2, clf2, x_train, x_delta, xy_delta)
        print("\nRandom Forest", regr.score(x_test, y_test))
        print("\nGradient Boosted Random Forest", gb_regr.score(x_test, y_test))
        print("\nExtreme Gradient Boosted Regressor", xgb_regr.score(x_test, y_test))
    return None

def print_performance_cv(title, scores, cv_results, cols):
    '''When cross validation is used.
    Prints the R^2 and largest weights (absolute value).'''
    print("\n" + title)
    print("R^2: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))
    d = {}
    for fold in cv_results['estimator']:
        for i, txt in enumerate(fold.coef_):
            if abs(txt) > 0.05:
                if cols[i] in d:
                    d[cols[i]].append(txt)
                else:
                    d[cols[i]] = [txt]

    for key, lst in d.items():
        if len(lst) >= 3:
            print(key + " : %0.5f (+/- %0.5f)" % (np.mean(lst), np.std(lst) * 2))


#analyzing performance of models
def print_performance(title, actual, prediction, clf, train, x_delta, xy_delta):
    '''When cross validation is not used.
    Prints the R^2, MAE, bias, and largest weights (absolute value).
    Saves a scatter plot of the weights'''
    print("\n" + title)
    print('Mean absolute error', mean_absolute_error(actual, prediction))
    print('R^2', r2_score(actual, prediction))
    print("Bias", clf.intercept_)

    fig, ax = plt.subplots()
    ax.scatter(range(len(clf.coef_)), clf.coef_, s = 5)
    # print if weight >.05
    for i, txt in enumerate(clf.coef_):
        if abs(txt) > 0.05:
            ax.annotate(i, (i+0.5,txt), fontsize=7)
            print(train.columns[i], ":", round(txt, 5))

    plt.xlabel("Index of feature")
    plt.ylabel("Weight Value")
    d = ""
    if x_delta:
        d = "XDelta "
    elif xy_delta:
        d = "XYDelta "
    plt.title(d + title + " Linear Regression Weight Value vs Index of feature")
    plt.legend(["R^2: " + str(round(r2_score(actual, prediction),2))], loc="lower right")
    plt.savefig(d + title + 'weights.png')
    plt.clf()


def categorize_counties():
    '''Categorizes counties into one of five groups (in relation to predict):
        1. accelerating
        2. decelerating
        3. stable
        4. increasing
        5. decreasing'''
    return

DATA_PATH = os.path.join(os.getcwd(),  '..', 'datasets', 'super_clean_analytic_data' + str(DATA_YEAR) + '.csv')
P_DATA_PATH = os.path.join(os.getcwd(),  '..', 'datasets', 'super_clean_analytic_data' + str(PREDICT_YEAR) + '.csv')

prev_year = pd.read_csv(DATA_PATH)
curr_year = pd.read_csv(P_DATA_PATH)

curr_year = remove_rows_cols(curr_year)
prev_year = remove_rows_cols(prev_year)

prev_year, curr_year = data_both_years(prev_year, curr_year)

prev_x = get_x(prev_year)
prev_y = get_y(prev_year)
curr_y = get_y(curr_year)
print("\nR^2 between prev y and curr y:", r2_score(curr_y, prev_y))

model(prev_x, curr_y, False, False)

if X_DELTA and PREDICT_YEAR != DATA_YEAR:
    delta_X, delta_Y = x_deltas(prev_year, curr_year)

    print("\nX DELTA RESULTS")
    model(delta_X, delta_Y, True, False)

if XY_DELTA and PREDICT_YEAR != DATA_YEAR:
    xydelta_x, xydelta_y = xy_deltas(prev_year, curr_year)

    print("\nXY DELTA RESULTS")
    model(xydelta_x, xydelta_y, False, True)

#-------------------------VISUALIZATIONS----------------------------------------
def blockPrint():
    '''Blocks printing to the terminal'''
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    '''Restores printing to the terminal'''
    sys.stdout = sys.__stdout__

def create_coef_map():
    '''Creates a figure that shows the correlation coefficients between all the
    features and the feature being predicted'''
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
    sns.heatmap(np.expand_dims(cov_mat_row[:mid], axis=1), ax = ax,
                cbar=False, annot=True, fmt='.2f', cmap ="Reds",
                yticklabels=[x[:-10] for x in cols[:mid]],
                xticklabels=False)
    sns.heatmap(np.expand_dims(cov_mat_row[mid:], axis=1),
                ax = ax2, cbar=False, annot=True, fmt='.2f', cmap = "Blues",
                yticklabels=[x[:-10] for x in cols[mid:]],
                xticklabels=False)
    ax2.yaxis.tick_right()
    ax2.tick_params(rotation=0)
    fig.suptitle('Correlation coefficients between features and ' + predict[:-10])
    fig.canvas.draw()
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("coef.png")
    plt.cla()

def map_xdeltas():
    '''Creates a heatmap of the improvement of using x_deltas compared to get_x
    and get_y for predict for L1 linear regression'''
    global CV
    CV = True
    DATA_YEARS = [2013, 2014, 2015, 2016, 2017, 2018]
    PREDICT_YEARS = [2019, 2018, 2017, 2016, 2015, 2014]
    diff = np.zeros((6,6))
    mask = np.zeros_like(diff)

    for p_i, pred_yr in enumerate(PREDICT_YEARS):
        for d_i, data_yr in enumerate(DATA_YEARS):
            if data_yr < pred_yr:
                DATA_PATH = os.path.join(os.getcwd(),  '..', 'datasets', 'super_clean_analytic_data' + str(data_yr) + '.csv')
                P_DATA_PATH = os.path.join(os.getcwd(),  '..', 'datasets', 'super_clean_analytic_data' + str(pred_yr) + '.csv')
                prev_year = pd.read_csv(DATA_PATH)
                curr_year = pd.read_csv(P_DATA_PATH)
                curr_year = remove_rows_cols(curr_year)
                prev_year = remove_rows_cols(prev_year)
                prev_year, curr_year = data_both_years(prev_year, curr_year)
                prev_x = get_x(prev_year)
                curr_y = get_y(curr_year)
                blockPrint()
                mean_xy, std2_xy = model(prev_x, curr_y, False, False)
                enablePrint()
                delta_X, delta_Y = x_deltas(prev_year, curr_year)
                blockPrint()
                mean_xdel, std2_xdel = model(delta_X, delta_Y, True, False)
                enablePrint()
                diff[p_i, d_i] = mean_xdel - mean_xy

    mask[np.where(diff == 0)] = True #returns lower triangle
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7, 5))
        ax = sns.heatmap(diff, mask=mask, annot=True, xticklabels=[str(x) for x in DATA_YEARS],
            yticklabels=[str(x) for x in PREDICT_YEARS], fmt="+.3f", cmap = "Greens", vmax=.3, square=True)
    ax.tick_params(rotation=0)
    plt.title("Improvement from x & y to x deltas + prev y & y")
    plt.ylabel("Predict Year")
    plt.xlabel("Data Year")
    plt.savefig("improve.png")
    plt.cla()

def map_xdeltas_r2():
    '''Creates a heatmap of the improvement of using x_deltas compared to
    predicting the current year's mortality as the previous year's mortality
    for predict for L1 linear regression'''
    global CV
    CV = True
    PREDICT_YEARS = [2019, 2018, 2017, 2016, 2015, 2014]
    results = np.zeros((6,1))
    for p_i, pred_yr in enumerate(PREDICT_YEARS):
        data_yr = pred_yr - 1
        DATA_PATH = os.path.join(os.getcwd(),  '..', 'datasets', 'super_clean_analytic_data' + str(data_yr) + '.csv')
        P_DATA_PATH = os.path.join(os.getcwd(),  '..', 'datasets', 'super_clean_analytic_data' + str(pred_yr) + '.csv')
        prev_year = pd.read_csv(DATA_PATH)
        curr_year = pd.read_csv(P_DATA_PATH)
        curr_year = remove_rows_cols(curr_year)
        prev_year = remove_rows_cols(prev_year)
        prev_year, curr_year = data_both_years(prev_year, curr_year)
        delta_X, delta_Y = x_deltas(prev_year, curr_year)
        prev_y = get_y(prev_year)
        curr_y = get_y(curr_year)
        blockPrint()
        mean_xy, std2_xy = model(delta_X, curr_y, True, False)
        enablePrint()
        results[p_i] = mean_xy - r2_score(curr_y, prev_y)
    fig, ax = plt.subplots(figsize=(5, 7))
    ylabels = np.array([str(y) for y in PREDICT_YEARS])
    ax = sns.heatmap(results, cbar=True, annot=True, fmt='+.3f', cmap='Greens',
                yticklabels=ylabels, xticklabels=False)
    ax.tick_params(rotation=0)
    plt.title("Improvement from y r2 to x deltas + prev y & y")
    plt.savefig("improve_r2.png")
    plt.cla()



if create_map:
    create_coef_map()
    map_xdeltas()
    map_xdeltas_r2()

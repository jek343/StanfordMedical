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

predict = "Premature age-adjusted mortality raw value"

possible_y = ["Premature death raw value", "Life expectancy raw value",
"Injury deaths raw value", "Premature age-adjusted mortality raw value",
"Alcohol-impaired driving deaths raw value"]

assert predict in possible_y

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
fields = []
coef = "all"

def open_csv(path):
    data_csv_file = open(path)
    return csv.reader(data_csv_file, delimiter=',')


DATA_PATH = os.path.join(os.getcwd(),  '..', 'datasets', 'super_clean_analytic_data' + str(DATA_YEAR) + '.csv')
P_DATA_PATH = os.path.join(os.getcwd(),  '..', 'datasets', 'super_clean_analytic_data' + str(PREDICT_YEAR) + '.csv')

#---------------------new code
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
    #maybe drop row instead of just filling
    delta.fillna(value=0.0, inplace=True)
    # print(delta.columns)
    # print(len(delta.columns))
    delta_X = delta
    # print(delta_X.head())
    return delta_X, curr_y


delta_X, delta_Y = deltas(prev_year, curr_year)
x_train_d, x_test_d, y_train_d, y_test_d= train_test_split(delta_X, delta_Y, test_size=0.2, random_state=0)

# calculate correlation between X and y and print in decreasing magnitude
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

# create_coef_map(dict["shrink"], dict["filename"], dict["label_font_size"], dict["coef_font_size"], dict["fig_size"], dict["title_size"])

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
    # print if weight >.1
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

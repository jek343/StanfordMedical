import csv
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np
import csvs.csv_ops as csv_ops
import pandas as pd

blank = np.array([]).astype(int)
data_path = os.path.join(os.getcwd(), '..',  'datasets', 'analytic_data2019.csv')
output_data_path = os.path.join(os.getcwd(),  '..', 'datasets', 'cheesed_analytic_data2019.csv')
#clean_data_path = os.path.join(os.getcwd(),  '..', 'datasets', 'clean_analytic_data2019.csv')
if not os.path.exists(output_data_path):
    with open(data_path) as dataset:
         data_reader = csv.reader(dataset, delimiter=',')
         first_row = next(data_reader)
         for row in data_reader:
             row = np.array(row)
             blank = np.union1d(blank, np.where(row == "")[0])
         not_blank = np.setdiff1d(np.arange(len(first_row)), blank)
         e_cols = np.take(first_row, not_blank)

    csv_ops.save_columns(str(data_path), e_cols, 0, str(output_data_path))

with open(output_data_path) as dataset:
    data_reader = csv.reader(dataset, delimiter=',')
    first_row = next(data_reader)
    X_cols = first_row[:-2]
    Y_cols = first_row[-2:]
    #Y_train_cols = first_row[-1]

dataset = pd.read_csv(output_data_path)

non_numeric = ['Name', 'State Abbreviation']
for c in non_numeric:
    labels = dataset[c].astype('category').cat.categories.tolist()
    replace_map_comp = {c : {k:v for k,v in zip(labels, list(range(1,len(labels)+1)))}}
    dataset.replace(replace_map_comp, inplace = True)


X = dataset[X_cols]
y = dataset[Y_cols]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
y_train = y_train.iloc[:, 1]
y_test = y_test.iloc[:, 0]
clf = LogisticRegression(penalty = "l1").fit(X_train, y_train)

pred_y = clf.predict_proba(X_test)[:,1]
print(mean_absolute_error(y_test, pred_y))
#clf.score(X_test, y_test)

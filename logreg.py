import csv
import os
import sklearn.linear_model
import numpy as np
import csvs.csv_ops as csv_ops

blank = np.array([]).astype(int)
data_path = os.path.join(os.getcwd(), '..',  'datasets', 'analytic_data2019.csv')
output_data_path = os.path.join(os.getcwd(),  '..', 'datasets', 'clean_analytic_data2019.csv')
with open(data_path) as dataset:
     data_reader = csv.reader(dataset, delimiter=',')
     first_row = next(data_reader)
     for row in data_reader:
         row = np.array(row)
         blank = np.union1d(blank, np.where(row == ""))

     not_blank = np.setdiff1d(np.arange(len(first_row)), blank)
     print(len(first_row), len(blank), len(not_blank))
     e_cols = np.take(first_row, not_blank)

csv_ops.save_columns(data_path, e_cols, 0, output_data_path)

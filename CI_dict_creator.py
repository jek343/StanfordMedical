import os
import pandas as pd

def get_CI_dict(year):
    '''Year is an integer and the super_clean_analytic_data csv for that year
    has already been created'''
    CI_dict = {}
    dataset = pd.read_csv(os.path.join(os.getcwd(),  '..', 'datasets', 'super_clean_analytic_data' + str(year) + '.csv'))
    features = list(dataset)
    features = [i[:-7] for i in features if "CI low" in i]

    for index, row in dataset.iterrows():
        row_dict = {}
        fips = row["5-digit FIPS Code"]
        for feat in features:
            low = row[feat + " CI low"]
            actual = row[feat + " raw value"]
            high = row[feat + " CI high"]
            row_dict[feat] = (low, actual, high)
        CI_dict[fips] = row_dict
    return CI_dict

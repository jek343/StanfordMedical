import os
import pandas as pd

def get_CI_dict(year):
    '''Year is an integer and the super_clean_analytic_data csv for that year
    has already been created'''
    CI_dict = {}
    dataset = pd.read_csv(os.path.join(os.getcwd(),  '..', 'datasets', 'super_clean_analytic_data' + str(year) + '.csv'))
    features = list(dataset)
    features = [i[:-10] for i in features if "raw value" in i]
    features_CI = [i[:-7] for i in features if "CI low" in i]
    features_num = [i[:-10] for i in features if "numerator" in i]

    for index, row in dataset.iterrows():
        row_dict = {}
        fips = row["5-digit FIPS Code"]
        for feat in features:
            actual = row[feat + " raw value"]
            low = None
            high = None
            numerator = None
            denom = None
            if feat in features_CI:
                low = row[feat + " CI low"]
                high = row[feat + " CI high"]
            elif feat in features_num:
                numerator = row[feat + " numerator"]
                denom = row[feat + " denominator"]
            row_dict[feat] = (low, actual, high, numerator, denom)
        CI_dict[fips] = row_dict
    return CI_dict

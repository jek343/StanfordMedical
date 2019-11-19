import os
import pandas as pd

def get_CI_dict(year):
    '''Year is an integer and the super_clean_analytic_data csv for that year
    has already been created'''
    CI_dict = {}
    dataset = pd.read_csv(os.path.join(os.getcwd(),  '..', 'datasets', 'super_clean_analytic_data' + str(year) + '.csv'))
    features = list(dataset)
    features_raw = [i[:-10] for i in features if "raw value" in i]
    features_CI = [i[:-7] for i in features if "CI low" in i]
    # print(*features_CI, sep = "\n")
    features_num = [i[:-10] for i in features if "numerator" in i]
    # print(*features_CI, sep = "\n")

    for index, row in dataset.iterrows():
        row_dict = {}
        fips = row["5-digit FIPS Code"]
        for feat in features_raw:
            actual = row[feat + " raw value"]
            low = None
            high = None
            numerator = None
            denom = None
            if feat in features_CI:
                if feat + " CI low" in features:
                    low = row[feat + " CI low"]
                if feat + " CI high" in features:
                    high = row[feat + " CI high"]
            elif feat in features_num:
                if feat + " numerator" in features:
                    numerator = row[feat + " numerator"]
                if feat + " denominator" in features:
                    denom = row[feat + " denominator"]
            row_dict[feat] = (low, actual, high, numerator, denom)
        CI_dict[fips] = row_dict
    return CI_dict

# dict = get_CI_dict(2019)
# print(dict[1001])

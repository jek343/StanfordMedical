"""
Script to make a dataset of mortalities 2013-2019 from CHR csvs
"""

import csv
import pandas as pd
import os

mortality = 'Premature age-adjusted mortality raw value'
years = [2013, 2014, 2015, 2016, 2017, 2018, 2019]
datasets = []

if __name__ == "__main__":
    for year in years:
        path = os.path.join(os.getcwd(),  '..', '..', 'datasets', 'super_clean_analytic_data' + str(year) + '.csv')
        data = pd.read_csv(path)
        mort = data[[mortality]]
        # data = pd.read_csv(path)
        # print(mortality in data.columns)
        mort.dropna(axis='index', how='any', inplace=True)
        mort.rename(index=dict(zip(data.index, data['5-digit FIPS Code'])), inplace=True)
        mort.rename(columns={'Premature age-adjusted mortality raw value': str(year)}, inplace=True)
        datasets += [mort]

    merged = datasets[0].merge(datasets[1], left_index=True, right_index=True)

    for df in datasets[2:]:
        merged = merged.merge(df, left_index=True, right_index=True)

    # print(merged.columns)
    END_CSV_PATH = os.path.join(os.getcwd(),  '../..', 'datasets', 'mort_data.csv')
    merged.to_csv(path_or_buf=END_CSV_PATH, index=True)

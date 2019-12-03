import csv
import os
from sklearn.metrics import r2_score

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

DATA_PATH2019 = os.path.join(os.getcwd(),  '..', 'datasets', 'super_clean_analytic_data2019.csv')
DATA_PATH2018 = os.path.join(os.getcwd(),  '..', 'datasets', 'super_clean_analytic_data2018.csv')
FIELD_NAMES2019 = read_fields(open_csv(DATA_PATH2019), 0)
FIELD_NAMES2018 = read_fields(open_csv(DATA_PATH2018), 0)

DATA_DICT2019 = get_data_as_dicts(open_csv(DATA_PATH2019), [], FIELD_NAMES2019)
DATA_DICT2018 = get_data_as_dicts(open_csv(DATA_PATH2018), [], FIELD_NAMES2018)

fips_mort2019 = {}
mort2018 = []
mort2019 = []

for county in DATA_DICT2019:
    fips_mort2019.update({county['5-digit FIPS Code']:county["Premature age-adjusted mortality raw value"]})

for county in DATA_DICT2018:
    fips = county['5-digit FIPS Code']
    if fips in fips_mort2019 and county["Premature age-adjusted mortality raw value"] != 'Premature age-adjusted mortality raw value':
        mort2018 += [float(county["Premature age-adjusted mortality raw value"])]
        mort2019 += [float(fips_mort2019[fips])]

# print('mort2018', mort2018)
# print('mort2019', mort2019)

print('r2', r2_score(y_pred=mort2018, y_true=mort2019))

import csv
import os
import peter_csv_ops as peter_csv_ops
import numpy as np

DATA_YEAR = 2019
COUNT = 0

def read_fields(csv, field_row):
    '''Gets all the field names'''
    row_num = 0
    for row in csv:
        if row_num == field_row:
            out = []
            for x in row:
                out.append(x)
            return out
        row_num += 1
    return None

def clean_datasets(source_csv_path, ignore_rows, field_row, output_writer):
    '''Creates a csv map of the clean dictionary to be saved'''
    field_names = read_fields(csv.reader(open(source_csv_path)), field_row)
    #get entire csv in dictionary form except header (first two rows)
    data_dict = peter_csv_ops.get_data_as_dicts(csv.reader(open(source_csv_path)), ignore_rows, field_names)

    def clean_remap_func(row_num, row_dict):
        global COUNT
        '''Returns None if row_num is in ignore_rows or if any of the clean_fields in row_num is empty
            Otherwise, returns a dictionary for the row of each entry in clean_fields'''
        if row_num in ignore_rows:
            return None
        if row_dict["Premature age-adjusted mortality raw value"] == '':
            COUNT += 1
            return None
        return {field: row_dict[field] for field in field_names}

    peter_csv_ops.csv_map(source_csv_path, field_names, clean_remap_func, output_writer)


if __name__ == "__main__":

    SOURCE_CSV_PATH = os.path.join(os.getcwd(),  '../..', 'datasets', 'analytic_data' + str(DATA_YEAR) + '.csv')
    END_CSV_PATH = os.path.join(os.getcwd(),  '../..', 'datasets', 'map_data' + str(DATA_YEAR) + '.csv')
    IGNORE_ROWS = [0,1]
    FIELD_ROW = 0
    OUTPUT_WRITER = open(END_CSV_PATH, "w")
    clean_datasets(SOURCE_CSV_PATH, IGNORE_ROWS, FIELD_ROW, OUTPUT_WRITER)
    print(COUNT)

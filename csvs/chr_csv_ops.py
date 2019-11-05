import csv
import os
import peter_csv_ops as peter_csv_ops
import numpy as np

DATA_YEAR = 2018

def get_clean_fields(data_dict, field_names, min_appearance_percentage):
    field_counts = np.zeros(len(field_names), dtype = np.int)
    for pt in data_dict:
        for i in range(len(field_names)):
            field_name = field_names[i]
            if pt[field_name] != '':
                field_counts[i] += 1
    out = []
    for i in range(len(field_counts)):
        if float(field_counts[i]) / float(len(data_dict)) > min_appearance_percentage:
            out.append(field_names[i])

    return out

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

def clean_datasets(source_csv_path, ignore_rows, field_row, min_appearance_percentage, output_writer):
    field_names = read_fields(csv.reader(open(source_csv_path)), field_row)
    data_dict = peter_csv_ops.get_data_as_dicts(csv.reader(open(source_csv_path)), ignore_rows, field_names)
    clean_fields = get_clean_fields(data_dict, field_names, min_appearance_percentage)
    def clean_remap_func(row_num, row_dict):
        if row_num in ignore_rows:
            return None
        for clean_field in clean_fields:
            #if clean_field not in row_dict:
            if row_dict[clean_field] == '':
                return None

        return {clean_field: row_dict[clean_field] for clean_field in clean_fields}
    peter_csv_ops.csv_map(source_csv_path, field_names, clean_remap_func, output_writer)


if __name__ == "__main__":

    SOURCE_CSV_PATH = os.path.join(os.getcwd(),  '../..', 'datasets', 'analytic_data' + str(DATA_YEAR) + '.csv')
    END_CSV_PATH = os.path.join(os.getcwd(),  '../..', 'datasets', 'super_clean_analytic_data' + str(DATA_YEAR) + '.csv')
    IGNORE_ROWS = [0,1]
    FIELD_ROW = 0
    OUTPUT_WRITER = open(END_CSV_PATH, "w")
    clean_datasets(SOURCE_CSV_PATH, IGNORE_ROWS, FIELD_ROW, 0.95, OUTPUT_WRITER)

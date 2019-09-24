import csv

def __row_to_dict(row, field_order):
    return {field_order[i]: row[i] for i in range(len(field_order))}


def __get_all_remapped_fields(source_csv, field_order, row_remap_func):
    def __fold_func(row, x):
        row_num, remapped_fields = x
        row_dict = __row_to_dict(row, field_order)
        remapped_row_dict = row_remap_func(row_num, row_dict)
        if remapped_row_dict is not None:
            for k in remapped_row_dict:
                if not k in remapped_fields:
                    remapped_fields.append(k)
        return (row_num + 1, remapped_fields)
    return csv_fold(source_csv, __fold_func, (0, []))[1]



def csv_fold(source_csv, f, x0):
    out = x0
    for row in source_csv:
        out = f(row, out)
    return out

def load_csv(source_csv_path):
    source_csv_file = open(PATH)
    return csv.reader(source_csv_file, delimiter=',')

#row remap func returns None if the row shouldn't be included.
#row_remap_func of form row_remap_func(row_num, row_dict)
def csv_map(source_csv_path, field_order, row_remap_func, output_writer):
    remap_fields = __get_all_remapped_fields(load_csv(source_csv_path), field_order, row_remap_func)
    print("remap_fields: ", remap_fields)
    field_str = ",".join(remap_fields)
    field_str = field_str[:len(field_str) - 1] + "\n"
    output_writer.write(field_str)


    row_num = 0
    for row in load_csv(source_csv_path):
        row_dict = __row_to_dict(row, field_order)
        remapped_row_dict = row_remap_func(row_num, row_dict)
        if remapped_row_dict is not None:
            row_str = ""
            for field in remap_fields:
                value = remapped_row_dict[field]
                row_str += value +  ","
            row_str = row_str[:len(row_str) - 1] + "\n"
            output_writer.write(row_str)
        row_num += 1



def save_columns(source_csv_path, column_names, field_row, output_csv_path):
    source_csv_file = open(source_csv_path)
    source_csv = csv.reader(source_csv_file, delimiter=',')
    def row_remap_func(row_num, row_dict):
        if row_num == field_row:
            return None
        out = {}
        for col in column_names:
            if col in row_dict:
                out[col] = row_dict[col]
        return out

    input_field_ordering = 0
    for row in source_csv:
        if input_field_ordering == field_row:
            input_field_ordering = row
            break
        input_field_ordering += 1
    output_writer = open(output_csv_path, "w")
    csv_map(source_csv, input_field_ordering, row_remap_func, output_writer)






if __name__ == "__main__":

    PATH = "C:/Users/peter/OneDrive/Desktop/ML/CDS/stanford_medical/datasets/analytic_data2017.csv"
    SAVE_PATH = "C:/Users/peter/OneDrive/Desktop/ML/CDS/stanford_medical/datasets/analytic_data2017_remapped.csv"
    save_columns(PATH, ["State Abbreviation", "Premature death CI high", "Release Year"], 0, SAVE_PATH)
    '''source_csv_file = open(PATH)
    source_csv = csv.reader(source_csv_file, delimiter=',')
    field_order = None
    for row in source_csv:
        field_order = row
        #print("row: ", row)
        break
    #only includes points from Alabama, and only includes certain fields
    def row_remap_func(row_num, row_dict):
        if row_num < 2:
            return None
        if row_dict["State Abbreviation"] != "AL":
            return None

        return {
            "state": row_dict["State Abbreviation"],
            "Premature death CI high": row_dict["Premature death CI high"]
        }

    output_writer = open(SAVE_PATH, "w")

    csv_map(source_csv, field_order, row_remap_func, output_writer)'''

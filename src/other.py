import os
import random

import pandas as pd


def convert_xlsx_to_csv(filename):
    path = 'dataset'
    read_file = pd.read_excel(os.path.join(path, filename + '.xlsx'))
    read_file.to_csv(os.path.join(path, filename + '.csv'), index=None, header=True)


def drop_columns(df, columns_to_drop: list):
    for column in columns_to_drop:
        df.pop(column)


def load_dataset():
    df_text = pd.DataFrame(pd.read_csv(os.path.join(os.path.dirname(__file__), 'dataset', 'OpArticles.csv')))
    df_adu = pd.DataFrame(pd.read_csv(os.path.join(os.path.dirname(__file__), 'dataset', 'OpArticles_ADUs.csv')))
    create_index_column(df_adu)
    return df_adu, df_text


def create_index_column(df):
    df["id"] = df.index + 1


def write_df_result_to_random_file(df):
    df.to_csv(os.path.join('results', f"{random.randint(0, 90000).__str__()}.csv"))


def write_new_csv_datatest(df, filename):
    df.to_csv(os.path.join('dataset', f"{filename.replace('.csv', '')}.csv"), index=False)


def remove_dataframe_rows_by_id(df_to_remove, list_ids_to_remove):
    df_to_remove.set_index("id", inplace=True)

    df_to_remove.drop(list_ids_to_remove, inplace=True)

    df_to_remove.reset_index(inplace=True)

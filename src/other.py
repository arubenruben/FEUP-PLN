import os
import pandas as pd


def convert_xlsx_to_csv(filename):
    path = 'dataset'
    read_file = pd.read_excel(os.path.join(path, filename + '.xlsx'))
    read_file.to_csv(os.path.join(path, filename + '.csv'), index=None, header=True)


def drop_columns(df, columns_to_drop: list):
    for column in columns_to_drop:
        df.pop(column)


def load_dataset():
    df_text = pd.DataFrame(pd.read_csv(os.path.join('dataset', 'OpArticles.csv')))
    df_adu = pd.DataFrame(pd.read_csv(os.path.join('dataset', 'OpArticles_ADUs.csv')))
    return df_adu, df_text

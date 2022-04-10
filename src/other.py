import os
import random
from joblib import dump, load
import pandas as pd


def convert_xlsx_to_csv(filename):
    path = 'dataset'
    read_file = pd.read_excel(os.path.join(path, filename + '.xlsx'))
    read_file.to_csv(os.path.join(path, filename + '.csv'), index=None, header=True)


def drop_columns(df, columns_to_drop: list):
    df.drop(columns_to_drop, errors='ignore', inplace=True)


def load_dataset(text_augmentation=False):
    df_text = pd.DataFrame(pd.read_csv(os.path.join(os.path.dirname(__file__), 'dataset', 'OpArticles.csv')))

    if not text_augmentation:
        df_adu = pd.DataFrame(pd.read_csv(os.path.join(os.path.dirname(__file__), 'dataset', 'OpArticles_ADUs.csv')))
    else:
        df_adu = pd.DataFrame(
            pd.read_csv(os.path.join(os.path.dirname(__file__), 'dataset', 'OpArticles_ADUs_translator.csv')))

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


def column_to_csv_conditional(df, column_1, column_2, value):
    df_value = df[df[column_2] == value]
    df_value.to_excel(os.path.join(os.path.dirname(__file__), 'dataset', value + '.xlsx'), header=True)
    df_value[column_1].to_excel(os.path.join(os.path.dirname(__file__), 'dataset', column_1 + '.xlsx'), header=True)


def add_xlsx_to_df(df_adu, column_1, value):
    df_value = pd.read_excel(os.path.join(os.path.dirname(__file__), 'dataset', value + '.xlsx'), index_col=0)
    df_column = pd.read_excel(os.path.join(os.path.dirname(__file__), 'dataset', column_1 + '.xlsx'), index_col=0)

    for index, _ in df_value.iterrows():
        df_value.at[index, column_1] = df_column.loc[index].at[column_1]

    df_adu = df_adu.append(df_value, ignore_index=True)
    df_adu.to_csv(os.path.join(os.path.dirname(__file__), 'dataset', 'OpArticles_ADUs_translator.csv'), index=None,
                  header=True)


def save_classifier_to_disk(clf, clf_name):
    dump(clf, os.path.join('classifiers', f"{clf_name}_{random.randint(0, 90000).__str__()}.joblib"))


def load_classifier_from_disk(filename):
    return load(os.path.join('classifiers', f"{filename.replace('.joblib', '')}.joblib"))

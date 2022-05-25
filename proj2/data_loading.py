import os

import pandas as pd
from sklearn.model_selection import train_test_split

ENCODING = {"label": {"Value": 0, "Value(+)": 1, "Value(-)": 2, "Fact": 3, "Policy": 4}}


def load_dataset():
    df_text = pd.DataFrame(pd.read_csv(os.path.join('dataset', 'OpArticles.csv')))

    df_adu = pd.DataFrame(
        pd.read_csv(os.path.join('dataset', 'OpArticles_ADUs.csv')))

    return df_adu, df_text


def normalize_dataset(df):
    df.drop(columns=['article_id', 'annotator', 'node', 'ranges'], inplace=True)
    df.rename(columns={"tokens": "text"}, inplace=True)
    df.replace(ENCODING, inplace=True)


def split_train_test(df):
    train, test = train_test_split(df, test_size=0.2)

    return train, test

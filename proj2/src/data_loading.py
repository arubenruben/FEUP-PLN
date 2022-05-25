import os

import pandas as pd
from datasets import Dataset
from datasets import DatasetDict

ENCODING = {"label": {"Value": 0, "Value(+)": 1, "Value(-)": 2, "Fact": 3, "Policy": 4}}


def load_dataset():
    df_text = pd.DataFrame(pd.read_csv(os.path.join('dataset', 'OpArticles.csv')))

    df_adu = pd.DataFrame(
        pd.read_csv(os.path.join('dataset', 'OpArticles_ADUs.csv')))

    return df_adu, df_text


def normalize_dataset(df):
    df.drop(columns=['article_id', 'annotator', 'node', 'ranges'], inplace=True)
    df.replace(ENCODING, inplace=True)

    dataset_hf = Dataset.from_pandas(df)

    return dataset_hf


def split_train_test(df, test_percentage=0.2, validation_percentage=0.5):
    dataset = normalize_dataset(df)

    if test_percentage == 1.0:
        return DatasetDict({
            'test': dataset
        })

    train_test = dataset.train_test_split(test_size=test_percentage)

    # Split the 10% test+validation set in half test, half validation
    valid_test = train_test['test'].train_test_split(test_size=(1.0 - validation_percentage))

    train_valid_test_dataset = DatasetDict({
        'train': train_test['train'],
        'validation': valid_test['train'],
        'test': valid_test['test']
    })

    return train_valid_test_dataset


def load_data_for_masking(df):
    return list(df['body'])
    """
    df.drop(columns=['article_id', 'title', 'authors', 'meta_description', 'topics', 'keywords', 'publish_date',
                     'url_canonical'], inplace=True)

    df.rename(columns={'body': 'tokens'}, inplace=True)

    dataset_hf = Dataset.from_pandas(df)

    return DatasetDict({
        'unsupervised': dataset_hf,
    })
    """

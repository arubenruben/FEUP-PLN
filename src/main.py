import os
import pandas as pd
import re


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report


def load_dataset():
    df_text = pd.DataFrame(pd.read_csv('.\dataset\OpArticles.csv'))
    df_adu = pd.DataFrame(pd.read_csv('.\dataset\OpArticles_ADUs.csv'))
    return df_adu, df_text


"""
    ADUS: node ids,annotator    
"""


def drop_columns(df, columns_to_drop: list):
    for column in columns_to_drop:
        df.pop(column)


def join_datasets(df_adu, df_text):
    """
        Join By Some Criteria
    """
    return df_adu


def corpus_extraction(df):

    corpus = []

    for row in df['tokens']:
        corpus.append(row)

    return corpus


def label_extraction(df):
    labels = []

    for row in df['label']:
        labels.append(row)

    return labels


def normalize_corpus(corpus):

    corpus_aux = []

    for i in range(0, len(corpus)):
        tokens = re.sub('[^a-zA-Z]', ' ', corpus[i])
        tokens = tokens.lower()

        corpus_aux.append()

    corpus = corpus_aux

    return corpus


def vectorize_bag_of_words(corpus, labels, max_features=1500):

    vectorizer = CountVectorizer(max_features=max_features)

    X = vectorizer.fit_transform(corpus).toarray()
    y = labels

    return X, y


def split_train_test(X, y, test_size=0.20):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=2, stratify=y)

    return X_train, X_test, y_train, y_test


def apply_naive_bayes(X_train, y_train, X_test):

    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    return y_pred


def evaluate_results(y_pred, y_test):
    print(classification_report(y_test, y_pred))
    #print(confusion_matrix(y_test, y_pred))


def main():
    df_adu, df_text = load_dataset()

    drop_columns(df_adu, ['article_id', 'node', 'annotator'])
    drop_columns(df_text, ['article_id', 'title', 'authors', 'meta_description',
                 'topics', 'keywords', 'publish_date', 'url_canonical'])

    corpus = corpus_extraction(df_adu)
    labels = label_extraction(df_adu)

    X, y = vectorize_bag_of_words(corpus, labels)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    y_pred = apply_naive_bayes(X_train=X_train, y_train=y_train, X_test=X_test)

    evaluate_results(y_pred=y_pred, y_test=y_test)

    #print(df_adu.head())
    #print(df_text.head())


if __name__ == "__main__":
    main()

# import spacy
import random

import nltk
import numpy as np
import pandas as pd
import scipy
import scipy as sp
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from scipy import sparse
from sklearn import preprocessing
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB, CategoricalNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.evaluation import evaluate_results
from src.exploratory_analyses import size_vocabulary, outlier_detection, deal_with_outliers
from src.other import drop_columns, write_df_to_file, load_dataset
from src.vectorizers import vectorize_bag_of_words, vectorize_tf_idf, vectorize_1_hot


def clf_factory(algorithm, *params):
    if algorithm == 'naive_bayes':
        return GaussianNB(*params)

    if algorithm == 'knn':
        return KNeighborsClassifier(*params)

    if algorithm == 'logistic_regression':
        return LogisticRegression(max_iter=1000, *params)

    if algorithm == 'svm':
        return SVC(*params)

    if algorithm == 'decision_tree':
        return DecisionTreeClassifier(*params)

    if algorithm == 'bagging':
        return BaggingClassifier(*params)

    if algorithm == 'random_forest':
        return RandomForestClassifier(*params)

    if algorithm == 'neural_net':
        return MLPClassifier(*params)

    raise "Invalid Algorithm"


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

    return np.array(labels)


def normalize_corpus(corpus):
    corpus_aux = []

    ''' Lemmatizer
    nlp = spacy.load("pt_core_news_sm")
    
    for doc in nlp.pipe(corpus, batch_size=32, n_process=3, disable=["parser", "ner"]):
        lemmas = []
        for token in doc:
            lemmas.append(token.lemma_)

        row = ' '.join([w for w in lemmas if w not in set(stopwords.words('portuguese'))])
    '''

    stemmer = SnowballStemmer('portuguese')

    for row in corpus:
        word_list = nltk.word_tokenize(row)
        row = ' '.join([stemmer.stem(w) for w in word_list if not w in set(stopwords.words('portuguese'))])
        corpus_aux.append(row)

        corpus_aux.append(row)

    return corpus_aux


def split_train_test(X, y, test_size=0.20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    return X_train, X_test, y_train, y_test


def apply_clf(clf, X_train, y_train, X_test):
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    return y_pred


def label_encoding(y):
    encoder = preprocessing.LabelEncoder()
    y = encoder.fit_transform([1, 2, 2, 6])
    return y, encoder


def baseline(df_adu, df_text, algorithm='naive_bayes'):
    drop_columns(df_adu, ['article_id', 'node', 'annotator'])
    drop_columns(df_text, ['article_id', 'title', 'authors', 'meta_description',
                           'topics', 'keywords', 'publish_date', 'url_canonical'])

    corpus = corpus_extraction(df_adu)

    y = label_extraction(df_adu)

    X, vec, vectorizer = vectorize_bag_of_words(corpus)

    size_vocabulary(vectorizer)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    clf = clf_factory(algorithm)

    y_pred = apply_clf(clf, X_train=X_train, y_train=y_train, X_test=X_test)

    evaluate_results(y_pred=y_pred, y_test=y_test)


def test_tf_idf(df_adu, df_text, algorithm='naive_bayes'):
    drop_columns(df_adu, ['article_id', 'node', 'annotator'])

    corpus = corpus_extraction(df_adu)

    y = label_extraction(df_adu)

    X, vec, vectorizer = vectorize_tf_idf(corpus)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    clf = clf_factory(algorithm)

    y_pred = apply_clf(clf, X_train=X_train, y_train=y_train, X_test=X_test)
    print('Policy' in y_pred)

    evaluate_results(y_pred=y_pred, y_test=y_test)


def test_1_hot_vector(df_adu, df_text, algorithm='naive_bayes'):
    drop_columns(df_adu, ['article_id', 'node', 'annotator'])
    drop_columns(df_text, ['article_id', 'title', 'authors', 'meta_description',
                           'topics', 'keywords', 'publish_date', 'url_canonical'])

    corpus = corpus_extraction(df_adu)

    y = label_extraction(df_adu)

    X, vec, vectorizer = vectorize_1_hot(corpus)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    clf = clf_factory(algorithm)

    y_pred = apply_clf(clf, X_train=X_train, y_train=y_train, X_test=X_test)

    evaluate_results(y_pred=y_pred, y_test=y_test)


def test_different_features_sizes(df_adu, df_text, algorithm='knn'):
    """
    The First Iteration Must Collect the vocabulary size
    """

    drop_columns(df_adu, ['article_id', 'node', 'annotator'])
    drop_columns(df_text, ['article_id', 'title', 'authors', 'meta_description',
                           'topics', 'keywords', 'publish_date', 'url_canonical'])

    corpus_base = normalize_corpus(corpus_extraction(df_adu))

    y = label_extraction(df_adu)

    for i in range(10):
        if i == 0:
            vec_len = 0

        corpus = corpus_base.copy()

        if i == 0:
            X, vec, vectorizer = vectorize_tf_idf(corpus)
            vec_len = size_vocabulary(vectorizer)
        else:
            rand_int = random.randint(1, vec_len)
            X, vec, vectorizer = vectorize_tf_idf(corpus, rand_int)

            X_train, X_test, y_train, y_test = split_train_test(X, y)

            clf = clf_factory(algorithm)

            y_pred = apply_clf(clf, X_train=X_train, y_train=y_train, X_test=X_test)

            print(f"Results for {rand_int}")

            evaluate_results(y_pred=y_pred, y_test=y_test)


def baseline_with_normalization(df_adu, df_text, algorithm='decision_tree'):
    drop_columns(df_adu, ['article_id', 'node', 'annotator'])
    drop_columns(df_text, ['article_id', 'title', 'authors', 'meta_description',
                           'topics', 'keywords', 'publish_date', 'url_canonical'])

    corpus = normalize_corpus(corpus_extraction(df_adu))

    y = label_extraction(df_adu)

    X, vec, vectorizer = vectorize_bag_of_words(corpus)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    clf = clf_factory(algorithm)
    y_pred = apply_clf(clf, X_train=X_train, y_train=y_train, X_test=X_test)

    evaluate_results(y_pred=y_pred, y_test=y_test)


def oversample_with_smote(X_train, y_train, sampling_strategy="auto"):
    over_sampler = SMOTE(sampling_strategy=sampling_strategy)

    X_train, y_train = over_sampler.fit_resample(X_train, y_train)

    return X_train, y_train


def baseline_2(df_adu):
    drop_columns(df_adu, ['article_id', 'node', 'annotator'])

    corpus = corpus_extraction(df_adu)

    X, vec, vectorizer = vectorize_bag_of_words(corpus)

    y = label_extraction(df_adu)

    X_train, X_test, y_train, y_test = split_train_test(X.toarray(), y, 0.20)

    X_train, y_train = oversample_with_smote(X_train, y_train)

    # pd_df = pd.DataFrame(y_train, columns=['label'])

    # class_distribution(pd_df)

    clf = clf_factory('decision_tree')

    y_pred = apply_clf(clf, X_train=X_train, y_train=y_train, X_test=X_test)

    evaluate_results(y_pred=y_pred, y_test=y_test)

    """
    """


def baseline_deleting_outliers(df_adu, outlier_strategy='delete'):
    dict_collisions = outlier_detection(df_adu)

    deal_with_outliers(df_adu, dict_collisions, outlier_strategy)

    drop_columns(df_adu, ['article_id', 'node', 'annotator'])

    corpus = corpus_extraction(df_adu)

    X, vec, vectorizer = vectorize_bag_of_words(corpus)

    y = label_extraction(df_adu)

    X_train, X_test, y_train, y_test = split_train_test(X.toarray(), y, 0.20)

    # X_train, y_train = oversample_with_smote(X_train, y_train)

    # pd_df = pd.DataFrame(y_train, columns=['label'])

    # class_distribution(pd_df)

    clf = clf_factory('naive_bayes')

    y_pred = apply_clf(clf, X_train=X_train, y_train=y_train, X_test=X_test)

    evaluate_results(y_pred=y_pred, y_test=y_test)


def model_annotator_explicit(df_adu):
    drop_columns(df_adu, ['article_id', 'node'])

    corpus = corpus_extraction(df_adu)

    X, vec, vectorizer = vectorize_bag_of_words(corpus)

    vocab = vec.get_feature_names_out()

    X = pd.DataFrame(X.toarray(), columns=vocab)

    """
    Scipy Deals Awfully with string. One Hot Encodings are always required
    """
    ord_enc = OrdinalEncoder()

    X["annotator"] = ord_enc.fit_transform(df_adu[["annotator"]])

    # print(X.head())

    X = sparse.csr_matrix(X.values)

    y = label_extraction(df_adu)

    X_train, X_test, y_train, y_test = split_train_test(X.toarray(), y, 0.20)

    clf = clf_factory('naive_bayes')

    y_pred = apply_clf(clf, X_train=X_train, y_train=y_train, X_test=X_test)

    evaluate_results(y_pred=y_pred, y_test=y_test)


def model_for_each_annotator():
    for _ in range(3):
        df_adu, df_text = load_dataset()
        df_adu = df_adu[df_adu['annotator'] == 'A']
        print("Annotator A")
        baseline(df_adu, df_text)
        print("-------")
        df_adu, df_text = load_dataset()
        df_adu = df_adu[df_adu['annotator'] == 'B']
        print("Annotator B")
        baseline(df_adu, df_text)
        print("-------")
        df_adu, df_text = load_dataset()
        df_adu = df_adu[df_adu['annotator'] == 'C']
        print("Annotator C")
        baseline(df_adu, df_text)
        print("-------")
        df_adu, df_text = load_dataset()

        df_adu = df_adu[df_adu['annotator'] == 'D']
        print("Annotator D")
        baseline(df_adu, df_text)
        print("-------")

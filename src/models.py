import random

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.evaluation import evaluate_results
from src.exploratory_analyses import size_vocabulary
from src.other import drop_columns
from src.vectorizers import vectorize_bag_of_words, vectorize_tf_idf, vectorize_1_hot


def clf_factory(algorithm, *params):
    if algorithm == 'naive_bayes':
        return MultinomialNB(*params)

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

    return labels


def normalize_corpus(corpus):
    corpus_aux = []

    stemmer = SnowballStemmer('portuguese')

    for i in range(0, len(corpus)):
        row = corpus[i].lower()
        row = ' '.join([stemmer.stem(w) for w in row.split() if not w in set(stopwords.words('portuguese'))])

        corpus_aux.append(row)

    return corpus_aux


def split_train_test(X, y, test_size=0.20):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=2, stratify=y)

    return X_train, X_test, y_train, y_test


def apply_clf(clf, X_train, y_train, X_test):
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    return y_pred


def baseline(df_adu, df_text, algorithm='decision_tree'):
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


def test_tf_idf(df_adu, df_text, algorithm='decision_tree'):
    drop_columns(df_adu, ['article_id', 'node', 'annotator'])
    drop_columns(df_text, ['article_id', 'title', 'authors', 'meta_description',
                           'topics', 'keywords', 'publish_date', 'url_canonical'])

    corpus = corpus_extraction(df_adu)

    y = label_extraction(df_adu)

    X, vec, vectorizer = vectorize_tf_idf(corpus)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    clf = clf_factory(algorithm)

    y_pred = apply_clf(clf, X_train=X_train, y_train=y_train, X_test=X_test)

    evaluate_results(y_pred=y_pred, y_test=y_test)


def test_1_hot_vector(df_adu, df_text, algorithm='decision_tree'):
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

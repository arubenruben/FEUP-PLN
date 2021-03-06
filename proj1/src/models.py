import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy import sparse, stats
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, Lasso
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

from evaluation import evaluate_results
from exploratory_analyses import size_vocabulary, outlier_detection, deal_with_outliers
from lexicons import load_lexicons, get_polarity
from other import drop_columns, load_dataset, load_embedding_model
from pos import get_pos_numbers
from text_processing import insert_previous_and_after_sentence_to_adu, tokenization, normalize_corpus, normalize_corpus_embedding
from vectorizers import vectorize_bag_of_words, vectorize_tf_idf, vectorize_1_hot
from embeddings import train_embedding_model, apply_embedding_model


def clf_factory(algorithm, *params):
    if algorithm == 'naive_bayes':
        return GaussianNB()

    if algorithm == 'knn':
        return KNeighborsClassifier(*params)

    if algorithm == 'logistic_regression':
        return LogisticRegression(max_iter=1000, *params)
    if algorithm == 'svm':
        return SGDClassifier(loss='modified_huber', max_iter=30000, n_iter_no_change=20, random_state=42,
                             class_weight='balanced',
                             n_jobs=-1)

    if algorithm == 'decision_tree':
        return DecisionTreeClassifier(max_depth=15, random_state=42)

    if algorithm == 'bagging':
        return BaggingClassifier(*params, random_state=42)

    if algorithm == 'random_forest':
        return RandomForestClassifier(n_estimators=1000, class_weight='balanced', random_state=1, n_jobs=-1)

    if algorithm == 'neural_net':
        return MLPClassifier(*params, learning_rate='adaptive', random_state=42)

    if algorithm == 'LASSO':
        return Lasso(alpha=0.1)

    raise "Invalid Algorithm"


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


def split_train_test(X, y, test_size=0.20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=test_size)

    return X_train, X_test, y_train, y_test


def apply_clf(clf, X_train, y_train, X_test):
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    return y_pred


def apply_cross_validation(clf, X, y):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=7, random_state=1)

    scores = cross_validate(clf, X, y, cv=cv, n_jobs=-1, scoring=['roc_auc_ovo_weighted', 'f1_macro'])

    return scores


def baseline(df_adu, df_text, algorithm='naive_bayes'):
    drop_columns(df_adu, ['article_id', 'node', 'annotator'])
    drop_columns(df_text, ['article_id', 'title', 'authors', 'meta_description',
                           'topics', 'keywords', 'publish_date', 'url_canonical'])

    corpus = corpus_extraction(df_adu)

    y = label_extraction(df_adu)

    # ngram_range=(1,3) ---> uni-grams, bi-grams and trigrams
    X, vec, vectorizer = vectorize_bag_of_words(corpus, ngram_range=(1, 3), max_features=20000)

    size_vocabulary(vectorizer)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    clf = clf_factory(algorithm)

    y_pred = apply_clf(clf, X_train=X_train, y_train=y_train, X_test=X_test)

    evaluate_results(y_pred=y_pred, y_test=y_test, clf=clf, X_test=X_test)


def test_tf_idf(df_adu, algorithm='naive_bayes'):
    drop_columns(df_adu, ['article_id', 'node', 'annotator'])

    corpus = corpus_extraction(df_adu)

    y = label_extraction(df_adu)

    X, vec, vectorizer = vectorize_tf_idf(corpus)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    clf = clf_factory(algorithm)

    y_pred = apply_clf(clf, X_train=X_train, y_train=y_train, X_test=X_test)
    print('Policy' in y_pred)

    evaluate_results(y_pred=y_pred, y_test=y_test, clf=clf, X_test=X_test)


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

    evaluate_results(y_pred=y_pred, y_test=y_test, clf=clf, X_test=X_test)


def baseline_with_normalization(df_adu, df_text, algorithm='naive_bayes'):
    drop_columns(df_adu, ['article_id', 'node', 'annotator'])
    drop_columns(df_text, ['article_id', 'title', 'authors', 'meta_description',
                           'topics', 'keywords', 'publish_date', 'url_canonical'])

    corpus = normalize_corpus(corpus_extraction(df_adu))
    print("Normalization Done")
    y = label_extraction(df_adu)

    X, vec, vectorizer = vectorize_bag_of_words(corpus, max_features=20000)
    print("Vectorization done")
    size_vocabulary(vectorizer)

    X_train, X_test, y_train, y_test = split_train_test(X, y)
    print("Train Test Split done")
    clf = clf_factory(algorithm)

    y_pred = apply_clf(clf, X_train=X_train, y_train=y_train, X_test=X_test)
    print("Apply Model Done")
    evaluate_results(y_pred=y_pred, y_test=y_test, clf=clf, X_test=X_test)


def oversample_with_smote(X_train, y_train, sampling_strategy="auto"):
    strategy = {'Value(+)': 800, 'Value(-)': 800, 'Policy': 800, 'Fact': 800}
    over_sampler = SMOTE(sampling_strategy=sampling_strategy, n_jobs=-1, random_state=1)

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

    evaluate_results(y_pred=y_pred, y_test=y_test, clf=clf, X_test=X_test)

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

    evaluate_results(y_pred=y_pred, y_test=y_test, clf=clf, X_test=X_test)


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

    evaluate_results(y_pred=y_pred, y_test=y_test, clf=clf, X_test=X_test)


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


"""
    Lots of Duplicates
"""


def baseline_with_all_paragraph(df_adu, df_text):
    insert_previous_and_after_sentence_to_adu(df_adu, df_text)
    # print(df_adu.head())
    # print(df_text.head())


def baseline_with_lexicons(df_adu):
    load_lexicons()

    drop_columns(df_adu, ['article_id', 'node'])

    df_adu['positive_words'] = 0
    df_adu['neutral_words'] = 0
    df_adu['negative_words'] = 0
    df_adu['unknown_words'] = 0

    for i, row in df_adu.iterrows():
        positives = 0
        neutrals = 0
        negatives = 0
        unknowns = 0

        list_tokens = tokenization(row['tokens'])

        for token in list_tokens:

            polarity = get_polarity(token)

            if polarity == -1:
                negatives += 1
            elif polarity == 0:
                neutrals += 1
            elif polarity == 1:
                positives += 1
            elif polarity == 2:
                unknowns += 1

        df_adu.at[i, 'positive_words'] = positives
        df_adu.at[i, 'neutral_words'] = neutrals
        df_adu.at[i, 'negative_words'] = negatives
        df_adu.at[i, 'unknown_words'] = unknowns

    # write_new_csv_datatest(df_adu, 'dataset_with_lexicons')

    corpus = corpus_extraction(df_adu)

    X, vec, vectorizer = vectorize_bag_of_words(corpus)

    vocab = vec.get_feature_names_out()

    X = pd.DataFrame(X, columns=vocab)

    X['positive_words'] = df_adu['positive_words']
    X['neutral_words'] = df_adu['neutral_words']
    X['negative_words'] = df_adu['negative_words']
    X['unknown_words'] = df_adu['unknown_words']

    """
    Scipy Style Translation
    """

    X = sparse.csr_matrix(X.values)

    y = label_extraction(df_adu)

    X_train, X_test, y_train, y_test = split_train_test(X.toarray(), y, 0.20)

    clf = clf_factory('naive_bayes')

    y_pred = apply_clf(clf, X_train=X_train, y_train=y_train, X_test=X_test)

    evaluate_results(y_pred=y_pred, y_test=y_test, clf=clf, X_test=X_test)


def baseline_with_pos(df_adu):
    drop_columns(df_adu, ['article_id', 'node'])

    df_adu['number_adj'] = 0
    df_adu['number_interjections'] = 0
    df_adu['number_verbs'] = 0
    df_adu['number_proper_nouns'] = 0

    for i, row in df_adu.iterrows():
        number_adj, number_interjections, number_verbs, number_proper_nouns = get_pos_numbers(row['tokens'])
        df_adu.at[i, 'number_adj'] = number_adj
        df_adu.at[i, 'number_interjections'] = number_interjections
        df_adu.at[i, 'number_verbs'] = number_verbs
        df_adu.at[i, 'number_proper_nouns'] = number_proper_nouns

    corpus = corpus_extraction(df_adu)

    X, vec, vectorizer = vectorize_bag_of_words(corpus)

    vocab = vec.get_feature_names_out()

    X = pd.DataFrame(X, columns=vocab)

    X['number_adj'] = df_adu['number_adj']
    X['number_interjections'] = df_adu['number_interjections']
    X['number_verbs'] = df_adu['number_verbs']
    X['number_proper_nouns'] = df_adu['number_proper_nouns']

    """
    Scipy Style Translation
    """

    # write_new_csv_datatest(X, 'dataset_with_lexicons')

    X = sparse.csr_matrix(X.values)

    y = label_extraction(df_adu)

    X_train, X_test, y_train, y_test = split_train_test(X.toarray(), y, 0.20)

    clf = clf_factory('naive_bayes')

    y_pred = apply_clf(clf, X_train=X_train, y_train=y_train, X_test=X_test)

    evaluate_results(y_pred=y_pred, y_test=y_test, clf=clf, X_test=X_test)


def baseline_with_embeddings(df_adu, load_model=False, algorithm='naive_bayes'):
    
    corpus = corpus_extraction(df_adu)
    corpus = normalize_corpus_embedding(corpus)

    y = label_extraction(df_adu)

    if load_model:
        model = load_embedding_model("skip_s100.txt")
    else:
        model = train_embedding_model(corpus)

    #lens = [len(c.split()) for c in corpus]
    #print(np.min(lens), np.max(lens), np.mean(lens), np.std(lens), stats.mode(lens))
    X = apply_embedding_model(model, corpus)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    clf = clf_factory(algorithm)

    y_pred = apply_clf(clf, X_train=X_train, y_train=y_train, X_test=X_test)

    evaluate_results(y_pred=y_pred, y_test=y_test, clf=clf, X_test=X_test)

def densify_matrix(model_name, X):
    if model_name == 'naive_bayes':
        return X.toarray()

    return X

import random

from evaluation import evaluate_results
from exploratory_analyses import size_vocabulary
from models import split_train_test, clf_factory, apply_clf, corpus_extraction, label_extraction
from other import drop_columns
from text_processing import normalize_corpus
from vectorizers import vectorize_tf_idf


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

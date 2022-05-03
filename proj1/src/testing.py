import random

from evaluation import evaluate_results
from exploratory_analyses import size_vocabulary
from models import split_train_test, clf_factory, apply_clf, corpus_extraction, label_extraction
from other import drop_columns, load_dataset, load_classifier_from_disk
from text_processing import normalize_corpus
from vectorizers import vectorize_tf_idf, vectorize_bag_of_words


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

            evaluate_results(y_pred=y_pred, y_test=y_test, clf=clf, X_test=X_test)


def load_classifier():
    df_adu, _ = load_dataset(text_augmentation=True)

    drop_columns(df_adu, ['article_id', 'node', 'annotator'])

    print("Start Normalizing")
    corpus = normalize_corpus(corpus_extraction(df_adu))

    print("Start Vectorize")
    X, vec, vectorizer = vectorize_bag_of_words(corpus, ngram_range=(1, 3), max_features=1000)

    y = label_extraction(df_adu)

    X_train, X_test, y_train, y_test = split_train_test(X, y, 0.20)
    """
    print("Start SMOTE")
    X_train, y_train = oversample_with_smote(X_train, y_train)
    """

    clf = load_classifier_from_disk('naive_bayes_11764')

    print("Start Training")

    y_pred = clf.predict(X_test)

    print("Start Evaluating")

    evaluate_results(y_pred=y_pred, y_test=y_test, clf=clf, X_test=X_test)

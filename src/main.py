from models import *
from other import load_dataset, save_classifier_to_disk
from testing import load_classifier


def main():
    # dev()
    prod()
    # load_classifier()


def dev():
    df_adu, df_text = load_dataset()
    # baseline_with_lexicons(df_adu)
    # baseline_with_pos(df_adu)
    baseline_with_normalization(df_adu, df_text)
    # baseline(df_adu, df_text, algorithm="decision_tree")
    # baseline(df_adu, df_text)

    # test_1_hot_vector(df_adu, df_text)
    # test_tf_idf(df_adu, df_text)
    # print("---------------")


def prod():
    print("Loading Dataset")

    df_adu, _ = load_dataset(text_augmentation=True)

    print("Start Removing Disagreements")

    dict_collisions = outlier_detection(df_adu)

    deal_with_outliers(df_adu, dict_collisions, 'delete')

    drop_columns(df_adu, ['article_id', 'node', 'annotator'])

    print("Start Normalizing")
    corpus = normalize_corpus(corpus_extraction(df_adu))

    print("Start Vectorize")
    X, vec, vectorizer = vectorize_tf_idf(corpus, ngram_range=(1, 3), max_features=25000)

    y = label_extraction(df_adu)

    X_train, X_test, y_train, y_test = split_train_test(X, y, 0.20)
    """
    print("Start SMOTE")
    X_train, y_train = oversample_with_smote(X_train, y_train)
    """
    clf_name = 'naive_bayes'
    clf = clf_factory(clf_name)

    print("Start Training")
    y_pred = apply_clf(clf, X_train=X_train, y_train=y_train, X_test=X_test)

    print("Start Evaluating")

    evaluate_results(y_pred=y_pred, y_test=y_test)

    print("Start Saving CLF to disk")
    save_classifier_to_disk(clf, clf_name)


if __name__ == "__main__":
    main()

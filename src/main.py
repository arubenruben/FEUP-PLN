from models import *
from other import load_dataset


def main():
    dev()
    # prod()


def dev():
    df_adu, df_text = load_dataset()
    # baseline_with_lexicons(df_adu)
    # baseline_with_pos(df_adu)
    # baseline_with_normalization(df_adu, df_text)
    baseline(df_adu, df_text)
    # baseline(df_adu, df_text)

    # test_1_hot_vector(df_adu, df_text)
    # test_tf_idf(df_adu, df_text)
    # print("---------------")


def prod():
    df_adu, _ = load_dataset(text_augmentation=True)

    drop_columns(df_adu, ['article_id', 'node', 'annotator'])

    corpus = normalize_corpus(corpus_extraction(df_adu))

    X, vec, vectorizer = vectorize_bag_of_words(corpus)

    y = label_extraction(df_adu)

    X_train, X_test, y_train, y_test = split_train_test(X, y, 0.20)

    X_train, y_train = oversample_with_smote(X_train, y_train)

    clf = clf_factory('random_forest')

    y_pred = apply_clf(clf, X_train=X_train, y_train=y_train, X_test=X_test)

    evaluate_results(y_pred=y_pred, y_test=y_test)


if __name__ == "__main__":
    main()

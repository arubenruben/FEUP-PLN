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

    print("Start POS Tagging")

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

    print("Start Lexicons")

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
        # df_adu.at[i, 'unknown_words'] = unknowns

    print("Start Vectorize")
    X, vec, vectorizer = vectorize_tf_idf(corpus, ngram_range=(1, 3), max_features=25000)

    vocab = vec.get_feature_names_out()

    X = pd.DataFrame(X, columns=vocab)

    X['number_adj'] = df_adu['number_adj']
    X['number_interjections'] = df_adu['number_interjections']
    X['number_verbs'] = df_adu['number_verbs']
    X['number_proper_nouns'] = df_adu['number_proper_nouns']

    X['positive_words'] = df_adu['positive_words']
    X['neutral_words'] = df_adu['neutral_words']
    X['negative_words'] = df_adu['negative_words']
    # X['unknown_words'] = df_adu['unknown_words']

    X = sparse.csr_matrix(X.values)

    y = label_extraction(df_adu)

    X_train, X_test, y_train, y_test = split_train_test(X, y, 0.20)

    # print("Start SMOTE")
    # X_train, y_train = oversample_with_smote(X_train, y_train)

    clf_name = 'svm'
    clf = clf_factory(clf_name)

    print("Start Training")
    y_pred = apply_clf(clf, X_train=X_train, y_train=y_train, X_test=X_test)

    print("Start Evaluating")

    evaluate_results(y_pred=y_pred, y_test=y_test, clf=clf, X_test=X_test)

    print("Start Saving CLF to disk")
    save_classifier_to_disk(clf, clf_name)


if __name__ == "__main__":
    main()

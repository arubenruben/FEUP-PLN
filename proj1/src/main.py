from models import *
from other import load_dataset, save_classifier_to_disk


def main():
    #dev()
    prod()


def dev():
    df_adu, df_text = load_dataset()
    # baseline_with_lexicons(df_adu)
    # baseline_with_pos(df_adu)
    # baseline_with_normalization(df_adu, df_text)
    # baseline(df_adu, df_text)
    # baseline(df_adu, df_text)

    # test_1_hot_vector(df_adu, df_text)
    # test_tf_idf(df_adu, df_text)
    # print("---------------")
    baseline_with_embeddings(df_adu, load_model=True)


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

    print("Start Vectorize")

    X, vec, vectorizer = vectorize_tf_idf(corpus, ngram_range=(1, 2), max_features=None)

    vocab = vec.get_feature_names_out()
    print(f"Size of Vocabulary:{len(vocab)}")

    X = pd.DataFrame(X, columns=vocab)

    X['number_adj'] = df_adu['number_adj']
    X['number_interjections'] = df_adu['number_interjections']
    X['number_verbs'] = df_adu['number_verbs']

    X = sparse.csr_matrix(X.values)

    y = label_extraction(df_adu)

    clf_name = 'svm'

    """
    Some Models require dense matrix
    """
    X = densify_matrix(clf_name, X)

    clf = clf_factory(clf_name)

    print("Start Cross Validation")

    scores = apply_cross_validation(clf, X, y)

    print(scores)

    print("Start Saving CLF to disk")

    save_classifier_to_disk(clf, clf_name)


if __name__ == "__main__":
    main()

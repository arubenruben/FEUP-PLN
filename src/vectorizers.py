from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def vectorize_bag_of_words(corpus, max_features=None, binary=False):
    vectorizer = CountVectorizer(max_features=max_features)

    vec = vectorizer.fit(corpus)

    X = vectorizer.transform(corpus)

    return X, vec, vectorizer


def vectorize_tf_idf(corpus, max_features=None):
    vectorizer = TfidfVectorizer(max_features=max_features)

    vec = vectorizer.fit(corpus)

    X = vectorizer.transform(corpus).toarray()

    return X, vec, vectorizer


def vectorize_1_hot(corpus, max_features=None):
    return vectorize_bag_of_words(corpus, max_features, True)

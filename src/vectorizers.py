import nltk
from nltk import bigrams
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from text_processing import tokenization


def vectorize_bag_of_words(corpus, ngram_range=(1,1), max_features=None, binary=False, lowercase=False):
    vectorizer = CountVectorizer(tokenizer=tokenization, token_pattern=None, ngram_range=ngram_range, max_features=max_features, binary=binary,
                                 lowercase=lowercase)

    vec = vectorizer.fit(corpus)

    X = vectorizer.transform(corpus).todense()

    return X, vec, vectorizer


def vectorize_tf_idf(corpus, max_features=None, lowercase=False):
    vectorizer = TfidfVectorizer(tokenizer=tokenization, token_pattern=None, max_features=max_features,
                                 lowercase=False,
                                 max_df=0.5,
                                 min_df=3,
                                 sublinear_tf=True)

    vec = vectorizer.fit(corpus)

    # print(f"Vocabulary:{vec.stop_words_}")

    X = vectorizer.transform(corpus).todense()

    return X, vec, vectorizer


def vectorize_1_hot(corpus, max_features=None, lowercase=False):
    return vectorize_bag_of_words(corpus, max_features, True, lowercase)

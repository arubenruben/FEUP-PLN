from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import gensim

from text_processing import tokenization, corpus_to_list


def vectorize_bag_of_words(corpus, ngram_range=(1, 1), max_features=None, binary=False, lowercase=False):
    vectorizer = CountVectorizer(tokenizer=tokenization, token_pattern=None, ngram_range=ngram_range,
                                 max_features=max_features, binary=binary,
                                 lowercase=lowercase)

    vec = vectorizer.fit(corpus)

    X = vectorizer.transform(corpus).todense()

    return X, vec, vectorizer


def vectorize_tf_idf(corpus, ngram_range=(1, 1), max_features=None, lowercase=False):
    vectorizer = TfidfVectorizer(tokenizer=tokenization, token_pattern=None, max_features=max_features,
                                 ngram_range=ngram_range,
                                 max_df=0.10,
                                 min_df=2,
                                 lowercase=lowercase,
                                 sublinear_tf=True)

    vec = vectorizer.fit(corpus)

    X = vectorizer.transform(corpus).todense()

    return X, vec, vectorizer



def vectorize_1_hot(corpus, ngram_range=(1, 1), max_features=None, lowercase=False):
    return vectorize_bag_of_words(corpus, ngram_range=ngram_range, max_features=max_features, binary=True,
                                  lowercase=lowercase)

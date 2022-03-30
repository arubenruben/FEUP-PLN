#import spacy
import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from evaluation import evaluate_results
from other import drop_columns


def clf_factory(algorithm, *params):
    if algorithm == 'naive_bayes':
        return MultinomialNB(*params)

    if algorithm == 'knn':
        return KNeighborsClassifier(*params)

    if algorithm == 'logistic_regression':
        return LogisticRegression(max_iter=1000, *params)

    if algorithm == 'svm':
        return SVC(*params)

    if algorithm == 'decision_tree':
        return DecisionTreeClassifier(*params)

    if algorithm == 'bagging':
        return BaggingClassifier(*params)

    if algorithm == 'random_forest':
        return RandomForestClassifier(*params)

    if algorithm == 'neural_net':
        return MLPClassifier(*params)

    raise "Invalid Algorithm"


def join_datasets(df_adu, df_text):
    """
        Join By Some Criteria
    """
    return df_adu


def corpus_extraction(df):
    corpus = []

    for row in df['tokens']:
        corpus.append(row)

    return corpus


def label_extraction(df):
    labels = []

    for row in df['label']:
        labels.append(row)

    return labels


def normalize_corpus(corpus):
    corpus_aux = []

    ''' Lemmatizer
    nlp = spacy.load("pt_core_news_sm")
    
    for doc in nlp.pipe(corpus, batch_size=32, n_process=3, disable=["parser", "ner"]):
        lemmas = []
        for token in doc:
            lemmas.append(token.lemma_)

        row = ' '.join([w for w in lemmas if w not in set(stopwords.words('portuguese'))])
    '''
    
    stemmer = SnowballStemmer('portuguese')

    for row in corpus:
        word_list = nltk.word_tokenize(row)
        row = ' '.join([stemmer.stem(w) for w in word_list if not w in set(stopwords.words('portuguese'))])
        corpus_aux.append(row)

    return corpus_aux


def vectorize_bag_of_words(corpus, labels, max_features=None):
    vectorizer = CountVectorizer(max_features=max_features)

    X = vectorizer.fit_transform(corpus).toarray()
    y = labels

    return X, y


def split_train_test(X, y, test_size=0.20):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=2, stratify=y)

    return X_train, X_test, y_train, y_test


def apply_clf(clf, X_train, y_train, X_test):
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    return y_pred


def baseline(df_adu, df_text, algorithm='naive_bayes'):
    drop_columns(df_adu, ['article_id', 'node', 'annotator'])
    drop_columns(df_text, ['article_id', 'title', 'authors', 'meta_description',
                           'topics', 'keywords', 'publish_date', 'url_canonical'])

    corpus = normalize_corpus(corpus_extraction(df_adu))

    labels = label_extraction(df_adu)

    X, y = vectorize_bag_of_words(corpus, labels)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    clf = clf_factory(algorithm)
    y_pred = apply_clf(clf, X_train=X_train, y_train=y_train, X_test=X_test)

    evaluate_results(y_pred=y_pred, y_test=y_test)
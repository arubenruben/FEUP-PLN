import plotly.express as px


def most_common_words(X, vec, n=None):
    sum_words = X.sum(axis=0)

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

    for word, freq in words_freq[:n]:
        print(word, freq)


def size_vocabulary(vectorizer):
    print(f"Size of the Vocabulary:{len(vectorizer.get_feature_names_out())}")
    return len(vectorizer.get_feature_names_out())


def class_distribution(df):
    aggr_df = df.groupby(['label'])['label'].count().reset_index(name='counts')

    fig = px.bar(aggr_df, x='label', y='counts')
    fig.show()

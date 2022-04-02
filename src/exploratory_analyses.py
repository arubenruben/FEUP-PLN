import json
import pickle

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


"""
    Consider ADUs with different classifications 
"""

"""
    Split ADUs por artigo
    {
        'article_id':
                    {
                        'A'
                        'B'
                        'C'
                    }
    }
    """


def outlier_detection(df_adu):
    results = {}

    for _, row in df_adu.iterrows():
        if row['article_id'] not in results.keys():
            results[row['article_id']] = {
                'A': [],
                'B': [],
                'C': [],
                'D': []
            }

        if row['annotator'] == 'A':
            results[row['article_id']]['A'].append({
                'ranges': row['ranges'],
                'tokens': row['tokens'],
                'label': row['label']
            })

        elif row['annotator'] == 'B':
            results[row['article_id']]['B'].append({
                'ranges': row['ranges'],
                'tokens': row['tokens'],
                'label': row['label']
            })

        elif row['annotator'] == 'C':
            results[row['article_id']]['C'].append({
                'ranges': row['ranges'],
                'tokens': row['tokens'],
                'label': row['label']
            })

        elif row['annotator'] == 'D':
            results[row['article_id']]['D'].append({
                'ranges': row['ranges'],
                'tokens': row['tokens'],
                'label': row['label']
            })

    for article_id in results.keys():
        for adu_A in results[article_id]['A']:
            adu_matching(adu_A, results[article_id]['B'], results[article_id]['C'], results[article_id]['D'])
        # TODO: Complete with other cases
        # for adu_B in results[article_id]['B']:
        #   adu_matching(adu_B, results[article_id]['B'], results[article_id]['C'], results[article_id]['D'])


def adu_matching(adu, list_annotater_X, list_annotater_Y, list_annotater_Z):
    list_collisions = []
    # TODO: Remove those elements
    for iterable in [list_annotater_X, list_annotater_Y, list_annotater_Z]:
        for elem in iterable:
            if json.loads(adu['ranges'])[0][0] < json.loads(elem['ranges'])[0][0] < json.loads(adu['ranges'])[0][1]:
                if adu['label'] == elem['label']:
                    pass
                    # print("Agreement")
                else:
                    print(f"Disagreement between:\n{adu['tokens']} \n and \n {elem['tokens']}")

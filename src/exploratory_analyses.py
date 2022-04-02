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


def outlier_detection(df_adu, option='delete'):
    results = {}
    list_collision = []

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

        counter = 0

        for adu_A in results[article_id]['A']:
            counter += adu_matching(adu_A, results[article_id]['B'])
            counter += adu_matching(adu_A, results[article_id]['C'])
            counter += adu_matching(adu_A, results[article_id]['D'])
        for adu_B in results[article_id]['B']:
            counter += adu_matching(adu_B, results[article_id]['A'])
            counter += adu_matching(adu_B, results[article_id]['C'])
            counter += adu_matching(adu_B, results[article_id]['D'])
        for adu_C in results[article_id]['C']:
            counter += adu_matching(adu_C, results[article_id]['A'])
            counter += adu_matching(adu_C, results[article_id]['B'])
            counter += adu_matching(adu_C, results[article_id]['D'])
        for adu_D in results[article_id]['D']:
            counter += adu_matching(adu_D, results[article_id]['A'])
            counter += adu_matching(adu_D, results[article_id]['B'])
            counter += adu_matching(adu_D, results[article_id]['C'])

        if counter > 0:
            list_collision.append(article_id)

    return list_collision


def adu_matching(adu, list_annotater):
    list_collisions = []

    for elem in list_annotater:
        if json.loads(adu['ranges'])[0][0] < json.loads(elem['ranges'])[0][0] < json.loads(adu['ranges'])[0][1]:
            if adu['label'] != elem['label']:
                # print(f"Disagreement between:\n{adu['tokens']} \n and \n {elem['tokens']}")
                list_collisions.append(elem)

    return len(list_collisions)


def deal_with_outliers(df_adu, list_collisions, option='delete'):
    # print(f"Before:{df_adu.describe()}")

    if option == 'delete':
        for elem in list_collisions:
            df_adu = df_adu[df_adu.article_id != elem]
    elif option == 'majority':
        pass

    # print(f"After:{df_adu.describe()}")

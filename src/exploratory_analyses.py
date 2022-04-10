import json
import random
import matplotlib.pyplot as plt
import plotly.express as px

from other import remove_dataframe_rows_by_id


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
    dict_collisions = {}
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
                'id': row['id'],
                'ranges': row['ranges'],
                'tokens': row['tokens'],
                'label': row['label']
            })

        elif row['annotator'] == 'B':
            results[row['article_id']]['B'].append({
                'id': row['id'],
                'ranges': row['ranges'],
                'tokens': row['tokens'],
                'label': row['label']
            })

        elif row['annotator'] == 'C':
            results[row['article_id']]['C'].append({
                'id': row['id'],
                'ranges': row['ranges'],
                'tokens': row['tokens'],
                'label': row['label']
            })

        elif row['annotator'] == 'D':
            results[row['article_id']]['D'].append({
                'id': row['id'],
                'ranges': row['ranges'],
                'tokens': row['tokens'],
                'label': row['label']
            })

    for article_id in results.keys():

        for adu_A in results[article_id]['A']:
            adu_matching(adu_A, results[article_id]['B'], results[article_id]['C'], results[article_id]['D'],
                         dict_collisions)
        for adu_B in results[article_id]['B']:
            adu_matching(adu_B, results[article_id]['A'], results[article_id]['C'], results[article_id]['D'],
                         dict_collisions)
        for adu_C in results[article_id]['C']:
            adu_matching(adu_C, results[article_id]['A'], results[article_id]['B'], results[article_id]['D'],
                         dict_collisions)
        for adu_D in results[article_id]['D']:
            adu_matching(adu_D, results[article_id]['A'], results[article_id]['B'], results[article_id]['C'],
                         dict_collisions)

    return dict_collisions


def adu_matching(adu, list_annotater_X, list_annotater_Y, list_annotater_Z, dict_collisions):
    for iterator in [list_annotater_X, list_annotater_Y, list_annotater_Z]:
        for elem in iterator:
            if json.loads(adu['ranges'])[0][0] < json.loads(elem['ranges'])[0][0] < json.loads(adu['ranges'])[0][1]:
                if adu['label'] != elem['label']:
                    # print(f"Disagreement between:\n{adu['tokens']} \n and \n {elem['tokens']}")
                    if adu['id'] not in dict_collisions.keys():
                        dict_collisions[adu['id']] = [elem['id']]
                    else:
                        dict_collisions[adu['id']].append(elem['id'])


def deal_with_outliers(df_adu, dict_collisions, option='delete'):
    # print(f"Before:{df_adu.describe()}")

    if option == 'delete':
        list_to_remove = []

        for key_left in dict_collisions.keys():
            list_to_remove.append(key_left)
            for elem in dict_collisions[key_left]:
                list_to_remove.append(elem)

        remove_dataframe_rows_by_id(df_adu, list_to_remove)

    elif option == 'majority':
        list_to_remove = []
        for key_left in dict_collisions.keys():
            counters = {
                'Fact': 0,
                'Policy': 0,
                'Value': 0,
                'Value(+)': 0,
                'Value(-)': 0,
            }

            majority_vote = None
            number_votes = 0

            adu = df_adu.loc[df_adu['id'] == key_left].iloc[0]

            counters[adu['label']] += 1

            for elem in dict_collisions[key_left]:
                adu = df_adu.loc[df_adu['id'] == elem].iloc[0]
                counters[adu['label']] += 1

            for elem in counters.keys():
                number_votes += counters[elem]

            """
            Find the majority vote type
            Majority_Vote returns a Valid Label
            """

            for elem in counters.keys():
                if counters[elem] / number_votes >= 0.5:
                    majority_vote = elem
                    break

            if not majority_vote:
                continue

            if adu['label'] != majority_vote:
                list_to_remove.append(adu['id'])

            for elem in dict_collisions[key_left]:
                elem_adu = df_adu.loc[df_adu['id'] == elem].iloc[0]
                if elem_adu['label'] != majority_vote:
                    list_to_remove.append(elem_adu['id'])

        remove_dataframe_rows_by_id(df_adu, list_to_remove)

    # print(f"After:{df_adu.describe()}")


def study_sparsity_of_matrix(X):
    plt.spy(X)
    plt.title("Sparse Matrix")

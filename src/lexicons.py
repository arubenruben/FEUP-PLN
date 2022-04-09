import os.path

import pandas as pd

from other import drop_columns

lexicons = pd.DataFrame()


def load_lexicons():
    global lexicons

    lexicons = pd.read_csv(os.path.join('dataset', 'lexicons', 'lexico_v3.0.csv'))

    drop_columns(lexicons, ['type', 'idk'])

    lexicons = lexicons.to_dict('records')
    """
    res = defaultdict(list)
    for sub in lexicons:
        for key in sub:
            res[key].append(sub[key])

    lexicons = res
    """

    newDict = {}

    for element in lexicons:
        newDict[element['token']] = element['polarity']

    lexicons = newDict

    # print(newDict)


"""
    Words are lowercase because lexicons are all lowercase
    
    Negative connotation:-1
    Neutral connotation:0
    Positive connotation:1
    Unknown connotation:2
"""


def get_polarity(word: str) -> int:
    if word.lower() not in lexicons.keys():
        return 2

    polarity = lexicons[word.lower()]

    # print(f"Polarity of {word} is: {polarity}")

    return polarity

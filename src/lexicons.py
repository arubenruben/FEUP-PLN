import os.path
import pandas as pd

from src.other import drop_columns

lexicons = pd.DataFrame()


def load_lexicons():
    global lexicons

    lexicons = pd.read_csv(os.path.join('dataset', 'lexicons', 'lexico_v3.0.csv'))

    drop_columns(lexicons, ['type', 'idk'])


"""
    Words are lowercase because lexicons are all lowercase
    
    Negative connotation:-1
    Neutral connotation:0
    Positive connotation:1
    Unknown connotation:2
"""


def get_polarity(word: str) -> int:
    df_polarity = lexicons[lexicons['token'] == word.lower()]

    if len(df_polarity.index) == 0:
        return 2

    polarity = df_polarity.iloc[0]['polarity']

    # print(f"Polarity of {word} is: {polarity}")

    return polarity

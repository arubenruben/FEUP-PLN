import spacy

nlp = spacy.load("pt_core_news_sm")

"""
    ADJ: adjective
    INTJ: interjection
    VERB: verb
    PROPN: proper noun
"""


def get_pos_numbers(sentence):
    doc = nlp(sentence)
    number_adj = 0
    number_adverbs = 0
    number_determinants = 0
    number_interjections = 0
    number_verbs = 0
    number_proper_nouns = 0

    for token in doc:
        if token.pos_ == 'ADJ':
            number_adj += 1
        elif token.pos_ == 'ADV':
            number_adverbs += 1
        elif token.pos_ == 'DET':
            number_determinants += 1
        elif token.pos_ == 'INTJ':
            number_interjections += 1
        elif token.pos_ == 'VERB':
            number_verbs += 1
        elif token.pos_ == 'PROPN':
            number_proper_nouns += 1

    return number_adj, number_interjections, number_verbs, number_proper_nouns

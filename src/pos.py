import spacy

nlp = spacy.load("pt_core_news_sm", disable=["tok2vec", "parser", "attribute_ruler", "lemmatizer"])


def get_pos_numbers(sentence):
    doc = nlp(sentence)
    number_adj = 0
    number_interjections = 0
    number_verbs = 0
    number_proper_nouns = 0

    """
    ADJ: adjective
    INTJ: interjection
    VERB: verb
    PROPN: proper
    noun
    """
    for token in doc:
        print(token.text, token.pos_)

    return number_adj, number_interjections, number_verbs, number_proper_nouns

import nltk
import spacy as spacy
from nltk import tokenize, SnowballStemmer
from nltk.corpus import stopwords
import re

from src.other import write_new_csv_datatest

nlp = spacy.load("pt_core_news_sm")

stemmer = SnowballStemmer('portuguese')

stop_words = set(stopwords.words('portuguese'))


def tokenization(sentence):
    return tokenize.word_tokenize(sentence, language='portuguese')


def lemmatization(sentence):
    lemmas = []

    for token in nlp(sentence):
        lemmas.append(token.lemma_)

    return ' '.join(lemmas)


def stemming(token):
    return stemmer.stem(token)


def is_stop_word(token):
    return token in stop_words


def sentence_segmentation(raw_text):
    sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
    sentences = sent_tokenizer.tokenize(raw_text)
    return sentences


def insert_previous_and_after_sentence_to_adu(df_adu, df_text):
    for pandas_index, row in df_adu.iterrows():
        article_row = df_text.loc[df_text['article_id'] == row['article_id']].iloc[0]
        sentences = sentence_segmentation(article_row['body'])

        for sentence_index, value in enumerate(sentences):
            if value.find(row['tokens']) != -1:
                new_str = ""

                if sentence_index - 1 > 0:
                    new_str += sentences[sentence_index - 1]

                new_str += value

                if len(sentences) < sentence_index + 1:
                    new_str += sentences[sentence_index + 1]

                df_adu.at[pandas_index, 'tokens'] = new_str

                break

    write_new_csv_datatest(df_adu, "neighbours_sentence")


def remove_punctuation(sentence):
    return re.sub(r'[^\w\s]', '', sentence)


def normalize_corpus(corpus):
    corpus_aux = []

    for row in corpus:
        filtered_list = []

        for token in tokenization(row):

            if is_stop_word(token):
                continue

            # filtered_list.append(stemming(token))
            
            filtered_list.append(token)

        sentence_without_punctuation = remove_punctuation(' '.join(filtered_list))

        corpus_aux.append(sentence_without_punctuation)

    # print(corpus_aux)

    return corpus_aux

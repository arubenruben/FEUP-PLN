import nltk
from nltk import tokenize

# TODO:Insert Transformation In this File
from src.other import write_new_csv_datatest


def tokenization(text):
    return tokenize.word_tokenize(text, language='portuguese')


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

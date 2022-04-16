from sklearn.manifold import TSNE
import numpy as np
from datetime import datetime
import gensim

from other import load_embedding_model, save_embedding_model
from text_processing import corpus_to_list


def train_embedding_model(corpus):
    corpus = corpus_to_list(corpus)
    model = gensim.models.Word2Vec(corpus, vector_size=100, window=3, min_count=2, workers=10, sg=1)
    save_embedding_model(model, "trained_model")
    return load_embedding_model("trained_model", saved_model=True)

def text_to_vector(embeddings, text, sequence_len):
    tokens = text.split()
    
    vec = []
    n = 0
    i = 0
    while i < len(tokens) and n < sequence_len:  
        try:
            vec.extend(embeddings.get_vector(tokens[i]))
            n += 1
        except KeyError:
            True  
        finally:
            i += 1

    for j in range(sequence_len - n):
        vec.extend(np.zeros(embeddings.vector_size,))
    
    return vec

def apply_embedding_model(model, corpus):
    embeddings_corpus = []
    for row in corpus:
        embeddings_corpus.append(text_to_vector(model, row, 10))

    return  np.array(embeddings_corpus)


"""
Things that help to work with embeddings
"""

from typing import List, Tuple
import numpy as np
from utils import preprocess_sentence


def get_embedding_matrix(path: str, skip_line: bool = False) -> Tuple[np.ndarray, dict]:
    """
    Function returns:
    1) Embedding matrix
    2) Vocabulary
    """

    embeddings = dict()
    vocabulary = []

    with open(path, 'r') as file:
        if skip_line:
            file.readline()
        for line in file:
            values = line.split()
            word = values[0]
            coefs = np.array(values[1:], dtype=np.float64)
            embeddings[word] = coefs
            vocabulary.append(word)

    embedding_size = list(embeddings.values())[1].shape[0]

    embedding_matrix = np.zeros((len(vocabulary) + 1, embedding_size))
    embedding_matrix[-1] = np.mean(np.array(list(embeddings.values())), axis=0)

    vocab = dict()
    vocab['UNKNOWN_TOKEN'] = len(vocabulary)
    for i, word in enumerate(vocabulary):
        embedding_matrix[i] = embeddings[word]
        vocab[word] = i

    return embedding_matrix, vocab


def tokens_to_indexes(words: List[str], vocab: dict) -> List[int]:
    indexes = []
    for word in words:
        if word in vocab:
            indexes.append(vocab[word])
    return indexes


def sentence_to_indexes(sentence: str, vocab: dict) -> List[int]:
    tokens = preprocess_sentence(sentence)
    if len(tokens) == 0:
        return [vocab['UNKNOWN_TOKEN']]
    indexes = []
    for token in tokens:

        if token in vocab:
            indexes.append(vocab[token])
        else:
            indexes.append(vocab['UNKNOWN_TOKEN'])

    return indexes


def inds_to_embeddings(indexes: List[int], emb_matrix: np.ndarray, ngrams: int = 1) -> np.ndarray:
    if ngrams > 1:
        embedded_sent = emb_matrix[indexes]
        sent_len = len(indexes)
        remainder = sent_len % ngrams

        splitted = np.split(embedded_sent, np.arange(ngrams, sent_len, ngrams))

        if remainder == 0:
            embedded_sent = np.mean(splitted, axis=1)
        else:
            padded = np.zeros(splitted[0].shape)
            padded[:remainder] = splitted[-1]
            splitted[-1] = padded
            embedded_sent = np.mean(splitted, axis=1)

        return embedded_sent.T
    # shape: [d, n] (embedding dim, number of words)
    return emb_matrix[indexes].T

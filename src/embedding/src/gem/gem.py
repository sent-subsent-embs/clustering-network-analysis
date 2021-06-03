

from typing import List
import numpy as np
from embeddings import inds_to_embeddings, sentence_to_indexes
from tqdm import tqdm


def modified_gram_schmidt_qr(_a):
    n_rows, n_cols = _a.shape
    q = np.zeros((n_rows, n_cols))
    r = np.zeros((n_rows, n_cols))
    for j in range(n_cols):
        u = np.copy(_a[:, j])
        for i in range(j):
            proj = np.dot(u, q[:, i]) * q[:, i]
            u -= proj
        u_norm = np.linalg.norm(u, ord=2, axis=0)
        if u_norm != 0:
            u /= u_norm
        q[:, j] = u
    for j in range(n_cols):
        for i in range(j+1):
            r[i, j] = _a[:, j].dot(q[:, i])
    return q, r


class SentenceEmbedder:

    def __init__(self,
                 sentences_raw: List[str],
                 embedding_matrix: np.ndarray,
                 vocab: dict) -> None:
        self.vocab = vocab
        self.sentences_raw = sentences_raw
        self.sentences = []
        for sent in self.sentences_raw:
            self.sentences.append(sentence_to_indexes(sent, self.vocab))
        self.embedding_matrix = embedding_matrix
        self.emb_dim = self.embedding_matrix.shape[1]
        self.singular_values = None

    def gem(self,
            window_size: int = 7,
            k: int = 45,
            h: int = 17,
            sigma_power: int = 3,
            ngrams: int = 1) -> np.ndarray:
        X = np.zeros((self.emb_dim, len(self.sentences)))
        embedded_sentences = []
        for i, sent in enumerate(self.sentences):
            embedded_sent = inds_to_embeddings(sent, self.embedding_matrix, ngrams)
            embedded_sentences.append(embedded_sent)
            U, s, Vh = np.linalg.svd(embedded_sent, full_matrices=False)
            X[:, i] = U.dot(s ** sigma_power)
        D, s, _ = np.linalg.svd(X, full_matrices=False)
        self.singular_values = s.copy()
        D = D[:, :k]
        s = s[:k]

        C = np.zeros((self.emb_dim, len(self.sentences)))
        for j, sent in tqdm(enumerate(self.sentences)):
            embedded_sent = embedded_sentences[j]
            order = s * np.linalg.norm(embedded_sent.T.dot(D), axis=0)
            toph = order.argsort()[::-1][:h]
            alpha = np.zeros(embedded_sent.shape[1])

            for i in range(embedded_sent.shape[1]):
                window_matrix = self._context_window(i, window_size, embedded_sent)
                Q, R = modified_gram_schmidt_qr(window_matrix)
                q = Q[:, -1]
                r = R[:, -1]
                alpha_n = np.exp(r[-1] / (np.linalg.norm(r, ord=2, axis=0)) + 1e-18)
                alpha_s = r[-1] / window_matrix.shape[1]
                alpha_u = np.exp(-np.linalg.norm(s[toph] * (q.T.dot(D[:, toph]))) / h)
                alpha[i] = alpha_n + alpha_s + alpha_u

            C[:, j] = embedded_sent.dot(alpha)
            C[:, j] = C[:, j] - D.dot(D.T.dot(C[:, j]))

        sentence_embeddings = C.T
        return sentence_embeddings

    def mean_embeddings(self) -> np.ndarray:
        """
        Averages embeddings of words to get a sentence representation.
        Returns:
            sentence_embeddings: shape [n, d]
        """
        C = np.zeros((self.emb_dim, len(self.sentences)))

        for i, sent in enumerate(self.sentences):
            embedded_sent = inds_to_embeddings(indexes=sent, emb_matrix=self.embedding_matrix)
            C[:, i] = np.mean(embedded_sent, axis=1)

        sentence_embeddings = C.T
        return sentence_embeddings

    def _context_window(self, i: int, m: int, embeddings: np.ndarray) -> np.ndarray:
        """
        Given embedded sentence returns  the contextual window matrix of word w_i
        """
        left_window = embeddings[:, i - m:i]
        right_window = embeddings[:, i + 1:i + m + 1]
        word_embedding = embeddings[:, i][:, None]
        window_matrix = np.hstack([left_window, right_window, word_embedding])
        return window_matrix

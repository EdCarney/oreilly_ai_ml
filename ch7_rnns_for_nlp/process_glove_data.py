import numpy as np
from typing import Dict
from sarcasm_data_processing import get_sarcasm_tokenizer


def _get_embeddings(filepath: str) -> Dict[str, np.ndarray]:
    glove_embeddings = dict()

    with open(filepath, mode="r") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coeffs = np.asarray(values[1:], dtype="float32")
            glove_embeddings[word] = coeffs

    return glove_embeddings


def _get_tokenizer_embedding(
    vocab_sz: int, embeddings: Dict[str, np.ndarray]
) -> np.ndarray:

    tokenizer = get_sarcasm_tokenizer(vocab_sz)
    embedding_sz = len(next(iter(embeddings.values())))
    embedding_mat = np.zeros((vocab_sz, embedding_sz))

    for word, index in tokenizer.word_index.items():
        if index == vocab_sz:
            break
        embedding_vec = embeddings.get(word)
        if embedding_vec is None:
            continue
        embedding_mat[index] = embedding_vec

    return embedding_mat


def get_embedding_matrix(filepath: str, vocab_sz: int) -> np.ndarray:
    raw_embeddings = _get_embeddings(filepath)
    tokenizer_embedding = _get_tokenizer_embedding(vocab_sz, raw_embeddings)
    return tokenizer_embedding

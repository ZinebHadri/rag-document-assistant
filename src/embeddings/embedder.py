from functools import lru_cache

import numpy as np


MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Load the SentenceTransformer model once
    and keep it cached in memory.
    """

    from sentence_transformers import (
        SentenceTransformer
    )

    return SentenceTransformer(
        MODEL_NAME
    )


def embed_texts(texts):
    """
    Generate normalized embeddings
    for a list of texts.
    """

    if not texts:
        return np.empty(
            (0, 0),
            dtype="float32"
        )

    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return np.asarray(
        embeddings,
        dtype="float32"
    )


def embed_query(query):
    """
    Generate one normalized embedding
    for a query.
    """

    if not query:
        return None

    embeddings = embed_texts(
        [query]
    )

    if embeddings.size == 0:
        return None

    return embeddings
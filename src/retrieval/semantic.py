import numpy as np

from src.embeddings import (
    get_embedding_model,
    embed_query
)

from src.vectorstore import (
    create_faiss_index,
    load_or_rebuild_index
)


def expand_question(question):
    """
    Add lightweight query expansion
    for common technical abbreviations.
    """

    expanded_question = question

    question_lower = (
        question
        .lower()
        .strip()
    )

    if "csp" in question_lower:

        expanded_question += (
            " Constraint Satisfaction Problem"
            " Constraint Programming"
            " variables domains constraints"
        )

    if "pca" in question_lower:

        expanded_question += (
            " Principal Component Analysis"
            " dimensionality reduction"
        )

    if "rag" in question_lower:

        expanded_question += (
            " Retrieval Augmented Generation"
            " retrieval generation embeddings"
        )

    return expanded_question


def semantic_search(
    question,
    chunks,
    index,
    top_k=10
):
    """
    Retrieve chunks using semantic similarity.
    """

    if (
        not question
        or not chunks
        or index is None
        or index.ntotal == 0
    ):

        return []

    expanded_question = (
        expand_question(
            question
        )
    )

    query_embedding = embed_query(
        expanded_question
    )

    if query_embedding is None:
        return []

    search_size = min(
        top_k,
        index.ntotal
    )

    scores, indices = (
        index.search(
            query_embedding,
            search_size
        )
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        idx = int(
            idx
        )

        if idx < 0:
            continue

        results.append(
            {
                "index": idx,

                "chunk":
                    chunks[idx],

                "semantic_score":
                    float(score)
            }
        )

    return results
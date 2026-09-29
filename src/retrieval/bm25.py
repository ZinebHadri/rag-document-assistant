import re

import numpy as np

from rank_bm25 import BM25Okapi

from src.retrieval.semantic import (
    expand_question
)


def tokenize_text(text):
    """
    Convert text into lowercase word tokens.
    """

    if not text:
        return []

    return re.findall(
        r"\b\w+\b",
        text.lower()
    )


def build_bm25_index(chunks):
    """
    Build a BM25 index from document chunks.
    """

    if not chunks:
        return None

    tokenized_corpus = [
        tokenize_text(
            chunk.get(
                "text",
                ""
            )
        )
        for chunk in chunks
    ]

    return BM25Okapi(
        tokenized_corpus
    )


def get_bm25_scores(
    question,
    chunks
):
    """
    Calculate BM25 scores for all chunks.

    Returns a dictionary:
        chunk_index -> BM25 score
    """

    if not chunks:
        return {}

    bm25 = build_bm25_index(
        chunks
    )

    if bm25 is None:
        return {}

    expanded_question = expand_question(
        question
    )

    query_tokens = tokenize_text(
        expanded_question
    )

    scores = bm25.get_scores(
        query_tokens
    )

    return {
        idx: float(score)
        for idx, score
        in enumerate(scores)
    }


def bm25_search(
    question,
    chunks,
    top_k=10
):
    """
    Retrieve chunks using BM25 lexical search.
    """

    if (
        not question
        or not chunks
    ):
        return []

    scores = get_bm25_scores(
        question,
        chunks
    )

    if not scores:
        return []

    ranked_indices = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    results = []

    for idx in ranked_indices[:top_k]:

        results.append(
            {
                "index": idx,
                "chunk": chunks[idx],
                "keyword_score": scores[idx]
            }
        )

    return results
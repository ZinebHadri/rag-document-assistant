import numpy as np

from src.retrieval import (
    tokenize_text,
    normalize_scores
)


def test_tokenize_text():

    text = (
        "Principal Component Analysis reduces dimensions."
    )

    tokens = tokenize_text(
        text
    )

    assert "principal" in tokens
    assert "component" in tokens
    assert "analysis" in tokens
    assert "dimensions" in tokens


def test_tokenize_text_lowercase():

    text = "FAISS BM25 RAG"

    tokens = tokenize_text(
        text
    )

    assert tokens == [
        "faiss",
        "bm25",
        "rag"
    ]


def test_normalize_scores():

    scores = {
        0: 2.0,
        1: 4.0,
        2: 6.0
    }

    normalized = normalize_scores(
        scores
    )

    assert normalized[0] == 0.0
    assert normalized[2] == 1.0

    assert 0.0 <= normalized[1] <= 1.0


def test_normalize_identical_scores():

    scores = {
        0: 5.0,
        1: 5.0,
        2: 5.0
    }

    normalized = normalize_scores(
        scores
    )

    assert normalized[0] == 1.0
    assert normalized[1] == 1.0
    assert normalized[2] == 1.0
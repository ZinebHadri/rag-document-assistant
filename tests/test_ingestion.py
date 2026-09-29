from src.ingestion import split_text_by_sentences


def test_split_text_returns_chunks():

    text = (
        "This is the first sentence. "
        "This is the second sentence. "
        "This is the third sentence."
    )

    chunks = split_text_by_sentences(
        text,
        max_chars=50
    )

    assert isinstance(
        chunks,
        list
    )

    assert len(chunks) > 0


def test_chunks_are_not_empty():

    text = (
        "Machine learning is useful. "
        "RAG combines retrieval and generation."
    )

    chunks = split_text_by_sentences(
        text,
        max_chars=100
    )

    for chunk in chunks:

        assert chunk.strip() != ""


def test_chunk_size_is_reasonable():

    text = (
        "Sentence one. "
        "Sentence two. "
        "Sentence three. "
        "Sentence four."
    )

    chunks = split_text_by_sentences(
        text,
        max_chars=40
    )

    for chunk in chunks:

        assert len(chunk) <= 60
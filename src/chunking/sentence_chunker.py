import nltk


def ensure_nltk_resources():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]

    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(
                resource_name,
                quiet=True
            )


def split_text_by_sentences(
    text,
    max_chars=500
):
    """
    Split text into chunks while trying to preserve
    complete sentences.

    Parameters
    ----------
    text : str
        Input text.

    max_chars : int
        Approximate maximum chunk size.

    Returns
    -------
    list[str]
        List of text chunks.
    """

    if not text:
        return []

    text = text.strip()

    if not text:
        return []

    ensure_nltk_resources()

    sentences = nltk.sent_tokenize(
        text
    )

    chunks = []
    current_chunk = ""

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        candidate = (
            f"{current_chunk} {sentence}".strip()
        )

        if (
            current_chunk
            and len(candidate) > max_chars
        ):
            chunks.append(
                current_chunk
            )

            current_chunk = sentence

        else:
            current_chunk = candidate

    if current_chunk:
        chunks.append(
            current_chunk
        )

    return chunks
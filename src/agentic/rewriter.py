from src.agentic.grader import (
    generate_short_text,
)


MAX_QUERY_WORDS = 20


def clean_rewritten_query(
    text: str,
    fallback: str,
) -> str:
    """
    Clean the LLM output so that
    it stays a retrieval query.
    """

    if not text:
        return fallback

    text = text.strip()

    text = (
        text
        .splitlines()[0]
        .strip()
    )

    prefixes = [
        "Rewritten query:",
        "Query:",
        "Search query:",
    ]

    for prefix in prefixes:

        if text.lower().startswith(
            prefix.lower()
        ):

            text = (
                text[len(prefix):]
                .strip()
            )

    words = text.split()

    if (
        len(words)
        > MAX_QUERY_WORDS
    ):

        text = " ".join(
            words[
                :MAX_QUERY_WORDS
            ]
        )

    if not text:
        return fallback

    return text


def rewrite_search_query(
    question: str,
    current_query: str,
) -> str:
    """
    Ask Qwen to produce a short
    retrieval-oriented query.
    """

    prompt = f"""
You rewrite search queries for a document retrieval system.

Original question:
{question}

Previous search query:
{current_query}

The previous retrieval attempt was not relevant.

Create one better retrieval query.

Rules:
- Keep the original meaning.
- Expand abbreviations when useful.
- Add important technical keywords.
- Maximum 15 words.
- Do not answer the question.
- Do not explain.
- Do not write complete paragraphs.
- Output only the search query.

Example:

Question:
What is PCA?

Output:
Principal Component Analysis PCA dimensionality reduction variance principal components

Search query:
""".strip()

    raw_query = generate_short_text(
        prompt,
        max_new_tokens=25,
    )

    return clean_rewritten_query(
        raw_query,
        fallback=current_query,
    )
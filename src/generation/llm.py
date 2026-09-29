from functools import lru_cache


MODEL_NAME = (
    "Qwen/Qwen2.5-0.5B-Instruct"
)


@lru_cache(maxsize=1)
def get_generator():
    """
    Load and cache the local generation model.
    """

    from transformers import (
        pipeline
    )

    return pipeline(
        "text-generation",
        model=MODEL_NAME,
        device=-1
    )


def build_context(
    retrieved_results,
    max_sources=3
):
    """
    Build the context sent to the LLM
    using the best retrieved chunks.
    """

    parts = []

    for result in (
        retrieved_results[
            :max_sources
        ]
    ):

        chunk = result[
            "chunk"
        ]

        source = chunk.get(
            "source",
            "Unknown"
        )

        page = chunk.get(
            "page",
            "Unknown"
        )

        text = chunk.get(
            "text",
            ""
        )

        parts.append(
            f"""
Source: {source}
Page: {page}

{text}
""".strip()
        )

    return "\n\n".join(
        parts
    )


def generate_answer(
    question,
    retrieved_results
):
    """
    Generate a grounded answer
    using only retrieved context.
    """

    if not retrieved_results:

        return (
            "I cannot find enough information "
            "in the provided documents."
        )

    context = build_context(
        retrieved_results,
        max_sources=3
    )

    prompt = f"""
Answer the question using ONLY the context below.

Rules:
- Use only the provided context.
- Do not invent information.
- Give a concise answer in 2 complete sentences.
- Do not stop in the middle of a sentence.
- If the answer is not in the context, say:
  "I cannot find enough information in the provided documents."

Context:
{context}

Question:
{question}

Answer:
""".strip()

    generator = get_generator()

    result = generator(
        prompt,
        max_new_tokens=128,
        do_sample=False,
        return_full_text=False
    )

    answer = (
        result[0][
            "generated_text"
        ]
        .strip()
    )

    if not answer:

        return (
            "I cannot find enough information "
            "in the provided documents."
        )

    return answer
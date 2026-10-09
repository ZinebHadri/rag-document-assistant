from functools import lru_cache

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


@lru_cache(maxsize=1)
def get_llm():
    """
    Load tokenizer and Qwen model only once.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME
    )

    return tokenizer, model


def build_context(
    results,
    max_chunks=3,
):
    """
    Build the RAG context from the
    top retrieved chunks.
    """

    context_parts = []

    for result in results[:max_chunks]:

        chunk = result.get(
            "chunk",
            {},
        )

        text = (
            chunk.get("text")
            or chunk.get("content")
            or ""
        )

        source = chunk.get(
            "source",
            "unknown",
        )

        page = chunk.get(
            "page",
            "unknown",
        )

        if not text:
            continue

        context_parts.append(
            f"""
Source: {source}
Page: {page}

{text}
""".strip()
        )

    return "\n\n---\n\n".join(
        context_parts
    )


def generate_answer(
    question: str,
    results: list,
) -> str:
    """
    Generate a grounded answer using
    only retrieved context.
    """

    if not results:
        return (
            "I don't have enough relevant "
            "information in the current "
            "knowledge base to answer "
            "this question reliably."
        )

    context = build_context(
        results
    )

    if not context:
        return (
            "I don't have enough relevant "
            "information in the current "
            "knowledge base to answer "
            "this question reliably."
        )

    tokenizer, model = get_llm()

    system_message = (
        "You are a document question-answering assistant. "
        "Answer only using the provided context. "
        "Do not use outside knowledge. "
        "If the context does not contain enough information, "
        "say that you do not have enough information."
    )

    user_message = f"""
Context:

{context}

Question:

{question}

Give a clear and concise answer in 2 to 4 complete sentences.
""".strip()

    messages = [
        {
            "role": "system",
            "content": system_message,
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    output_ids = model.generate(
        **inputs,
        max_new_tokens=180,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated_ids = output_ids[
        0,
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return answer.strip()
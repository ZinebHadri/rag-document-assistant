from functools import lru_cache

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

LOW_SEMANTIC_THRESHOLD = 0.30
HIGH_SEMANTIC_THRESHOLD = 0.55


@lru_cache(maxsize=1)
def get_agent_llm():
    """
    Load Qwen only once.

    The same model can be reused by
    the grader and query rewriter.
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
    Build a compact context from
    the best retrieved chunks.
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


def get_best_semantic_score(
    results,
) -> float:
    """
    Return the highest absolute
    semantic similarity score.
    """

    if not results:
        return 0.0

    scores = []

    for result in results:

        score = result.get(
            "semantic_score",
            0.0,
        )

        try:
            score = float(score)

        except (
            TypeError,
            ValueError,
        ):
            score = 0.0

        scores.append(
            score
        )

    if not scores:
        return 0.0

    return max(scores)


def generate_short_text(
    prompt: str,
    max_new_tokens: int,
) -> str:
    """
    Run deterministic local Qwen generation.
    """

    tokenizer, model = (
        get_agent_llm()
    )

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    formatted_prompt = (
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
    )

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated_ids = output_ids[
        0,
        inputs["input_ids"].shape[1]:
    ]

    text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return text.strip()


def grade_relevance(
    question: str,
    results: list,
) -> bool:
    """
    Three-zone relevance grader.

    LOW score:
        reject directly.

    HIGH score:
        accept directly.

    MIDDLE score:
        ask Qwen for additional judgement.
    """

    if not results:

        print(
            "[GRADER] No retrieval results."
        )

        return False

    best_semantic_score = (
        get_best_semantic_score(
            results
        )
    )

    print(
        f"[GRADER] Best semantic score: "
        f"{best_semantic_score:.4f}"
    )

    print(
        f"[GRADER] Low threshold: "
        f"{LOW_SEMANTIC_THRESHOLD:.2f}"
    )

    print(
        f"[GRADER] High threshold: "
        f"{HIGH_SEMANTIC_THRESHOLD:.2f}"
    )

    # -----------------------------------
    # ZONE 1:
    # CLEARLY IRRELEVANT
    # -----------------------------------

    if (
        best_semantic_score
        < LOW_SEMANTIC_THRESHOLD
    ):

        print(
            "[GRADER] Semantic score too low."
        )

        print(
            "[GRADER] Final decision: NO"
        )

        return False

    # -----------------------------------
    # ZONE 2:
    # CLEARLY RELEVANT
    # -----------------------------------

    if (
        best_semantic_score
        >= HIGH_SEMANTIC_THRESHOLD
    ):

        print(
            "[GRADER] Strong semantic match."
        )

        print(
            "[GRADER] Final decision: YES"
        )

        return True

    # -----------------------------------
    # ZONE 3:
    # AMBIGUOUS
    # ASK QWEN
    # -----------------------------------

    print(
        "[GRADER] Ambiguous semantic score."
    )

    print(
        "[GRADER] Asking Qwen..."
    )

    context = build_context(
        results
    )

    if not context:

        print(
            "[GRADER] Empty context."
        )

        return False

    prompt = f"""
You are checking whether retrieved text is useful
for answering a question.

Use ONLY the retrieved context.
Do not use outside knowledge.
Do not answer the question.

Question:
{question}

Retrieved context:
{context}

If the context clearly discusses the same concept
and contains useful information for answering
the question, return YES.

Otherwise return NO.

Output only YES or NO.
""".strip()

    raw_decision = generate_short_text(
        prompt,
        max_new_tokens=3,
    )

    raw_decision = (
        raw_decision
        .strip()
        .upper()
    )

    print(
        f"[LLM GRADER] Raw decision: "
        f"{raw_decision}"
    )

    decision = (
        raw_decision.split()[0]
        if raw_decision
        else ""
    )

    if decision == "YES":

        print(
            "[GRADER] Final decision: YES"
        )

        return True

    print(
        "[GRADER] Final decision: NO"
    )

    return False
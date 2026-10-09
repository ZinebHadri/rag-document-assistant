from src.agentic.state import AgentState

from src.agentic.grader import (
    grade_relevance,
)

from src.agentic.rewriter import (
    rewrite_search_query,
)

from src.storage import (
    load_chunks,
)

from src.retrieval import (
    load_or_rebuild_index,
    retrieve_chunks,
)

from src.generation import (
    generate_answer as generate_rag_answer,
)


MAX_RETRIEVAL_ATTEMPTS = 2


def analyze_query(
    state: AgentState,
):
    """
    Prepare the query before retrieval.
    """

    question = state["question"]

    current_query = state.get(
        "current_query",
        "",
    )

    if not current_query:
        current_query = question

    print()
    print(
        f"[ANALYZE] Query: {current_query}"
    )

    return {
        "current_query": current_query,
    }


def retrieve_documents(
    state: AgentState,
):
    """
    Retrieve document chunks using
    FAISS + BM25 hybrid retrieval.
    """

    query = state["current_query"]

    attempt = (
        state["retrieval_attempts"]
        + 1
    )

    print()
    print(
        f"[RETRIEVE] Attempt {attempt}"
    )

    print(
        f"[RETRIEVE] Searching for: {query}"
    )

    chunks = load_chunks()

    if not chunks:

        print(
            "[RETRIEVE] Knowledge base is empty."
        )

        return {
            "retrieved_results": [],
            "retrieval_attempts": attempt,
        }

    index = load_or_rebuild_index(
        chunks
    )

    if index is None:

        print(
            "[RETRIEVE] FAISS index unavailable."
        )

        return {
            "retrieved_results": [],
            "retrieval_attempts": attempt,
        }

    results = retrieve_chunks(
        query,
        chunks,
        index,
        top_k=5,
    )

    print(
        f"[RETRIEVE] {len(results)} chunks retrieved."
    )

    if results:

        best = results[0]

        chunk = best.get(
            "chunk",
            {},
        )

        print(
            "[RETRIEVE] Best result:"
        )

        print(
            f"  Source: {chunk.get('source')}"
        )

        print(
            f"  Page: {chunk.get('page')}"
        )

        print(
            f"  Hybrid score: "
            f"{best.get('hybrid_score', 0.0):.4f}"
        )

    return {
        "retrieved_results": results,
        "retrieval_attempts": attempt,
    }


def grade_documents(
    state: AgentState,
):
    """
    Ask Qwen whether the retrieved
    context is truly relevant.
    """

    question = state["question"]

    results = state[
        "retrieved_results"
    ]

    print()
    print(
        "[GRADE] Asking Qwen to evaluate retrieval..."
    )

    is_relevant = grade_relevance(
        question=question,
        results=results,
    )

    print(
        f"[GRADE] Relevant: {is_relevant}"
    )

    return {
        "retrieval_is_relevant":
            is_relevant,
    }


def rewrite_query(
    state: AgentState,
):
    """
    Ask Qwen to rewrite a weak search query.
    """

    question = state["question"]

    current_query = state[
        "current_query"
    ]

    print()
    print(
        "[REWRITE] Asking Qwen "
        "to improve the search query..."
    )

    rewritten_query = (
        rewrite_search_query(
            question=question,
            current_query=current_query,
        )
    )

    print(
        f"[REWRITE] Old: {current_query}"
    )

    print(
        f"[REWRITE] New: {rewritten_query}"
    )

    return {
        "current_query":
            rewritten_query,
    }


def generate_answer_node(
    state: AgentState,
):
    """
    Generate only if the retrieved
    context was judged relevant.

    Otherwise return a safe fallback.
    """

    question = state["question"]

    results = state[
        "retrieved_results"
    ]

    is_relevant = state[
        "retrieval_is_relevant"
    ]

    if not is_relevant:

        print()
        print(
            "[GENERATE] No sufficiently "
            "relevant context."
        )

        return {
            "answer": (
                "I don't have enough relevant "
                "information in the current "
                "knowledge base to answer "
                "this question reliably."
            )
        }

    print()
    print(
        "[GENERATE] Generating grounded answer..."
    )

    answer = generate_rag_answer(
        question,
        results,
    )

    return {
        "answer": answer,
    }


def route_after_grading(
    state: AgentState,
):
    """
    Decide whether to generate,
    rewrite, or safely stop.
    """

    is_relevant = state[
        "retrieval_is_relevant"
    ]

    attempts = state[
        "retrieval_attempts"
    ]

    if is_relevant:

        print()
        print(
            "[ROUTER] Retrieval accepted "
            "-> GENERATE"
        )

        return "generate"

    if attempts >= MAX_RETRIEVAL_ATTEMPTS:

        print()
        print(
            "[ROUTER] Maximum attempts "
            "reached -> SAFE FALLBACK"
        )

        return "generate"

    print()
    print(
        "[ROUTER] Retrieval rejected "
        "-> REWRITE"
    )

    return "rewrite"
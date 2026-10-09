from src.agentic.nodes import (
    route_after_grading,
    generate_answer_node,
)


def test_route_to_generate_when_relevant():
    state = {
        "question": "What is PCA?",
        "current_query": "What is PCA?",
        "retrieved_results": [],
        "retrieval_attempts": 1,
        "retrieval_is_relevant": True,
        "answer": "",
    }

    result = route_after_grading(state)

    assert result == "generate"


def test_route_to_rewrite_when_not_relevant():
    state = {
        "question": "Unknown topic",
        "current_query": "Unknown topic",
        "retrieved_results": [],
        "retrieval_attempts": 1,
        "retrieval_is_relevant": False,
        "answer": "",
    }

    result = route_after_grading(state)

    assert result == "rewrite"


def test_route_to_generate_after_max_attempts():
    state = {
        "question": "Unknown topic",
        "current_query": "Unknown topic",
        "retrieved_results": [],
        "retrieval_attempts": 2,
        "retrieval_is_relevant": False,
        "answer": "",
    }

    result = route_after_grading(state)

    assert result == "generate"


def test_safe_fallback_when_context_not_relevant():
    state = {
        "question": "Explain quantum teleportation",
        "current_query": "Quantum teleportation",
        "retrieved_results": [],
        "retrieval_attempts": 2,
        "retrieval_is_relevant": False,
        "answer": "",
    }

    result = generate_answer_node(state)

    assert (
        "don't have enough relevant information"
        in result["answer"]
    )


def test_fallback_answer_is_not_empty():
    state = {
        "question": "Unknown question",
        "current_query": "Unknown question",
        "retrieved_results": [],
        "retrieval_attempts": 2,
        "retrieval_is_relevant": False,
        "answer": "",
    }

    result = generate_answer_node(state)

    assert result["answer"]
    assert isinstance(
        result["answer"],
        str,
    )
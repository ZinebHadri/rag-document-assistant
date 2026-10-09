from typing import TypedDict, List, Dict, Any


class AgentState(TypedDict):
    """
    Shared state used by the Agentic RAG workflow.

    Every LangGraph node can read values
    from this state and return updated values.
    """

    question: str

    current_query: str

    retrieved_results: List[
        Dict[str, Any]
    ]

    retrieval_attempts: int

    retrieval_is_relevant: bool

    answer: str


def create_initial_state(
    question: str
) -> AgentState:
    """
    Create the initial state for one
    Agentic RAG execution.
    """

    return {
        "question": question,
        "current_query": question,
        "retrieved_results": [],
        "retrieval_attempts": 0,
        "retrieval_is_relevant": False,
        "answer": ""
    }
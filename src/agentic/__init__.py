from src.agentic.state import (
    AgentState,
    create_initial_state,
)


def build_agentic_rag_graph():
    """
    Lazy import of the LangGraph workflow.

    Importing graph only when this function is called
    avoids package initialization problems.
    """

    from src.agentic.graph import (
        build_agentic_rag_graph as _build_graph,
    )

    return _build_graph()


def run_agentic_rag(
    question: str,
):
    """
    Run the Agentic RAG workflow.

    The graph module is imported only when needed.
    """

    from src.agentic.graph import (
        run_agentic_rag as _run_agentic_rag,
    )

    return _run_agentic_rag(
        question
    )


__all__ = [
    "AgentState",
    "create_initial_state",
    "build_agentic_rag_graph",
    "run_agentic_rag",
]
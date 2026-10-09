from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from src.agentic.state import (
    AgentState,
    create_initial_state,
)

from src.agentic.nodes import (
    analyze_query,
    retrieve_documents,
    grade_documents,
    rewrite_query,
    generate_answer_node,
    route_after_grading,
)


def build_agentic_rag_graph():
    """
    Build and compile the Agentic RAG workflow.

    Workflow:

    START
      |
      v
    analyze
      |
      v
    retrieve
      |
      v
    grade
      |
      v
    relevant?
      |
      +---- yes ----> generate ----> END
      |
      +---- no -----> rewrite
                        |
                        v
                     retrieve
    """

    workflow = StateGraph(
        AgentState
    )

    # -----------------------------
    # NODES
    # -----------------------------

    workflow.add_node(
        "analyze",
        analyze_query,
    )

    workflow.add_node(
        "retrieve",
        retrieve_documents,
    )

    workflow.add_node(
        "grade",
        grade_documents,
    )

    workflow.add_node(
        "rewrite",
        rewrite_query,
    )

    workflow.add_node(
        "generate",
        generate_answer_node,
    )

    # -----------------------------
    # NORMAL EDGES
    # -----------------------------

    workflow.add_edge(
        START,
        "analyze",
    )

    workflow.add_edge(
        "analyze",
        "retrieve",
    )

    workflow.add_edge(
        "retrieve",
        "grade",
    )

    workflow.add_edge(
        "rewrite",
        "retrieve",
    )

    workflow.add_edge(
        "generate",
        END,
    )

    # -----------------------------
    # CONDITIONAL EDGE
    # -----------------------------

    workflow.add_conditional_edges(
        "grade",
        route_after_grading,
        {
            "generate": "generate",
            "rewrite": "rewrite",
        },
    )

    # Compile the workflow into
    # an executable LangGraph graph
    return workflow.compile()


def run_agentic_rag(
    question: str,
):
    """
    Run one complete Agentic RAG request.

    Steps:
    1. Create the graph.
    2. Create the initial state.
    3. Execute the graph.
    4. Return the final state.
    """

    graph = (
        build_agentic_rag_graph()
    )

    initial_state = (
        create_initial_state(
            question
        )
    )

    final_state = (
        graph.invoke(
            initial_state
        )
    )

    return final_state
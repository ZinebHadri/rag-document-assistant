from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
)

from pydantic import BaseModel


from src.ingestion import (
    process_pdf_bytes,
)

from src.storage import (
    load_chunks,
    save_chunks,
    get_document_names,
    remove_document,
    clear_chunks,
    FAISS_PATH,
)

from src.retrieval import (
    create_faiss_index,
)

from src.agentic import (
    run_agentic_rag,
)



app = FastAPI(
    title="Agentic RAG Document Assistant API",
    description=(
        "REST API for uploading PDF documents "
        "and asking questions using an Agentic "
        "Retrieval-Augmented Generation pipeline "
        "with hybrid retrieval, relevance grading, "
        "query rewriting and safe fallback."
    ),
    version="2.0.0",
)



class QuestionRequest(
    BaseModel
):
    question: str




@app.get("/")
def root():

    return {
        "message":
            "Agentic RAG Document Assistant API is running",
        "version":
            "2.0.0",
    }



@app.get("/health")
def health():

    chunks = (
        load_chunks()
    )

    documents = (
        get_document_names(
            chunks
        )
    )

    return {
        "status":
            "ok",

        "documents":
            len(documents),

        "chunks":
            len(chunks),

        "rag_mode":
            "agentic",
    }


# ============================================================
# GET DOCUMENTS
# ============================================================

@app.get("/documents")
def documents():

    chunks = (
        load_chunks()
    )

    document_names = (
        get_document_names(
            chunks
        )
    )

    result = []

    for document_name in document_names:

        chunk_count = sum(
            1
            for chunk in chunks
            if (
                chunk["source"]
                == document_name
            )
        )

        result.append(
            {
                "name":
                    document_name,

                "chunks":
                    chunk_count,
            }
        )

    return {
        "count":
            len(result),

        "documents":
            result,
    }


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
):


    if not (
        file.filename
        .lower()
        .endswith(".pdf")
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Only PDF files "
                "are supported."
            ),
        )


    chunks = (
        load_chunks()
    )


    existing_document = any(
        chunk["source"]
        == file.filename
        for chunk in chunks
    )

    if existing_document:

        raise HTTPException(
            status_code=409,
            detail=(
                "A document with this "
                "filename already exists."
            ),
        )


    file_bytes = (
        await file.read()
    )

    try:

        new_chunks = (
            process_pdf_bytes(
                file.filename,
                file_bytes,
            )
        )

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Could not process PDF: "
                f"{str(error)}"
            ),
        )

    if len(new_chunks) == 0:

        raise HTTPException(
            status_code=400,
            detail=(
                "No readable text "
                "was found in the PDF."
            ),
        )


    chunks.extend(
        new_chunks
    )

    save_chunks(
        chunks
    )


    create_faiss_index(
        chunks
    )

    return {
        "message":
            "Document uploaded successfully.",

        "document":
            file.filename,

        "new_chunks":
            len(new_chunks),

        "total_chunks":
            len(chunks),
    }



@app.delete(
    "/documents/{document_name}"
)
def delete_document(
    document_name: str,
):

    chunks = (
        load_chunks()
    )

    document_names = (
        get_document_names(
            chunks
        )
    )

    if (
        document_name
        not in document_names
    ):

        raise HTTPException(
            status_code=404,
            detail=(
                "Document not found."
            ),
        )

    remaining_chunks = (
        remove_document(
            chunks,
            document_name,
        )
    )

    if remaining_chunks:

        create_faiss_index(
            remaining_chunks
        )

    elif FAISS_PATH.exists():

        FAISS_PATH.unlink()

    return {
        "message":
            "Document deleted successfully.",

        "document":
            document_name,

        "remaining_chunks":
            len(
                remaining_chunks
            ),
    }



@app.delete("/documents")
def delete_all_documents():

    clear_chunks()

    if FAISS_PATH.exists():

        FAISS_PATH.unlink()

    return {
        "message":
            "Knowledge base cleared successfully."
    }


# ============================================================
# HELPER: BUILD SOURCES
# ============================================================

def build_sources(
    retrieved_results,
):
    """
    Convert retrieved chunks into a clean
    source list for the API response.

    Duplicate document/page combinations
    are removed.
    """

    sources = []

    seen_sources = set()

    for result in retrieved_results:

        chunk = result.get(
            "chunk",
            {},
        )

        document = chunk.get(
            "source",
            "unknown",
        )

        page = chunk.get(
            "page",
            "unknown",
        )

        key = (
            document,
            page,
        )

        if key in seen_sources:
            continue

        seen_sources.add(
            key
        )

        semantic_score = (
            result.get(
                "semantic_score",
                0.0,
            )
        )

        keyword_score = (
            result.get(
                "keyword_score",
                0.0,
            )
        )

        hybrid_score = (
            result.get(
                "hybrid_score",
                0.0,
            )
        )

        sources.append(
            {
                "document":
                    document,

                "page":
                    page,

                "semantic_score":
                    round(
                        float(
                            semantic_score
                        ),
                        3,
                    ),

                "keyword_score":
                    round(
                        float(
                            keyword_score
                        ),
                        3,
                    ),

                "hybrid_score":
                    round(
                        float(
                            hybrid_score
                        ),
                        3,
                    ),
            }
        )

    return sources



@app.post("/ask")
def ask_question(
    request: QuestionRequest,
):

    question = (
        request.question
        .strip()
    )


    if not question:

        raise HTTPException(
            status_code=400,
            detail=(
                "Question cannot "
                "be empty."
            ),
        )


    chunks = (
        load_chunks()
    )

    if len(chunks) == 0:

        raise HTTPException(
            status_code=400,
            detail=(
                "Knowledge base is empty. "
                "Upload documents first."
            ),
        )


    try:

        final_state = (
            run_agentic_rag(
                question
            )
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Agentic RAG workflow failed: "
                f"{str(error)}"
            ),
        )


    answer = final_state.get(
        "answer",
        "",
    )

    current_query = (
        final_state.get(
            "current_query",
            question,
        )
    )

    retrieval_attempts = (
        final_state.get(
            "retrieval_attempts",
            0,
        )
    )

    retrieval_is_relevant = (
        final_state.get(
            "retrieval_is_relevant",
            False,
        )
    )

    retrieved_results = (
        final_state.get(
            "retrieved_results",
            [],
        )
    )

    # --------------------------------------------------------
    # SOURCES
    # --------------------------------------------------------

    sources = (
        build_sources(
            retrieved_results
        )
        if retrieval_is_relevant
        else []
    )

    best_semantic_score = None

    if retrieved_results:

        semantic_scores = [
            float(
                result.get(
                    "semantic_score",
                    0.0,
                )
            )
            for result
            in retrieved_results
        ]

        if semantic_scores:

            best_semantic_score = (
                round(
                    max(
                        semantic_scores
                    ),
                    3,
                )
            )

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {
        "question":
            question,

        "answer":
            answer,

        "agent":
            {
                "mode":
                    "agentic_rag",

                "final_query":
                    current_query,

                "retrieval_attempts":
                    retrieval_attempts,

                "retrieval_relevant":
                    retrieval_is_relevant,
            },

        "best_similarity_score":
            best_semantic_score,

        "sources":
            sources,
    }
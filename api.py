from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File
)

from pydantic import BaseModel


from src.ingestion import (
    process_pdf_bytes
)

from src.storage import (
    load_chunks,
    save_chunks,
    get_document_names,
    remove_document,
    clear_chunks,
    FAISS_PATH
)

from src.retrieval import (
    create_faiss_index,
    load_or_rebuild_index,
    retrieve_chunks
)

from src.generation import (
    generate_answer
)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="RAG Document Assistant API",
    description=(
        "REST API for uploading PDF documents "
        "and asking questions using a "
        "Retrieval-Augmented Generation pipeline."
    ),
    version="1.0.0"
)


# ============================================================
# REQUEST MODELS
# ============================================================

class QuestionRequest(
    BaseModel
):

    question: str


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "message":
            "RAG Document Assistant API is running"
    }


# ============================================================
# HEALTH
# ============================================================

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
            len(chunks)
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

    for document_name in (
        document_names
    ):

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
                    chunk_count
            }
        )

    return {
        "count":
            len(result),

        "documents":
            result
    }


# ============================================================
# UPLOAD PDF
# ============================================================

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # VALIDATE FILE
    # --------------------------------------------------------

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
            )
        )


    # --------------------------------------------------------
    # LOAD EXISTING CHUNKS
    # --------------------------------------------------------

    chunks = (
        load_chunks()
    )


    # --------------------------------------------------------
    # CHECK DUPLICATE
    # --------------------------------------------------------

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
            )
        )


    # --------------------------------------------------------
    # READ PDF
    # --------------------------------------------------------

    file_bytes = (
        await file.read()
    )


    try:

        new_chunks = (
            process_pdf_bytes(
                file.filename,
                file_bytes
            )
        )

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Could not process PDF: "
                f"{str(error)}"
            )
        )


    if len(new_chunks) == 0:

        raise HTTPException(
            status_code=400,
            detail=(
                "No readable text "
                "was found in the PDF."
            )
        )


    # --------------------------------------------------------
    # SAVE CHUNKS
    # --------------------------------------------------------

    chunks.extend(
        new_chunks
    )

    save_chunks(
        chunks
    )


    # --------------------------------------------------------
    # REBUILD FAISS
    # --------------------------------------------------------

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
            len(chunks)
    }


# ============================================================
# DELETE ONE DOCUMENT
# ============================================================

@app.delete(
    "/documents/{document_name}"
)
def delete_document(
    document_name: str
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
            )
        )


    remaining_chunks = (
        remove_document(
            chunks,
            document_name
        )
    )


    create_faiss_index(
        remaining_chunks
    )


    return {
        "message":
            "Document deleted successfully.",

        "document":
            document_name,

        "remaining_chunks":
            len(
                remaining_chunks
            )
    }


# ============================================================
# CLEAR KNOWLEDGE BASE
# ============================================================

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
# ASK RAG
# ============================================================

@app.post("/ask")
def ask_question(
    request: QuestionRequest
):

    question = (
        request.question
        .strip()
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not question:

        raise HTTPException(
            status_code=400,
            detail=(
                "Question cannot "
                "be empty."
            )
        )


    # --------------------------------------------------------
    # LOAD KNOWLEDGE BASE
    # --------------------------------------------------------

    chunks = (
        load_chunks()
    )


    if len(chunks) == 0:

        raise HTTPException(
            status_code=400,
            detail=(
                "Knowledge base is empty. "
                "Upload documents first."
            )
        )


    index = (
        load_or_rebuild_index(
            chunks
        )
    )


    if index is None:

        raise HTTPException(
            status_code=500,
            detail=(
                "Vector index could "
                "not be loaded."
            )
        )


    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    retrieved_results = (
        retrieve_chunks(
            question,
            chunks,
            index,
            top_k=10
        )
    )


    if len(
        retrieved_results
    ) == 0:

        return {
            "question":
                question,

            "answer":
                (
                    "I cannot find enough "
                    "information in the "
                    "provided documents."
                ),

            "sources":
                []
        }


    # --------------------------------------------------------
    # CONFIDENCE THRESHOLD
    # --------------------------------------------------------

    similarity_threshold = (
        0.35
    )

    best_score = (
        retrieved_results[0]
        ["semantic_score"]
    )


    if (
        best_score
        <
        similarity_threshold
    ):

        return {
            "question":
                question,

            "answer":
                (
                    "I cannot find enough "
                    "information in the "
                    "provided documents."
                ),

            "best_similarity_score":
                round(
                    best_score,
                    3
                ),

            "sources":
                []
        }


    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------

    answer = (
        generate_answer(
            question,
            retrieved_results
        )
    )


    # --------------------------------------------------------
    # SOURCES
    # --------------------------------------------------------

    sources = []

    seen_sources = set()


    for result in (
        retrieved_results
    ):

        chunk = (
            result["chunk"]
        )

        key = (
            chunk["source"],
            chunk["page"]
        )


        if key in seen_sources:
            continue


        seen_sources.add(
            key
        )


        sources.append(
            {
                "document":
                    chunk["source"],

                "page":
                    chunk["page"],

                "semantic_score":
                    round(
                        result[
                            "semantic_score"
                        ],
                        3
                    ),

                "keyword_score":
                    result[
                        "keyword_score"
                    ],

                "hybrid_score":
                    round(
                        result[
                            "hybrid_score"
                        ],
                        3
                    )
            }
        )


    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {
        "question":
            question,

        "answer":
            answer,

        "best_similarity_score":
            round(
                best_score,
                3
            ),

        "sources":
            sources
    }
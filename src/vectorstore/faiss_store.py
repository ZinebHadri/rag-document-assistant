import faiss

from src.embeddings import (
    embed_texts
)

from src.storage import (
    FAISS_PATH,
    ensure_storage_folder
)


def create_faiss_index(chunks):
    """
    Build a FAISS index from document chunks.

    Embeddings are normalized, so
    IndexFlatIP behaves like cosine similarity.
    """

    if not chunks:
        return None

    texts = [
        chunk.get(
            "text",
            ""
        )
        for chunk in chunks
    ]

    embeddings = embed_texts(
        texts
    )

    if embeddings.size == 0:
        return None

    dimension = (
        embeddings.shape[1]
    )

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    ensure_storage_folder()

    faiss.write_index(
        index,
        str(FAISS_PATH)
    )

    return index


def load_faiss_index():
    """
    Load a persisted FAISS index.
    """

    if not FAISS_PATH.exists():
        return None

    try:

        return faiss.read_index(
            str(FAISS_PATH)
        )

    except Exception:

        return None


def load_or_rebuild_index(chunks):
    """
    Load the FAISS index when possible.

    If the index is missing or its number
    of vectors no longer matches the number
    of chunks, rebuild it.
    """

    if not chunks:
        return None

    index = load_faiss_index()

    if (
        index is not None
        and index.ntotal
        == len(chunks)
    ):

        return index

    return create_faiss_index(
        chunks
    )
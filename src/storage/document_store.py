import json
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

STORAGE_FOLDER = (
    PROJECT_ROOT
    / "storage"
)

CHUNKS_PATH = (
    STORAGE_FOLDER
    / "chunks.json"
)

FAISS_PATH = (
    STORAGE_FOLDER
    / "faiss_index.bin"
)


def ensure_storage_folder():
    """
    Ensure that the persistent storage
    directory exists.
    """

    STORAGE_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )


def load_chunks():
    """
    Load document chunks from disk.
    """

    ensure_storage_folder()

    if not CHUNKS_PATH.exists():
        return []

    try:

        with open(
            CHUNKS_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(
                file
            )

        if not isinstance(
            data,
            list
        ):
            return []

        return data

    except (
        json.JSONDecodeError,
        OSError
    ):

        return []


def save_chunks(chunks):
    """
    Persist chunks to JSON.
    """

    ensure_storage_folder()

    with open(
        CHUNKS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )


def get_document_names(chunks):
    """
    Return unique document names
    while preserving insertion order.
    """

    documents = []

    seen = set()

    for chunk in chunks:

        source = chunk.get(
            "source"
        )

        if not source:
            continue

        if source in seen:
            continue

        seen.add(
            source
        )

        documents.append(
            source
        )

    return documents


def remove_document(
    chunks,
    document_name
):
    """
    Remove every chunk belonging
    to one document.
    """

    filtered_chunks = [
        chunk
        for chunk in chunks
        if chunk.get("source")
        != document_name
    ]

    save_chunks(
        filtered_chunks
    )

    return filtered_chunks


def clear_chunks():
    """
    Remove all persisted chunks.
    """

    save_chunks(
        []
    )


def clear_faiss_index():
    """
    Delete the persisted FAISS index.
    """

    if FAISS_PATH.exists():

        FAISS_PATH.unlink()


def clear_storage():
    """
    Clear both chunk storage
    and FAISS index.
    """

    clear_chunks()

    clear_faiss_index()
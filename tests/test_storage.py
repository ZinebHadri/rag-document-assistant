from src.storage import get_document_names


def test_get_document_names():

    chunks = [
        {
            "source": "pca.pdf",
            "page": 1,
            "text": "PCA text"
        },
        {
            "source": "retrieval.pdf",
            "page": 1,
            "text": "Retrieval text"
        },
        {
            "source": "pca.pdf",
            "page": 2,
            "text": "More PCA text"
        }
    ]

    documents = get_document_names(
        chunks
    )

    assert documents == [
        "pca.pdf",
        "retrieval.pdf"
    ]


def test_get_document_names_empty():

    documents = get_document_names(
        []
    )

    assert documents == []
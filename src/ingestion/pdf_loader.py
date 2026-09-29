from io import BytesIO

from pypdf import PdfReader

from src.chunking.sentence_chunker import (
    split_text_by_sentences
)


def extract_pdf_pages(
    file_bytes
):
    """
    Extract text page by page from PDF bytes.

    Returns
    -------
    list[dict]
        Each dictionary contains:
        - page
        - text
    """

    pdf_stream = BytesIO(
        file_bytes
    )

    reader = PdfReader(
        pdf_stream
    )

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if not text:
            continue

        text = text.strip()

        if not text:
            continue

        pages.append(
            {
                "page": page_number,
                "text": text
            }
        )

    return pages


def process_pdf_bytes(
    file_name,
    file_bytes,
    max_chars=500
):
    """
    Convert a PDF into RAG chunks.

    Each chunk keeps its document source
    and page number as metadata.
    """

    pages = extract_pdf_pages(
        file_bytes
    )

    chunks = []

    for page_data in pages:

        page_number = page_data[
            "page"
        ]

        page_text = page_data[
            "text"
        ]

        page_chunks = (
            split_text_by_sentences(
                page_text,
                max_chars=max_chars
            )
        )

        for chunk_text in page_chunks:

            chunks.append(
                {
                    "source": file_name,
                    "page": page_number,
                    "text": chunk_text
                }
            )

    return chunks


def process_uploaded_pdf(
    uploaded_file,
    max_chars=500
):
    """
    Process a Streamlit UploadedFile.
    """

    file_bytes = (
        uploaded_file.getvalue()
    )

    return process_pdf_bytes(
        file_name=uploaded_file.name,
        file_bytes=file_bytes,
        max_chars=max_chars
    )
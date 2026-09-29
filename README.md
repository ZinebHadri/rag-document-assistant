# RAG Document Assistant

A modular Retrieval-Augmented Generation (RAG) application for asking questions about PDF documents.

The project combines semantic retrieval with FAISS, lexical retrieval with BM25, a FastAPI backend, a Streamlit interface, and a local Qwen language model.

## What the project does

The application allows users to:

- Upload PDF documents
- Extract and chunk their content
- Generate semantic embeddings
- Store embeddings in a FAISS vector index
- Retrieve relevant passages using FAISS and BM25
- Combine semantic and lexical retrieval
- Generate grounded answers using a local LLM
- Display the document sources and page numbers used
- Manage the indexed document collection through a web interface

## System design

The application follows a modular RAG architecture:

```text
PDF Documents
      ↓
PDF Ingestion
      ↓
Sentence-based Chunking
      ↓
SentenceTransformer Embeddings
      ↓
FAISS Vector Index
      ↓
┌──────────────────────────────┐
│      Hybrid Retrieval        │
│                              │
│  FAISS Semantic Search       │
│            +                 │
│  BM25 Lexical Search         │
└──────────────┬───────────────┘
               ↓
        Retrieved Context
               ↓
       Qwen2.5-0.5B
               ↓
        Grounded Answer
               ↓
      Sources + Page Numbers
```

The backend is exposed through FastAPI and consumed by a Streamlit frontend.

## Project architecture

```text
rag-document-assistant/
│
├── app.py
├── api.py
├── evaluation.py
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── src/
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── pdf_loader.py
│   │
│   ├── chunking/
│   │   ├── __init__.py
│   │   └── sentence_chunker.py
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embedder.py
│   │
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   └── faiss_store.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── semantic.py
│   │   ├── bm25.py
│   │   └── hybrid.py
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   └── llm.py
│   │
│   └── storage/
│       ├── __init__.py
│       └── document_store.py
│
├── tests/
│
├── data/
│
├── storage/
│
└── .github/
    └── workflows/
        └── ci.yml
```

## Main components

### PDF ingestion

PDF files are processed with PyPDF.

Text is extracted page by page so that retrieved chunks can preserve their original page number.

### Chunking

Extracted text is split into sentence-based chunks using NLTK.

The current chunk size is approximately 500 characters.

### Embeddings

Document chunks are encoded using:

```text
all-MiniLM-L6-v2
```

Embeddings are normalized before being stored.

### FAISS

FAISS provides semantic vector search.

The project uses:

```text
IndexFlatIP
```

with normalized vectors, which allows inner product similarity to behave similarly to cosine similarity.

### BM25

BM25 provides lexical retrieval.

Unlike semantic retrieval, BM25 is useful when exact words, technical expressions, acronyms, or uncommon terms appear in the query.

### Hybrid retrieval

The application combines semantic and lexical retrieval.

The current retrieval pipeline uses:

```text
70% semantic score
30% BM25 score
```

after score normalization.

This allows the system to benefit from both semantic similarity and keyword matching.

### Generation

The retrieved context is passed to:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

The model runs locally using the Hugging Face Transformers library.

The prompt instructs the model to answer only from the retrieved document context.

### FastAPI

FastAPI exposes the RAG pipeline through an API.

Main endpoints include:

```text
GET     /health
GET     /documents
POST    /ask
POST    /upload
DELETE  /documents/{document_name}
DELETE  /documents
```

Interactive API documentation is available through Swagger.

### Streamlit

Streamlit provides the user interface.

The application includes:

- Home dashboard
- Knowledge Base management
- Document upload
- RAG assistant
- Source visualization
- Architecture explanation

## Technologies

- Python
- FastAPI
- Streamlit
- SentenceTransformers
- FAISS
- BM25
- Qwen2.5
- Hugging Face Transformers
- PyTorch
- NLTK
- PyPDF
- NumPy
- Pytest
- Docker
- Docker Compose
- GitHub Actions

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/rag-document-assistant.git
```

Move into the project:

```bash
cd rag-document-assistant
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running locally

Start FastAPI:

```bash
python -m uvicorn api:app --reload
```

FastAPI documentation:

```text
http://localhost:8000/docs
```

In another terminal, start Streamlit:

```bash
python -m streamlit run app.py
```

Streamlit interface:

```text
http://localhost:8501
```

## Run with Docker

The entire application can also run inside Docker.

Build and start the application:

```bash
docker compose up --build
```

Streamlit:

```text
http://localhost:8501
```

FastAPI documentation:

```text
http://localhost:8000/docs
```

Stop the containers:

```bash
docker compose down
```

## Retrieval evaluation

The retrieval component is evaluated using document-specific questions.

The evaluation compares:

- FAISS semantic retrieval
- BM25 lexical retrieval
- Hybrid retrieval

The main metrics are:

- **Hit@K**: whether a relevant document appears in the top K results
- **Recall@K**: whether relevant sources are retrieved
- **MRR**: how highly the first relevant result is ranked
- **Retrieval time**: average search time per query

## Retrieval evaluation results

The retrieval pipeline was evaluated on 12 questions covering direct questions, paraphrases, and technical concepts.

| Method | Hit@1 | Hit@3 | Hit@5 | MRR | Avg. retrieval time |
|---|---:|---:|---:|---:|---:|
| FAISS | 0.833 | 1.000 | 1.000 | 0.917 | 0.0229s |
| BM25 | 0.833 | 0.917 | 1.000 | 0.892 | 0.0022s |
| Hybrid | 0.917 | 1.000 | 1.000 | 0.958 | 0.0246s |

On this evaluation set, hybrid retrieval achieved the strongest ranking quality.

It improved both Hit@1 and MRR compared with FAISS-only and BM25-only retrieval.

BM25 was the fastest retrieval method, while hybrid retrieval provided a stronger balance between semantic and lexical search.

The evaluation dataset is still relatively small, so these results should be interpreted as prototype validation rather than a general benchmark.

## Testing

Run the test suite with:

```bash
python -m pytest -v
```

The project includes automated tests for core RAG components.

GitHub Actions is used to run tests automatically when changes are pushed to the repository.

## Current limitations

- The evaluation dataset is still small
- The local Qwen model is lightweight and can be slow on CPU
- PDF ingestion currently focuses mainly on text-based PDFs
- OCR is not currently implemented
- BM25 is currently rebuilt when retrieval is performed
- Hybrid retrieval weights are currently heuristic
- The system is designed as a prototype rather than a large-scale production deployment

## Future improvements

Possible future improvements include:

- Larger retrieval evaluation dataset
- Chunk-level relevance evaluation
- Reranking models
- OCR support for scanned PDFs
- Improved query expansion
- Cached BM25 indexing
- More advanced retrieval fusion
- Larger or API-based language models
- Authentication
- Cloud deployment
- Monitoring and logging

## Goal

This project was built to understand and implement a complete Retrieval-Augmented Generation pipeline from document ingestion to retrieval, generation, evaluation, API integration, frontend development, testing, CI, and containerization.
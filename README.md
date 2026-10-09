# Agentic RAG Document Assistant

A modular Retrieval-Augmented Generation (RAG) application for asking questions about PDF documents.

The project combines semantic retrieval with FAISS, lexical retrieval with BM25, a FastAPI backend, a Streamlit interface, a local Qwen language model, and a LangGraph-based agentic workflow.

Unlike a traditional RAG pipeline that immediately generates an answer after retrieval, this system evaluates retrieval quality, rewrites weak queries, retries retrieval, and safely refuses to answer when the knowledge base does not contain enough relevant information.

## What the project does

The application allows users to:

- Upload PDF documents
- Extract and chunk their content
- Generate semantic embeddings
- Store embeddings in a FAISS vector index
- Retrieve relevant passages using FAISS and BM25
- Combine semantic and lexical retrieval
- Evaluate whether retrieved context is relevant
- Automatically rewrite weak retrieval queries
- Retry retrieval when necessary
- Generate grounded answers using a local LLM
- Avoid unsupported answers through a safe fallback mechanism
- Display document sources and page numbers
- Display an agent trace showing retrieval attempts and decisions
- Manage the indexed document collection through a web interface

## System design

The application follows an Agentic RAG architecture:

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
│       Hybrid Retrieval       │
│                              │
│   FAISS Semantic Search      │
│            +                 │
│   BM25 Lexical Search        │
└──────────────┬───────────────┘
               ↓
        Retrieved Context
               ↓
        Relevance Grader
               ↓
          Is it relevant?
           /          \
         YES           NO
          ↓             ↓
     Generation     Query Rewrite
          ↓             ↓
        Answer      Retrieve Again
                        ↓
                  Relevance Grader
                        ↓
                 Relevant?
                  /       \
                YES        NO
                 ↓          ↓
            Generation   Safe Fallback
                 ↓
               Answer
```

The workflow is orchestrated with LangGraph.

The backend is exposed through FastAPI and consumed by a Streamlit frontend.

## Traditional RAG vs Agentic RAG

A traditional RAG pipeline generally follows:

```text
Question
   ↓
Retrieve
   ↓
Generate
```

This project extends that architecture:

```text
Question
   ↓
Analyze Query
   ↓
Retrieve
   ↓
Evaluate Retrieval
   ↓
Relevant?
  /       \
YES        NO
 ↓          ↓
Generate   Rewrite Query
              ↓
           Retrieve Again
              ↓
           Evaluate
              ↓
       Generate or Fallback
```

This additional decision layer reduces the risk of generating answers from irrelevant retrieved context.

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
│   ├── agentic/
│   │   ├── __init__.py
│   │   ├── state.py
│   │   ├── nodes.py
│   │   ├── graph.py
│   │   ├── grader.py
│   │   └── rewriter.py
│   │
│   └── storage/
│       ├── __init__.py
│       └── document_store.py
│
├── tests/
│   └── test_agentic.py
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

Text is extracted page by page so that retrieved chunks preserve their original document and page number.

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

with normalized vectors, allowing inner product similarity to behave similarly to cosine similarity.

### BM25

BM25 provides lexical retrieval.

Unlike semantic retrieval, BM25 is particularly useful when exact words, technical expressions, acronyms, or uncommon terms appear in the query.

### Hybrid retrieval

The application combines semantic and lexical retrieval.

The current retrieval pipeline uses:

```text
70% semantic score
30% BM25 score
```

after score normalization.

This allows the system to benefit from both semantic similarity and keyword matching.

## Agentic RAG layer

The Agentic RAG workflow is orchestrated with LangGraph.

Instead of always performing:

```text
retrieve → generate
```

the system can perform:

```text
retrieve
→ evaluate
→ rewrite query
→ retrieve again
→ evaluate
→ generate or safely stop
```

### Agent state

LangGraph maintains a shared workflow state containing information such as:

```text
question
current_query
retrieved_results
retrieval_attempts
retrieval_is_relevant
answer
```

This state acts as the working memory of a single Agentic RAG execution.

### Agent nodes

The graph contains several nodes:

```text
analyze_query
retrieve_documents
grade_documents
rewrite_query
generate_answer
```

Each node performs one responsibility and updates the workflow state.

### Conditional routing

LangGraph uses conditional edges to determine what should happen after retrieval evaluation.

```text
Relevant context
→ Generate answer

Weak context
→ Rewrite query
→ Retrieve again
```

Retrieval attempts are limited to prevent infinite loops.

## Relevance grading

The project uses a multi-stage relevance guardrail.

A semantic similarity score is first evaluated.

The current thresholds are:

```text
score < 0.30
→ reject directly

score >= 0.55
→ accept directly

0.30 <= score < 0.55
→ ambiguous
→ ask Qwen relevance grader
```

This design was introduced because a small local LLM can sometimes be unreliable when used as the sole relevance judge.

The system therefore combines:

```text
Deterministic semantic similarity
+
LLM-based relevance evaluation
```

for ambiguous cases.

## Query rewriting

When retrieval is judged insufficient, the local Qwen model rewrites the search query.

For example:

```text
Original:
What is PCA?

Possible rewritten query:
PCA Principal Component Analysis
```

The rewritten query is then sent through hybrid retrieval again.

The system also cleans and limits rewritten queries to prevent the LLM from generating long explanatory paragraphs instead of search-oriented queries.

## Safe fallback

If relevant context cannot be found after the maximum number of retrieval attempts, the system does not ask the LLM to invent an answer.

Instead, it returns:

```text
I don't have enough relevant information in the current knowledge base to answer this question reliably.
```

This reduces unsupported answers for questions outside the indexed knowledge base.

For example:

```text
Question:
Explain quantum teleportation protocols
```

when quantum teleportation is not contained in the indexed documents:

```text
Attempt 1
→ semantic relevance too low
→ query rewrite

Attempt 2
→ semantic relevance still too low
→ safe fallback
```

## Generation

Relevant retrieved context is passed to:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

The model runs locally using Hugging Face Transformers.

The generation prompt instructs the model to answer only using retrieved document context.

Generation is performed locally without relying on an external LLM API.

## FastAPI

FastAPI exposes the application through a REST API.

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

The `/ask` endpoint now executes the complete Agentic RAG workflow.

An API response can include:

```json
{
  "question": "What is PCA?",
  "answer": "Principal component analysis...",
  "agent": {
    "mode": "agentic_rag",
    "final_query": "What is PCA?",
    "retrieval_attempts": 1,
    "retrieval_relevant": true
  },
  "best_similarity_score": 0.77,
  "sources": []
}
```

## Streamlit

Streamlit provides the user interface.

The application includes:

- Home dashboard
- Knowledge Base management
- PDF upload
- Agentic RAG assistant
- Source visualization
- Agent trace visualization
- Architecture explanation

For each assistant response, the interface can display:

```text
Retrieval attempts
Context relevance
Final retrieval query
Best semantic similarity
Sources
Semantic score
BM25 score
Hybrid score
```

This makes the agent workflow more transparent to the user.

## Technologies

- Python
- LangGraph
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
.\.venv\Scripts\Activate.ps1
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

The application can also run inside Docker.

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

## Agentic RAG validation

The Agentic RAG workflow was also tested with both in-domain and out-of-domain questions.

Example in-domain query:

```text
What is PCA?
```

Observed behavior:

```text
Hybrid retrieval
→ pca.pdf retrieved
→ semantic score ≈ 0.77
→ strong semantic match
→ generation
→ grounded PCA answer
```

Example out-of-domain query:

```text
Explain quantum teleportation protocols
```

Observed behavior:

```text
Attempt 1
→ semantic score ≈ 0.23
→ rejected

Query rewrite
→ second retrieval

Attempt 2
→ semantic score ≈ 0.21
→ rejected

→ safe fallback
```

This validates the core agentic routing behavior of the prototype.

## Testing

Run the full test suite with:

```bash
python -m pytest -v
```

The project includes automated tests for core RAG components and Agentic RAG routing.

Agentic tests cover behaviors including:

```text
Relevant context
→ generation route

Weak context
→ rewrite route

Maximum retrieval attempts
→ stop retry loop

Insufficient context
→ safe fallback
```

GitHub Actions can be used to run tests automatically when changes are pushed to the repository.

## Current limitations

- The retrieval evaluation dataset is still small
- Semantic thresholds are currently heuristic
- The local Qwen2.5-0.5B model is intentionally lightweight
- Small language models can be imperfect relevance judges
- The query rewriter is lightweight
- PDF ingestion currently focuses mainly on text-based PDFs
- OCR is not currently implemented
- BM25 is currently rebuilt during retrieval
- Hybrid retrieval weights are heuristic
- Agentic evaluation is currently prototype-level
- The system is designed as a portfolio prototype rather than a large-scale production deployment

## Future improvements

Possible future improvements include:

- Larger retrieval evaluation dataset
- Dedicated Agentic RAG evaluation dataset
- Chunk-level relevance labels
- Cross-encoder reranking
- Learned or tuned relevance thresholds
- Better query rewriting models
- Structured LLM outputs
- Persistent BM25 index
- More advanced retrieval fusion
- OCR support for scanned PDFs
- Larger local models
- Optional API-based language models
- Conversation memory
- Observability and tracing
- Authentication
- Cloud deployment
- Monitoring and logging

## Key engineering decisions

Several design choices were made during development.

### Hybrid retrieval instead of semantic-only retrieval

FAISS captures semantic similarity while BM25 captures lexical overlap.

Combining both improved ranking quality on the prototype evaluation set.

### Local LLM

Qwen2.5-0.5B-Instruct runs locally rather than through an external inference API.

This keeps the complete RAG pipeline self-contained.

### Agentic retrieval validation

A traditional retrieval score alone does not guarantee that retrieved context can answer a question.

The Agentic RAG layer therefore evaluates retrieval quality before generation.

### Semantic guardrails around the LLM grader

Testing showed that a small language model could incorrectly classify clearly relevant or irrelevant retrieval results.

The final design therefore uses semantic thresholds for clear cases and reserves the LLM grader for ambiguous cases.

### Limited retry loop

The agent is allowed to rewrite and retry retrieval, but the number of attempts is limited.

This prevents infinite agent loops.

### Safe fallback

If relevant information cannot be found, the system returns a controlled fallback instead of generating an unsupported answer.

## Goal

This project was built to understand and implement a complete modern RAG system from document ingestion to retrieval, generation, evaluation, agentic orchestration, API integration, frontend development, testing, CI, and containerization.

It demonstrates the progression from:

```text
Basic semantic RAG
        ↓
Hybrid RAG
        ↓
Evaluated RAG
        ↓
Agentic RAG
```

The project focuses not only on generating answers, but also on making retrieval decisions more reliable, observable, and explainable.
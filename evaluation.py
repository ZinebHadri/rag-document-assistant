import re
import time

import numpy as np

from rank_bm25 import BM25Okapi

from src.storage import load_chunks
from src.retrieval import (
    get_embedding_model,
    load_or_rebuild_index,
    expand_question
)


# CONFIG

TOP_K_VALUES = [
    1,
    3,
    5
]

RRF_K = 60


# EVALUATION DATASET

evaluation_questions = [

    # PCA

    {
        "question": "What is PCA?",
        "relevant_sources": [
            "pca.pdf"
        ]
    },

    {
        "question": "What is Principal Component Analysis?",
        "relevant_sources": [
            "pca.pdf"
        ]
    },

    {
        "question": (
            "How can dimensionality be reduced "
            "while preserving important information?"
        ),
        "relevant_sources": [
            "pca.pdf"
        ]
    },

    {
        "question": (
            "What are principal components?"
        ),
        "relevant_sources": [
            "pca.pdf"
        ]
    },


    # RETRIEVAL

    {
        "question": "What is retrieval?",
        "relevant_sources": [
            "retrieval.pdf"
        ]
    },

    {
        "question": (
            "How does an information retrieval system "
            "find relevant documents?"
        ),
        "relevant_sources": [
            "retrieval.pdf"
        ]
    },

    {
        "question": (
            "What does relevance mean in information retrieval?"
        ),
        "relevant_sources": [
            "retrieval.pdf"
        ]
    },

    {
        "question": (
            "How are documents ranked for a query?"
        ),
        "relevant_sources": [
            "retrieval.pdf"
        ]
    },


    # UNSUPERVISED LEARNING

    {
        "question": "What is unsupervised learning?",
        "relevant_sources": [
            "unspervisedlearning.pdf"
        ]
    },

    {
        "question": (
            "How can we learn patterns from data "
            "without labels?"
        ),
        "relevant_sources": [
            "unspervisedlearning.pdf"
        ]
    },

    {
        "question": (
            "What is clustering in unsupervised learning?"
        ),
        "relevant_sources": [
            "unspervisedlearning.pdf"
        ]
    },

    {
        "question": (
            "How can similar observations be grouped "
            "without a target variable?"
        ),
        "relevant_sources": [
            "unspervisedlearning.pdf"
        ]
    }
]


# HELPERS

def normalize_filename(filename):

    return filename.lower().strip()


def tokenize(text):

    return re.findall(
        r"\b\w+\b",
        text.lower()
    )


def is_relevant(
    result,
    relevant_sources
):

    source = normalize_filename(
        result["chunk"]["source"]
    )

    relevant_sources = {
        normalize_filename(source_name)
        for source_name in relevant_sources
    }

    return source in relevant_sources


# METRICS

def hit_at_k(
    results,
    relevant_sources,
    k
):

    return int(
        any(
            is_relevant(
                result,
                relevant_sources
            )
            for result in results[:k]
        )
    )


def recall_at_k(
    results,
    relevant_sources,
    k
):

    relevant_set = {
        normalize_filename(source)
        for source in relevant_sources
    }

    retrieved_relevant = set()

    for result in results[:k]:

        source = normalize_filename(
            result["chunk"]["source"]
        )

        if source in relevant_set:

            retrieved_relevant.add(
                source
            )

    if not relevant_set:
        return 0.0

    return (
        len(retrieved_relevant)
        / len(relevant_set)
    )


def reciprocal_rank(
    results,
    relevant_sources
):

    for rank, result in enumerate(
        results,
        start=1
    ):

        if is_relevant(
            result,
            relevant_sources
        ):

            return 1 / rank

    return 0.0


# FAISS

def retrieve_faiss(
    question,
    chunks,
    index,
    top_k=10
):

    model = get_embedding_model()

    expanded_question = expand_question(
        question
    )

    question_embedding = model.encode(
        [expanded_question],
        normalize_embeddings=True
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype="float32"
    )

    scores, indices = index.search(
        question_embedding,
        min(
            top_k,
            index.ntotal
        )
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        idx = int(idx)

        if idx < 0:
            continue

        results.append(
            {
                "chunk":
                    chunks[idx],

                "score":
                    float(score),

                "index":
                    idx
            }
        )

    return results


# BM25

def build_bm25(chunks):

    tokenized_corpus = [
        tokenize(
            chunk["text"]
        )
        for chunk in chunks
    ]

    return BM25Okapi(
        tokenized_corpus
    )


def retrieve_bm25(
    question,
    chunks,
    bm25,
    top_k=10
):

    expanded_question = expand_question(
        question
    )

    query_tokens = tokenize(
        expanded_question
    )

    scores = bm25.get_scores(
        query_tokens
    )

    ranked_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for idx in ranked_indices:

        idx = int(idx)

        results.append(
            {
                "chunk":
                    chunks[idx],

                "score":
                    float(
                        scores[idx]
                    ),

                "index":
                    idx
            }
        )

    return results


# HYBRID

def retrieve_hybrid(
    question,
    chunks,
    index,
    bm25,
    top_k=10
):

    candidate_size = min(
        max(
            top_k * 5,
            50
        ),
        len(chunks)
    )

    semantic_results = retrieve_faiss(
        question,
        chunks,
        index,
        top_k=candidate_size
    )

    lexical_results = retrieve_bm25(
        question,
        chunks,
        bm25,
        top_k=candidate_size
    )

    rrf_scores = {}


    for rank, result in enumerate(
        semantic_results,
        start=1
    ):

        idx = result[
            "index"
        ]

        rrf_scores[idx] = (
            rrf_scores.get(
                idx,
                0.0
            )

            +

            1 / (
                RRF_K
                + rank
            )
        )


    for rank, result in enumerate(
        lexical_results,
        start=1
    ):

        idx = result[
            "index"
        ]

        rrf_scores[idx] = (
            rrf_scores.get(
                idx,
                0.0
            )

            +

            1 / (
                RRF_K
                + rank
            )
        )


    ranked_indices = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True
    )

    results = []

    seen_texts = set()


    for idx in ranked_indices:

        chunk = chunks[idx]

        normalized_text = " ".join(
            chunk["text"]
            .lower()
            .split()
        )

        if normalized_text in seen_texts:
            continue

        seen_texts.add(
            normalized_text
        )

        results.append(
            {
                "chunk":
                    chunk,

                "score":
                    rrf_scores[idx],

                "index":
                    idx
            }
        )

        if len(results) >= top_k:
            break


    return results


# EVALUATION

def evaluate_method(
    method_name,
    retrieval_function
):

    method_results = []

    print()
    print(
        "=" * 75
    )

    print(
        method_name
    )

    print(
        "=" * 75
    )


    for item in evaluation_questions:

        question = item[
            "question"
        ]

        relevant_sources = item[
            "relevant_sources"
        ]


        start_time = (
            time.perf_counter()
        )

        results = retrieval_function(
            question
        )

        retrieval_time = (
            time.perf_counter()
            - start_time
        )


        metrics = {

            "question":
                question,

            "time":
                retrieval_time,

            "mrr":
                reciprocal_rank(
                    results,
                    relevant_sources
                )
        }


        print()

        print(
            f"QUESTION: {question}"
        )

        print(
            f"EXPECTED: "
            f"{', '.join(relevant_sources)}"
        )


        if results:

            first_result = (
                results[0]["chunk"]
            )

            print(
                f"TOP RESULT: "
                f"{first_result['source']} "
                f"(page {first_result['page']})"
            )


        for k in TOP_K_VALUES:

            hit = hit_at_k(
                results,
                relevant_sources,
                k
            )

            recall = recall_at_k(
                results,
                relevant_sources,
                k
            )

            metrics[
                f"hit@{k}"
            ] = hit

            metrics[
                f"recall@{k}"
            ] = recall


        print(
            f"Hit@1: "
            f"{metrics['hit@1']}"
        )

        print(
            f"Hit@3: "
            f"{metrics['hit@3']}"
        )

        print(
            f"Hit@5: "
            f"{metrics['hit@5']}"
        )

        print(
            f"MRR: "
            f"{metrics['mrr']:.3f}"
        )

        print(
            f"Time: "
            f"{retrieval_time:.4f}s"
        )


        method_results.append(
            metrics
        )


    return method_results


# SUMMARY

def summarize_results(
    method_name,
    results
):

    count = len(
        results
    )

    summary = {
        "method":
            method_name
    }


    for k in TOP_K_VALUES:

        summary[
            f"hit@{k}"
        ] = sum(

            result[
                f"hit@{k}"
            ]

            for result
            in results

        ) / count


        summary[
            f"recall@{k}"
        ] = sum(

            result[
                f"recall@{k}"
            ]

            for result
            in results

        ) / count


    summary[
        "mrr"
    ] = sum(

        result[
            "mrr"
        ]

        for result
        in results

    ) / count


    summary[
        "time"
    ] = sum(

        result[
            "time"
        ]

        for result
        in results

    ) / count


    return summary


# LOAD KNOWLEDGE BASE

chunks = load_chunks()

if not chunks:

    print(
        "Knowledge base is empty."
    )

    raise SystemExit


index = load_or_rebuild_index(
    chunks
)

if index is None:

    print(
        "FAISS index could not be loaded."
    )

    raise SystemExit


# BUILD BM25

print()
print(
    "Building BM25 index..."
)

bm25 = build_bm25(
    chunks
)

print(
    f"{len(chunks)} chunks loaded."
)


# WARM UP

print(
    "Warming up embedding model..."
)

embedding_model = (
    get_embedding_model()
)

embedding_model.encode(
    ["warm up"],
    normalize_embeddings=True
)


# RUN FAISS

faiss_results = evaluate_method(

    "FAISS ONLY",

    lambda question:
        retrieve_faiss(
            question,
            chunks,
            index,
            top_k=10
        )
)


# RUN BM25

bm25_results = evaluate_method(

    "BM25 ONLY",

    lambda question:
        retrieve_bm25(
            question,
            chunks,
            bm25,
            top_k=10
        )
)


# RUN HYBRID

hybrid_results = evaluate_method(

    "HYBRID - FAISS + BM25",

    lambda question:
        retrieve_hybrid(
            question,
            chunks,
            index,
            bm25,
            top_k=10
        )
)


# FINAL SUMMARY

summaries = [

    summarize_results(
        "FAISS",
        faiss_results
    ),

    summarize_results(
        "BM25",
        bm25_results
    ),

    summarize_results(
        "HYBRID",
        hybrid_results
    )
]


print()
print(
    "=" * 90
)

print(
    "FINAL COMPARISON"
)

print(
    "=" * 90
)


print(

    f"{'Method':<12}"

    f"{'Hit@1':<10}"

    f"{'Hit@3':<10}"

    f"{'Hit@5':<10}"

    f"{'MRR':<10}"

    f"{'Time':<12}"
)


print(
    "-" * 90
)


for summary in summaries:

    print(

        f"{summary['method']:<12}"

        f"{summary['hit@1']:<10.3f}"

        f"{summary['hit@3']:<10.3f}"

        f"{summary['hit@5']:<10.3f}"

        f"{summary['mrr']:<10.3f}"

        f"{summary['time']:<12.4f}"
    )


print()

print(
    f"Questions evaluated: "
    f"{len(evaluation_questions)}"
)

print(
    "Evaluation complete."
)
import numpy as np

from src.retrieval.semantic import (
    expand_question,
    get_embedding_model
)

from src.retrieval.bm25 import (
    get_bm25_scores
)


def normalize_scores(scores):
    """
    Normalize a dictionary of scores to [0, 1].

    Example:
        {
            0: 2.0,
            1: 4.0,
            2: 6.0
        }

    becomes approximately:

        {
            0: 0.0,
            1: 0.5,
            2: 1.0
        }
    """

    if not scores:
        return {}

    values = list(
        scores.values()
    )

    min_score = min(
        values
    )

    max_score = max(
        values
    )

    if max_score == min_score:

        return {
            key: 1.0
            for key in scores
        }

    return {
        key: (
            value - min_score
        ) / (
            max_score - min_score
        )
        for key, value
        in scores.items()
    }


def retrieve_chunks(
    question,
    chunks,
    index,
    top_k=10
):
    """
    Hybrid retrieval using:

    - FAISS semantic search
    - BM25 lexical search
    - normalized weighted score fusion

    Current weighting:

        70% semantic
        30% BM25
    """

    if (
        not question
        or not chunks
        or index is None
        or index.ntotal == 0
    ):
        return []

    candidate_size = min(
        max(
            top_k * 5,
            50
        ),
        index.ntotal
    )

    model = get_embedding_model()

    expanded_question = expand_question(
        question
    )

    query_embedding = model.encode(
        [expanded_question],
        normalize_embeddings=True,
        show_progress_bar=False
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )


    # --------------------------------------------------
    # 1. SEMANTIC CANDIDATES
    # --------------------------------------------------

    semantic_values, semantic_indices = (
        index.search(
            query_embedding,
            candidate_size
        )
    )

    semantic_scores = {}

    candidate_indices = set()

    for score, idx in zip(
        semantic_values[0],
        semantic_indices[0]
    ):

        idx = int(idx)

        if idx < 0:
            continue

        semantic_scores[idx] = float(
            score
        )

        candidate_indices.add(
            idx
        )


    # --------------------------------------------------
    # 2. BM25 CANDIDATES
    # --------------------------------------------------

    all_bm25_scores = get_bm25_scores(
        question,
        chunks
    )

    bm25_ranked_indices = sorted(
        all_bm25_scores,
        key=all_bm25_scores.get,
        reverse=True
    )[:candidate_size]

    for idx in bm25_ranked_indices:

        candidate_indices.add(
            idx
        )


    # --------------------------------------------------
    # 3. SEMANTIC SCORE FOR BM25-ONLY CANDIDATES
    # --------------------------------------------------

    for idx in candidate_indices:

        if idx in semantic_scores:
            continue

        vector = index.reconstruct(
            int(idx)
        )

        vector = np.asarray(
            vector,
            dtype="float32"
        )

        semantic_score = float(
            np.dot(
                query_embedding[0],
                vector
            )
        )

        semantic_scores[idx] = (
            semantic_score
        )


    # --------------------------------------------------
    # 4. BM25 SCORES FOR ALL CANDIDATES
    # --------------------------------------------------

    candidate_bm25_scores = {
        idx: all_bm25_scores.get(
            idx,
            0.0
        )
        for idx in candidate_indices
    }


    # --------------------------------------------------
    # 5. NORMALIZATION
    # --------------------------------------------------

    candidate_semantic_scores = {
        idx: semantic_scores.get(
            idx,
            0.0
        )
        for idx in candidate_indices
    }

    semantic_normalized = (
        normalize_scores(
            candidate_semantic_scores
        )
    )

    bm25_normalized = (
        normalize_scores(
            candidate_bm25_scores
        )
    )


    # --------------------------------------------------
    # 6. HYBRID FUSION
    # --------------------------------------------------

    results = []

    for idx in candidate_indices:

        semantic_score = (
            candidate_semantic_scores[
                idx
            ]
        )

        bm25_score = (
            candidate_bm25_scores[
                idx
            ]
        )

        semantic_norm = (
            semantic_normalized[
                idx
            ]
        )

        bm25_norm = (
            bm25_normalized[
                idx
            ]
        )

        hybrid_score = (
            0.70 * semantic_norm
            +
            0.30 * bm25_norm
        )

        results.append(
            {
                "chunk": chunks[idx],

                "semantic_score":
                    float(
                        semantic_score
                    ),

                "keyword_score":
                    float(
                        bm25_score
                    ),

                "semantic_normalized":
                    float(
                        semantic_norm
                    ),

                "bm25_normalized":
                    float(
                        bm25_norm
                    ),

                "hybrid_score":
                    float(
                        hybrid_score
                    ),

                "index":
                    int(idx)
            }
        )


    # --------------------------------------------------
    # 7. SORT
    # --------------------------------------------------

    results.sort(
        key=lambda item:
            item["hybrid_score"],
        reverse=True
    )


    # --------------------------------------------------
    # 8. REMOVE EXACT DUPLICATE CHUNKS
    # --------------------------------------------------

    unique_results = []

    seen_texts = set()

    for result in results:

        chunk_text = (
            result["chunk"]
            .get(
                "text",
                ""
            )
        )

        normalized_text = " ".join(
            chunk_text
            .lower()
            .split()
        )

        if normalized_text in seen_texts:
            continue

        seen_texts.add(
            normalized_text
        )

        unique_results.append(
            result
        )

        if len(
            unique_results
        ) >= top_k:

            break


    return unique_results
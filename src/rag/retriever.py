import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

VECTOR_STORE_DIR = (
    BASE_DIR
    / "data"
    / "vector_store"
)

INDEX_FILE = (
    VECTOR_STORE_DIR
    / "faiss_index.bin"
)

METADATA_FILE = (
    VECTOR_STORE_DIR
    / "chunk_metadata.json"
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# LOAD VECTOR STORE
# ============================================================

def load_vector_store():

    print("\nLoading FAISS vector store...")

    if not INDEX_FILE.exists():

        raise FileNotFoundError(
            f"\nFAISS index not found:\n{INDEX_FILE}"
        )

    if not METADATA_FILE.exists():

        raise FileNotFoundError(
            f"\nMetadata file not found:\n{METADATA_FILE}"
        )


    # Load FAISS index
    index = faiss.read_index(
        str(INDEX_FILE)
    )


    # Load metadata
    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(file)


    # Validate vector store
    if index.ntotal == 0:

        raise ValueError(
            "FAISS index contains no vectors."
        )


    if len(metadata) == 0:

        raise ValueError(
            "Metadata file contains no records."
        )


    if index.ntotal != len(metadata):

        print(
            "\nWARNING:"
        )

        print(
            "Number of FAISS vectors does not match "
            "number of metadata records."
        )


    print(
        "Vector store loaded successfully!"
    )

    print(
        f"Total vectors: {index.ntotal}"
    )

    print(
        f"Metadata records: {len(metadata)}"
    )


    return index, metadata


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def load_embedding_model():

    print(
        "\nLoading embedding model..."
    )

    print(
        f"Model: {MODEL_NAME}"
    )


    model = SentenceTransformer(
        MODEL_NAME
    )


    print(
        "Embedding model loaded successfully!"
    )


    return model


# ============================================================
# CONVERT DISTANCE TO RELEVANCE SCORE
# ============================================================

def calculate_relevance_score(
    distance
):

    """
    Converts FAISS L2 distance into a simple
    relevance score.

    Lower distance = higher relevance.
    """

    score = 1 / (
        1 + distance
    )

    return round(
        float(score),
        4
    )


# ============================================================
# REMOVE DUPLICATE / HIGHLY SIMILAR CONTENT
# ============================================================

def remove_duplicate_results(
    results
):

    unique_results = []

    seen_sources_and_content = set()


    for result in results:

        source = result[
            "source"
        ]

        content = result[
            "content"
        ].strip()


        # Use beginning of content as duplicate signature
        signature = (
            source,
            content[:250]
        )


        if signature in seen_sources_and_content:

            continue


        seen_sources_and_content.add(
            signature
        )


        unique_results.append(
            result
        )


    return unique_results


# ============================================================
# RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve(
    query,
    model,
    index,
    metadata,
    top_k=5
):

    # --------------------------------------------------------
    # VALIDATE QUERY
    # --------------------------------------------------------

    if not query:

        return []


    query = query.strip()


    if not query:

        return []


    # --------------------------------------------------------
    # MAKE SURE top_k IS VALID
    # --------------------------------------------------------

    top_k = min(
        top_k,
        index.ntotal
    )


    # --------------------------------------------------------
    # CONVERT QUERY INTO EMBEDDING
    # --------------------------------------------------------

    query_embedding = model.encode(

        [query],

        convert_to_numpy=True

    )


    query_embedding = (
        query_embedding.astype(
            "float32"
        )
    )


    # --------------------------------------------------------
    # SEARCH FAISS
    # --------------------------------------------------------

    distances, indices = index.search(

        query_embedding,

        top_k

    )


    results = []


    # --------------------------------------------------------
    # PROCESS SEARCH RESULTS
    # --------------------------------------------------------

    for rank, (
        distance,
        chunk_index
    ) in enumerate(

        zip(
            distances[0],
            indices[0]
        ),

        start=1

    ):


        # Skip invalid FAISS results
        if chunk_index == -1:

            continue


        # Skip invalid metadata indices
        if chunk_index >= len(metadata):

            continue


        chunk = metadata[
            chunk_index
        ]


        relevance_score = (
            calculate_relevance_score(
                distance
            )
        )


        results.append({

            "rank": rank,

            "distance": round(
                float(distance),
                4
            ),

            "relevance_score":
                relevance_score,

            "chunk_id": chunk.get(
                "chunk_id",
                f"chunk_{chunk_index}"
            ),

            "source": chunk.get(
                "source",
                "Unknown"
            ),

            "content": chunk.get(
                "content",
                ""
            ).strip()

        })


    # --------------------------------------------------------
    # REMOVE EMPTY RESULTS
    # --------------------------------------------------------

    results = [

        result

        for result in results

        if result[
            "content"
        ]

    ]


    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    results = (
        remove_duplicate_results(
            results
        )
    )


    # --------------------------------------------------------
    # RE-RANK RESULTS
    # --------------------------------------------------------

    results = sorted(

        results,

        key=lambda x: x[
            "distance"
        ]

    )


    # --------------------------------------------------------
    # UPDATE RANKS AFTER FILTERING
    # --------------------------------------------------------

    for rank, result in enumerate(

        results,

        start=1

    ):

        result[
            "rank"
        ] = rank


    return results


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    query,
    results
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RETRIEVAL RESULTS"
    )

    print(
        "=" * 70
    )


    print(
        f"\nQuery:\n{query}"
    )


    if not results:

        print(
            "\nNo relevant information found."
        )

        return


    for result in results:

        print(
            "\n"
            + "-" * 70
        )


        print(
            f"Rank: "
            f"{result['rank']}"
        )


        print(
            f"Chunk ID: "
            f"{result['chunk_id']}"
        )


        print(
            f"Source: "
            f"{result['source']}"
        )


        print(
            f"FAISS Distance: "
            f"{result['distance']:.4f}"
        )


        print(
            f"Relevance Score: "
            f"{result['relevance_score']:.4f}"
        )


        print(
            "\nRetrieved Content:"
        )


        print(
            result["content"]
        )


    print(
        "\n"
        + "=" * 70
    )


# ============================================================
# MAIN - RETRIEVAL TESTING
# ============================================================

def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "INSIGHTRAG RETRIEVAL SYSTEM"
    )

    print(
        "=" * 70
    )


    # Load vector store
    index, metadata = (
        load_vector_store()
    )


    # Load embedding model
    model = (
        load_embedding_model()
    )


    # Test queries
    test_queries = [

        (
            "Why did Clothing profitability decline "
            "in November and December 2025?"
        ),

        (
            "Which category has the highest sales?"
        ),

        (
            "Which products have the highest "
            "return rates?"
        ),

        (
            "What caused the Electronics "
            "profitability issue in Q3 2025?"
        ),

        (
            "What are the biggest business "
            "problems?"
        )

    ]


    # Run tests
    for query in test_queries:


        results = retrieve(

            query=query,

            model=model,

            index=index,

            metadata=metadata,

            top_k=5

        )


        display_results(

            query,

            results

        )


    print(
        "\nRETRIEVAL TESTING "
        "COMPLETED SUCCESSFULLY!"
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    main()
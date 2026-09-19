import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

CHUNKS_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "rag_chunks.json"
)

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
# LOAD CHUNKS
# ============================================================

def load_chunks():

    print("\nLoading RAG chunks...\n")

    if not CHUNKS_FILE.exists():

        raise FileNotFoundError(
            f"Chunks file not found:\n{CHUNKS_FILE}"
        )

    with open(
        CHUNKS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        chunks = json.load(file)

    print(
        f"Chunks loaded successfully!"
    )

    print(
        f"Total chunks: {len(chunks)}"
    )

    return chunks


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def load_embedding_model():

    print("\nLoading embedding model...")

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
# GENERATE EMBEDDINGS
# ============================================================

def generate_embeddings(
    model,
    chunks
):

    print(
        "\nGenerating embeddings..."
    )

    texts = [
        chunk["content"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype(
        "float32"
    )

    print(
        "\nEmbeddings generated successfully!"
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    return embeddings


# ============================================================
# CREATE FAISS INDEX
# ============================================================

def create_faiss_index(
    embeddings
):

    print(
        "\nCreating FAISS vector index..."
    )

    embedding_dimension = (
        embeddings.shape[1]
    )

    index = faiss.IndexFlatL2(
        embedding_dimension
    )

    index.add(
        embeddings
    )

    print(
        "FAISS index created successfully!"
    )

    print(
        f"Vector dimension: "
        f"{embedding_dimension}"
    )

    print(
        f"Total vectors indexed: "
        f"{index.ntotal}"
    )

    return index


# ============================================================
# SAVE VECTOR STORE
# ============================================================

def save_vector_store(
    index,
    chunks
):

    print(
        "\nSaving vector store..."
    )

    VECTOR_STORE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save FAISS index
    faiss.write_index(
        index,
        str(INDEX_FILE)
    )

    # Save metadata
    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(
        "\nVector store saved successfully!"
    )

    print(
        "\nFAISS Index:"
    )

    print(
        INDEX_FILE
    )

    print(
        "\nMetadata:"
    )

    print(
        METADATA_FILE
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print(
        "\n"
        + "=" * 60
    )

    print(
        "RAG VECTOR STORE CREATION"
    )

    print(
        "=" * 60
    )


    # Load chunks
    chunks = load_chunks()


    # Load embedding model
    model = (
        load_embedding_model()
    )


    # Generate embeddings
    embeddings = (
        generate_embeddings(
            model,
            chunks
        )
    )


    # Create FAISS index
    index = (
        create_faiss_index(
            embeddings
        )
    )


    # Save vector store
    save_vector_store(
        index,
        chunks
    )


    print(
        "\n"
        + "=" * 60
    )

    print(
        "VECTOR STORE CREATION "
        "COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":

    main()
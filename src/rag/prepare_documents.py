import os
import json
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

KNOWLEDGE_BASE_DIR = BASE_DIR / "data" / "knowledge_base"

OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_FILE = OUTPUT_DIR / "rag_chunks.json"


# ============================================================
# CONFIGURATION
# ============================================================

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


# ============================================================
# LOAD DOCUMENTS
# ============================================================

def load_documents():

    print("\nLoading knowledge base documents...\n")

    documents = []

    txt_files = list(KNOWLEDGE_BASE_DIR.glob("*.txt"))

    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found in: {KNOWLEDGE_BASE_DIR}"
        )

    for file_path in txt_files:

        with open(file_path, "r", encoding="utf-8") as file:

            content = file.read()

        document = {
            "source": file_path.name,
            "content": content
        }

        documents.append(document)

        print(f"Loaded: {file_path.name}")

    print("\nDocuments loaded successfully!")
    print(f"Total documents: {len(documents)}")

    return documents


# ============================================================
# TEXT CHUNKING
# ============================================================

def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end]

        # Avoid cutting words in the middle
        if end < text_length:

            last_space = chunk.rfind(" ")

            if last_space != -1:
                chunk = chunk[:last_space]
                end = start + last_space

        chunk = chunk.strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

        # Safety check
        if start < 0:
            start = 0

        if end >= text_length:
            break

    return chunks


# ============================================================
# CREATE DOCUMENT CHUNKS
# ============================================================

def create_chunks(documents):

    print("\nCreating document chunks...\n")

    all_chunks = []

    chunk_id = 1

    for document in documents:

        source = document["source"]

        content = document["content"]

        text_chunks = split_text(content)

        print(
            f"{source} -> {len(text_chunks)} chunks created"
        )

        for index, chunk in enumerate(text_chunks):

            chunk_data = {

                "chunk_id": f"chunk_{chunk_id:04d}",

                "source": source,

                "chunk_index": index,

                "content": chunk

            }

            all_chunks.append(chunk_data)

            chunk_id += 1

    print("\nChunking completed successfully!")
    print(f"Total chunks created: {len(all_chunks)}")

    return all_chunks


# ============================================================
# SAVE CHUNKS
# ============================================================

def save_chunks(chunks):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nChunks saved successfully!")

    print(f"Saved to:\n{OUTPUT_FILE}")


# ============================================================
# PREVIEW CHUNKS
# ============================================================

def preview_chunks(chunks, number_of_chunks=3):

    print("\n" + "=" * 60)

    print("CHUNK PREVIEW")

    print("=" * 60)

    for chunk in chunks[:number_of_chunks]:

        print(f"\nChunk ID: {chunk['chunk_id']}")

        print(f"Source: {chunk['source']}")

        print(f"Chunk Index: {chunk['chunk_index']}")

        print("\nContent:")

        print(chunk["content"])

        print("\n" + "-" * 60)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("\n" + "=" * 60)

    print("RAG DOCUMENT PREPARATION PIPELINE")

    print("=" * 60)


    # Load documents
    documents = load_documents()


    # Create chunks
    chunks = create_chunks(documents)


    # Save chunks
    save_chunks(chunks)


    # Preview chunks
    preview_chunks(chunks)


    print("\n" + "=" * 60)

    print("DOCUMENT PREPARATION COMPLETED SUCCESSFULLY")

    print("=" * 60)


if __name__ == "__main__":

    main()
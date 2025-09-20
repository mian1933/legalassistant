# retriever_service_en.py
import torch
import chromadb
from sentence_transformers import SentenceTransformer
import argparse  # For parsing command-line arguments

# --- Configuration (must be consistent with the build script) ---
CHROMA_DB_PATH = "/home/sa/bar-exam-housing/processed_data/passage/chroma_db_large2/"
COLLECTION_NAME = "legal_docs_large_collection2"
EMBEDDING_MODEL_NAME = "/home/sa/bar-exam-housing/bge-m3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K_RESULTS = 3


def main():
    # --- Step 1: Set up command-line argument parser ---
    parser = argparse.ArgumentParser(description="Retrieve documents related to a query from ChromaDB.")
    parser.add_argument("--query", type=str, required=True, help="The user's search query text.")
    args = parser.parse_args()

    # --- Step 2: Initialize model and database ---
    try:
        # Load the embedding model
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)

        # Connect to ChromaDB
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        # If initialization fails, print error to stderr and exit
        import sys
        print(f"Error initializing models or database: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Step 3: Encode query and perform retrieval ---
    query_embedding = embedding_model.encode(
        args.query,
        normalize_embeddings=True
    )
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=TOP_K_RESULTS,
        include=["documents"]
    )

    # --- Step 4: Format results and print to standard output ---
    documents = results.get('documents', [[]])[0]
    if not documents:
        # Print a clear identifier even if no results are found
        print("No relevant materials found in the database.")
        return

    # Join multiple document snippets into a single string and print
    context = "\n\n".join([f"--- Reference Material {i + 1} ---\n{doc}" for i, doc in enumerate(documents)])
    print(context)


if __name__ == "__main__":
    main()
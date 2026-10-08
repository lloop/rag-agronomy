from typing import Any, cast
import sys
from pathlib import Path
from unittest import result
import chromadb
from chromadb.utils import embedding_functions

def query_vector_db(target_folder_name: str, query_text: str, n_results: int = 3):
    base_dir = Path("..") / target_folder_name
    db_path = base_dir / "vector_db"

    if not db_path.exists():
        print(f"Error: Vector database not found at '{db_path.resolve()}'")
        print(f"Run 'python ingest_data.py {target_folder_name}' first.")
        sys.exit(1)

    chroma_client = chromadb.PersistentClient(path=str(db_path))
    embedding_fn = cast(Any, embedding_functions.DefaultEmbeddingFunction())

    collection = chroma_client.get_or_create_collection(
        name="agronomy",
        embedding_function=embedding_fn
    )

    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    
    # Check attributes of QueryResult
    # if hasattr(result, "__dict__"):
    #     print("QueryResult contents:", vars(result))
    # else:
    #     print("QueryResult contents:", result)

    return results

if __name__ == "__main__":
    target_dataset = sys.argv[1] if len(sys.argv) > 1 else "data_agronomy"
    search_query = sys.argv[2] if len(sys.argv) > 2 else "charcoals role in soil health"

    print(f"Searching in '{target_dataset}' for: \"{search_query}\"\n")
    
    query_results = query_vector_db(target_dataset, search_query)

    documents = (query_results.get("documents") or [[]])[0]
    metadatas = (query_results.get("metadatas") or [[]])[0]

    for idx, (doc, meta) in enumerate(zip(documents, metadatas), 1):
        if isinstance(meta, dict):
            source = meta.get("source_doc", "Unknown")
            folder = meta.get("enclosing_folder", "Unknown")
            page = meta.get("page_number", "?")
        else:
            source, folder, page = "Unknown", "Unknown", "?"
            
        print(f"--- Result {idx} ---")
        print(f"Citation: [{folder} / {source}] Page {page}")
        print(f"Content: {doc}\n")
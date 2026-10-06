from typing import Any, List, Dict, cast
import chromadb
from chromadb.utils import embedding_functions


class DocumentRetriever:
    def __init__(self, db_path: str = "./vector_db", collection_name: str = "agronomy"):
        self.chroma_client = chromadb.PersistentClient(path=db_path)

        # Cast to Any to prevent Pyright DefaultEmbeddingFunction type errors
        self.embedding_fn = cast(Any, embedding_functions.DefaultEmbeddingFunction())

        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )

    def retrieve(self, query_text: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Queries ChromaDB for vector matches and returns metadata-rich chunks."""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k
        )

        retrieved_chunks = []

        # Safe extraction guarding against Optional types for Pylance
        docs = results.get("documents")
        metas = results.get("metadatas")

        if docs is not None and metas is not None and len(docs) > 0 and len(metas) > 0:
            for doc, meta in zip(docs[0], metas[0]):
                retrieved_chunks.append({
                    "content": doc,
                    "source_doc": meta.get("source_doc", "Unknown"),
                    "page_number": meta.get("page_number", 0),
                    "enclosing_folder": meta.get("enclosing_folder", "Unknown")
                })

        return retrieved_chunks 
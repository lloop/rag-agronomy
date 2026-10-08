from typing import Any, List, Dict, cast
import chromadb
from chromadb.utils import embedding_functions
from config import Config


class DocumentRetriever:
    
    def __init__(self, db_path: str = str(Config.DATASETS["agronomy"]), collection_name: str = "agronomy"):
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
        dists = results.get("distances")

        if docs and metas and dists and len(docs) > 0:
            for doc, meta, dist in zip(docs[0], metas[0], dists[0]):
                retrieved_chunks.append({
                    "content": doc,
                    "text": doc,
                    "distance": dist,
                    "source_doc": meta.get("source_doc", "Unknown"),
                    "page_number": meta.get("page_number", 0),
                    "enclosing_folder": meta.get("enclosing_folder", "Unknown")
                })

        print(f"Retrieved {len(retrieved_chunks)} chunks for query: '{query_text}'", flush=True)

        return retrieved_chunks
from typing import Any, cast
from pathlib import Path
import pymupdf
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentIngester:
    def __init__(self, db_path: str = "./vector_db"):
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.embedding_fn = cast(Any, embedding_functions.DefaultEmbeddingFunction())
        self.collection = self.chroma_client.get_or_create_collection(
            name="agronomy_handbooks",
            embedding_function=self.embedding_fn
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100
        )

    def is_file_fully_indexed(self, pdf_path: Path) -> bool:
        """Checks if a document is already present in ChromaDB."""
        results = self.collection.get(
            where={"source_doc": pdf_path.name},
            limit=1
        )
        return len(results.get("ids", [])) > 0

    def process_pdf(self, pdf_path: Path, metadata: dict | None = None):
        """Extracts text page-by-page and appends new chunks without duplicating."""
        if self.is_file_fully_indexed(pdf_path):
            print(f"Skipping {pdf_path.name} (already indexed)")
            return

        doc = pymupdf.open(pdf_path)
        documents = []
        metadatas = []
        ids = []

        enclosing_folder = pdf_path.parent.name
        extra_metadata = metadata or {}

        chunk_counter = 0
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = str(page.get_text())
            if not text.strip():
                continue

            chunks = self.text_splitter.split_text(text)
            for chunk in chunks:
                chunk_id = f"{pdf_path.stem}_p{page_num + 1}_{chunk_counter}"
                documents.append(chunk)
                
                chunk_metadata = {
                    "source_doc": pdf_path.name,
                    "enclosing_folder": enclosing_folder,
                    "page_number": page_num + 1
                }
                chunk_metadata.update(extra_metadata)
                
                metadatas.append(chunk_metadata)
                ids.append(chunk_id)
                chunk_counter += 1

            if documents:
                batch_size = 1000
                total_chunks = len(documents)

                for i in range(0, total_chunks, batch_size):
                    self.collection.add(
                        documents=documents[i:i + batch_size],
                        metadatas=metadatas[i:i + batch_size],
                        ids=ids[i:i + batch_size]
                    )
                
                print(f"Indexed {total_chunks} chunks from {pdf_path.name}")
            else:
                print(f"Skipped {pdf_path.name}: No extractable text found.")
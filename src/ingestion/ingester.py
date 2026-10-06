import hashlib
from typing import Any, cast
from pathlib import Path
import pymupdf
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentIngester:
    def __init__(self, db_path: str = "./vector_db", collection_name: str = "general"):
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.embedding_fn = cast(Any, embedding_functions.DefaultEmbeddingFunction())
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )

    def _compute_file_hash(self, file_path: Path) -> str:
        """Computes an MD5 hash of the raw PDF content for directory-agnostic tracking."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def is_file_fully_indexed(self, pdf_path: Path) -> bool:
        """Checks if a document is already present in ChromaDB using its content hash."""
        file_hash = self._compute_file_hash(pdf_path)
        results = self.collection.get(
            where={"file_hash": file_hash},
            limit=1
        )
        return len(results.get("ids", [])) > 0

    def process_pdf(self, pdf_path: Path, metadata: dict | None = None):
            """Extracts text page-by-page and updates/appends chunks using content-hash tracking."""
            file_hash = self._compute_file_hash(pdf_path)

            if self.is_file_fully_indexed(pdf_path):
                print(f"Skipping {pdf_path.name} (already indexed)")
                return

            try:
                doc = pymupdf.open(pdf_path)
            except (pymupdf.FileDataError, Exception) as e:
                print(f"Skipping corrupt or unreadable PDF {pdf_path.name}: {e}")
                return

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
                    # Content hash in chunk ID ensures reorganization or renaming won't duplicate vectors
                    chunk_id = f"{file_hash}_p{page_num + 1}_{chunk_counter}"
                    documents.append(chunk)
                    
                    chunk_metadata = {
                        "source_doc": pdf_path.name,
                        "enclosing_folder": enclosing_folder,
                        "page_number": page_num + 1,
                        "file_hash": file_hash
                    }
                    chunk_metadata.update(extra_metadata)
                    
                    metadatas.append(chunk_metadata)
                    ids.append(chunk_id)
                    chunk_counter += 1

            if documents:
                batch_size = 1000
                total_chunks = len(documents)

                for i in range(0, total_chunks, batch_size):
                    # Using upsert ensures metadata updates if a file moves to a new folder
                    self.collection.upsert(
                        documents=documents[i:i + batch_size],
                        metadatas=metadatas[i:i + batch_size],
                        ids=ids[i:i + batch_size]
                    )
                
                print(f"Indexed {total_chunks} chunks from {pdf_path.name}")
            else:
                print(f"Skipped {pdf_path.name}: No extractable text found.")
from pathlib import Path
import pymupdf
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

class AgronomyDocumentIngester:
    def __init__(self, db_path: str = "./vector_db"):
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        # Using default sentence-transformers or OpenAI embeddings
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.chroma_client.get_or_create_collection(
            name="agronomy_handbooks",
            embedding_function=self.embedding_fn
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100
        )

    def process_pdf(self, pdf_path: Path):
        """Extracts text page-by-page to ensure exact page citations."""
        doc = pymupdf.open(pdf_path)
        documents = []
        metadatas = []
        ids = []

        chunk_counter = 0
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if not text.strip():
                continue

            chunks = self.text_splitter.split_text(text)
            for chunk in chunks:
                chunk_id = f"{pdf_path.stem}_p{page_num + 1}_{chunk_counter}"
                documents.append(chunk)
                metadatas.append({
                    "source_doc": pdf_path.name,
                    "page_number": page_num + 1
                })
                ids.append(chunk_id)
                chunk_counter += 1

        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Indexed {len(documents)} chunks from {pdf_path.name}")
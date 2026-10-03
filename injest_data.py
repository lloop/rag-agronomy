from pathlib import Path
from src.ingester import AgronomyDocumentIngester

if __name__ == "__main__":
    ingester = AgronomyDocumentIngester(db_path="./vector_db")
    data_dir = Path("./data")
    
    for pdf_file in data_dir.glob("*.pdf"):
        print(f"Indexing {pdf_file.name}...")
        ingester.process_pdf(pdf_file)
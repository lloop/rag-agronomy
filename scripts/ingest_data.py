import sys
from pathlib import Path
from ingestion.ingester import DocumentIngester

if __name__ == "__main__":
    target_folder_name = sys.argv[1] if len(sys.argv) > 1 else "data_agronomy"
    base_dir = Path("..") / target_folder_name
    db_path = base_dir / "vector_db"
    
    if not base_dir.exists():
        print(f"Error: Directory '{base_dir.resolve()}' does not exist.")
        sys.exit(1)

    print(f"Targeting Dataset: {base_dir.resolve()}")
    print(f"Vector Database Location: {db_path.resolve()}\n")

    ingester = DocumentIngester(db_path=str(db_path))
    
    pdf_count = 0
    for pdf_file in base_dir.rglob("*.pdf"):
        if "vector_db" in pdf_file.parts:
            continue
        pdf_count += 1
        enclosing_folder = pdf_file.parent.name
        print(f"Processing {pdf_file.name} (Folder: {enclosing_folder})...")
        ingester.process_pdf(pdf_file)

    if pdf_count == 0:
        print(f"No PDF files found in {base_dir.resolve()}")
from pathlib import Path
from src.ingester import AgronomyDocumentIngester
from src.rag_engine import AgronomyRAGEngine

def setup_demo_data():
    """Simulates creating or loading local handbooks."""
    pdf_dir = Path("data")
    pdf_dir.mkdir(exist_ok=True)
    
    # Place your handbook PDFs in the /data folder
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if pdf_files:
        ingester = AgronomyDocumentIngester()
        for pdf in pdf_files:
            ingester.process_pdf(pdf)

if __name__ == "__main__":
    setup_demo_data()

    engine = AgronomyRAGEngine()
    
    query = "What are the allowed applications and restrictions for copper fungicides in organic tomato production?"
    print(f"\nUser Query: '{query}'\n" + "="*60)
    
    result = engine.query(query)

    if not result.is_sufficient_context:
        print("\n[Warning]: Insufficient evidence found in source handbooks.")
        print(f"Notes: {result.answer}")
    else:
        print(f"\nAnswer:\n{result.answer}\n")
        print(f"Confidence Score: {result.confidence_score * 100}%")
        print("\nExact Page Citations:")
        for cite in result.citations:
            print(f" - [{cite.source_doc} | Page {cite.page_number}]")
            print(f"   Quote: \"{cite.snippet}\"")
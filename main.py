from dotenv import load_dotenv
load_dotenv()

from pipeline.rag_engine import RAGEngine

if __name__ == "__main__":
    # Chroma automatically loads the existing vector_db off disk!
    engine = RAGEngine(db_path="./vector_db")
    
       
    query = "Tell me about charcoal?"    


    result = engine.query(query)

    print("=== QUERY ===")
    print(query)
    
    print("=== ANSWER ===")
    print(result.answer)

    print("\n=== METADATA ===")
    print(f"Is Sufficient Context: {result.is_sufficient_context}")
    print(f"Confidence Score: {result.confidence_score}")

    print("\n=== CITATIONS ===")
    if result.citations:
        for i, citation in enumerate(result.citations, 1):
            print(f"Citation #{i}:")
            print(f"  Source Document: {citation.source_doc}")
            print(f"  Page Number:     {citation.page_number}")
            print(f"  Supporting Quote: \"{citation.snippet}\"")
    else:
        print("No citations provided.")
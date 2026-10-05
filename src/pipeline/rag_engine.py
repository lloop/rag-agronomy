import os
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from src.retriever import DocumentRetriever

load_dotenv()

# --- Schema Definitions ---

class Citation(BaseModel):
    enclosing_folder: str = Field(description="Domain or folder name containing the document.")
    source_doc: str = Field(description="Name of the PDF or handbook source file.")
    page_number: int = Field(description="Exact page number where the information was located.")
    snippet: str = Field(description="Direct quote or snippet supporting the fact.")

class Answer(BaseModel):
    answer: str = Field(description="Direct technical answer to the user's question.")
    citations: List[Citation] = Field(description="List of exact page citations backing the answer.")
    confidence_score: float = Field(description="Confidence rating between 0.0 and 1.0 based on retrieved context quality.")
    is_sufficient_context: bool = Field(description="Set to false if context lacks information to safely answer.")

# --- RAG Engine ---

class RAGEngine:
    def __init__(self, db_path: str = "../data_agronomy/vector_db", model: str = "gemini-2.5-flash"):
        gemini_key = os.getenv("GEMINI_API_KEY")
        
        self.client = OpenAI(
            api_key=gemini_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        self.model = model
        self.retriever = DocumentRetriever(db_path=db_path)

    def query(self, user_question: str, top_k: int = 4) -> Answer:
        # 1. Retrieve chunks via DocumentRetriever   
        chunks = self.retriever.retrieve(user_question, top_k=top_k)

        # 2. Format Context for Prompting
        formatted_context = ""
        for i, chunk in enumerate(chunks):
            formatted_context += f"\n--- CHUNK {i+1} ---\n"
            formatted_context += f"Domain: {chunk.get('enclosing_folder', 'Unknown')}\n"
            formatted_context += f"Source: {chunk['source_doc']} (Page {chunk['page_number']})\n"
            formatted_context += f"Content: {chunk['content']}\n"

        system_prompt = """
You are an expert Assistant. 
Answer questions accurately using ONLY the provided context snippets.
If the provided context does not contain enough evidence to answer safely, set `is_sufficient_context` to false and explain what is missing.
Every factual claim must have a corresponding item in the `citations` array matching the exact domain, source document, and page number.
"""

        user_prompt = f"""
Context:
{formatted_context}

Question: {user_question}
"""

        # 3. Call LLM with Strict Structured Output
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=Answer,
            temperature=0.0
        )

        # 4. Extract and type-guard parsed result for Pylance
        parsed_result = response.choices[0].message.parsed
        if parsed_result is None:
            raise ValueError("Failed to parse structured LLM response into Answer.")

        return parsed_result 
from typing import List, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

# --- Schema Definitions ---

class Citation(BaseModel):
    source_doc: str = Field(description="Name of the PDF or handbook source file.")
    page_number: int = Field(description="Exact page number where the information was located.")
    snippet: str = Field(description="Direct quote or snippet supporting the fact.")

class AgronomicAnswer(BaseModel):
    answer: str = Field(description="Direct technical answer to the user's question.")
    citations: List[Citation] = Field(description="List of exact page citations backing the answer.")
    confidence_score: float = Field(description="Confidence rating between 0.0 and 1.0 based on retrieved context quality.")
    is_sufficient_context: bool = Field(description="Set to false if context lacks information to safely answer.")

# --- Engine Logic ---

class AgronomyRAGEngine:
    def __init__(self, db_path: str = "./vector_db", model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.chroma_client.get_collection(
            name="agronomy_handbooks",
            embedding_function=self.embedding_fn
        )

    def query(self, user_question: str, top_k: int = 4) -> AgronomicAnswer:
        # 1. Retrieve relevant chunks
        results = self.collection.query(
            query_texts=[user_question],
            n_results=top_k
        )

        documents = results['documents'][0]
        metadatas = results['metadatas'][0]

        # 2. Format Context for Prompting
        formatted_context = ""
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            formatted_context += f"\n--- CHUNK {i+1} ---\n"
            formatted_context += f"Source: {meta['source_doc']} (Page {meta['page_number']})\n"
            formatted_context += f"Content: {doc}\n"

        system_prompt = """
You are an expert Agronomy and Crop Protection Assistant. 
Answer questions accurately using ONLY the provided context snippets.
If the provided context does not contain enough evidence to answer safely, set `is_sufficient_context` to false and explain what is missing.
Every factual claim must have a corresponding item in the `citations` array matching the exact source document and page number.
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
            response_format=AgronomicAnswer,
            temperature=0.0
        )

        return response.choices[0].message.parsed
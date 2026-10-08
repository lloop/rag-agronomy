import json
from config import Config
from google import genai
from google.genai import types

import threading
import time
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field
from src.retriever import DocumentRetriever

# Optional: Keep Ollama imports commented for reference
# from openai import OpenAI

DEFAULT_DB = Config.DATASETS["agronomy"]
MAX_DISTANCE = 1.3   # tune after looking at the printed distances


class Citation(BaseModel):
    enclosing_folder: str
    source_doc: str
    page_number: int
    snippet: str


class Answer(BaseModel):
    answer: str
    citations: List[Citation]
    confidence_score: float
    is_sufficient_context: bool


class LLMAnswer(BaseModel):
    """Small schema: the model only writes the answer and says which chunks it used."""
    answer: str = Field(description="Answer using only the context.")
    used_chunks: List[int] = Field(default_factory=list, description="Indices of the CHUNK numbers used.")
    is_sufficient_context: bool


SYSTEM_PROMPT = (
    "Answer using ONLY the provided context chunks. You can be verbose. "
    "List the CHUNK numbers you used in used_chunks. "
    "If the context cannot answer the question, set is_sufficient_context to false."
)


class RAGEngine:
    def __init__(self, db_path=DEFAULT_DB):
        # ------------------------------------------------------------------
        # COMMENTED OUT: Local Ollama Client setup
        # ------------------------------------------------------------------
        # self.client = OpenAI(
        #     base_url="http://localhost:11434/v1",
        #     api_key="ollama",
        #     timeout=120.0,
        #     max_retries=0,          # the SDK default of 2 silently triples the load
        # )
        # self.model = "qwen2.5:3b"
        # self._lock = threading.Lock()
        # ------------------------------------------------------------------

        # Gemini API Initialization
        self.client = genai.Client(api_key=Config.GEMINI_API_KEY)
        self.model = Config.GEMINI_MODEL
        self.retriever = DocumentRetriever(db_path=str(db_path))

    def query(self, user_question: str, top_k: int = 4) -> Answer:
        t0 = time.perf_counter()
        chunks = self.retriever.retrieve(user_question, top_k=top_k)
        t1 = time.perf_counter()
        
        # Temp timing log
        print(f"[TIMING] Retrieval took: {t1 - t0:.3f}s", flush=True)

        best = min((c["distance"] for c in chunks), default=None)

        if not chunks or best is None or best > MAX_DISTANCE:
            return Answer(
                answer="I couldn't find relevant material in your collection for that question.",
                citations=[], confidence_score=0.0, is_sufficient_context=False,
            )

        context = "\n".join(
            f"--- CHUNK {i} ---\n{c['content']}\n" for i, c in enumerate(chunks)
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {user_question}"

        # Gemini Structured Output Call
        response = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.0,
                max_output_tokens=500,
                response_mime_type="application/json",
                response_schema=LLMAnswer,
            ),
        )
        print(f"[TIMING] Gemini generation took: {time.perf_counter() - t1:.3f}s", flush=True)

        if not response.text:
            raise ValueError("Model returned empty or unparseable output.")

        # Parse JSON explicitly to bypass SDK type conversion issues
        parsed = LLMAnswer.model_validate_json(response.text)

        # Safely extract used_chunks handling both None and missing field outputs
        used_indices = getattr(parsed, "used_chunks", None) or []

        # Build citations from retrieval metadata, not from the model
        citations = []
        for i in dict.fromkeys(used_indices):   # dedupe, keep order
            if 0 <= i < len(chunks):
                c = chunks[i]
                citations.append(Citation(
                    enclosing_folder=str(c["enclosing_folder"]),
                    source_doc=str(c["source_doc"]),
                    page_number=int(c["page_number"] or 0),
                    snippet=c["content"][:300],
                ))

        confidence = max(0.0, min(1.0, 1.0 - best / 2.0))   # retrieval-based, not self-reported
        return Answer(
            answer=parsed.answer,
            citations=citations,
            confidence_score=confidence,
            is_sufficient_context=parsed.is_sufficient_context,
        )
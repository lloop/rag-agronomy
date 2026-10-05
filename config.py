import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BASE_DIR.parent


class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    # CHROMA_PERSIST_DIR = os.getenv(
    #     "CHROMA_PERSIST_DIR", str(BASE_DIR / "storage" / "chroma_db")
    # )
    # COLLECTION_NAME = os.getenv("COLLECTION_NAME", "agronomy_docs")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 3))
    FLASK_ENV = os.getenv("FLASK_ENV", "development")
    FLASK_PORT = int(os.getenv("FLASK_PORT", 5000))
    
    DATASETS = {
        "agronomy": WORKSPACE_DIR / "data_agronomy" / "chroma_db",
        "sustainability": WORKSPACE_DIR / "data_sustainability" / "chroma_db",
        "recipes": WORKSPACE_DIR / "data_recipes" / "chroma_db",
    }
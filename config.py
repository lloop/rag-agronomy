import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BASE_DIR.parent


class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash") 
    DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 3))
    FLASK_ENV = os.getenv("FLASK_ENV", "development")
    FLASK_PORT = int(os.getenv("FLASK_PORT", 5000))
    
    DATASETS = {
        "agronomy": WORKSPACE_DIR / "data_agronomy" / "vector_db",
        "sustainability": WORKSPACE_DIR / "data_sustainability" / "vector_db",
        "recipes": WORKSPACE_DIR / "data_recipes" / "vector_db",
    }
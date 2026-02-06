from pathlib import Path

REGION = "Oromia"

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
QA_JSON_PATH = PROCESSED_DATA_DIR / "qa_all.json"

INDEX_DIR = BASE_DIR / "index"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.json"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# LLM Settings
LLM_MODEL_NAME = "gemini-2.0-flash"
GEMINI_API_KEY_ENV = "GOOGLE_API_KEY"
SYSTEM_PROMPT_PATH = BASE_DIR / "config" / "system_prompt.md"

SUPPORTED_LANGUAGES = ["om", "am", "en"]

SUPPORTED_SECTORS = [
    "passport",
    "municipality",
    "utilities",
    "general",
]

DEFAULT_LANGUAGE = "om"
DEFAULT_SECTOR = "municipality"

DEMO_REQUIRED = True
DEMO_MARKER = "DEMO"

TOP_K = 3
CONFIDENCE_THRESHOLD = 0.55
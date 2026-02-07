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

MODEL_NAME = "castorini/afriberta_small"

# LLM Settings (Ollama)
# LLM_MODEL_NAME = "gemma3:1b"
# LLM_MODEL_NAME = "gpt-oss:120b-cloud"
LLM_MODEL_NAME = "gemini-3-pro-preview"
OLLAMA_BASE_URL = "http://localhost:11434"
SYSTEM_PROMPT_PATH = BASE_DIR / "config" / "system_prompt.md"

SUPPORTED_LANGUAGES = ["om", "am", "en"]

SUPPORTED_SECTORS = [
    "passport",
    "municipality",
    "utilities",
    "general",
]

DEFAULT_LANGUAGE = "om"
DEFAULT_SECTOR = "passport"

DEMO_REQUIRED = True
DEMO_MARKER = "DEMO"

TOP_K = 3
CONFIDENCE_THRESHOLD = 0.55
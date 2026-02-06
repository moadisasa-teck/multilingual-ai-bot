import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import faiss
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from config.settings import (
    FAISS_INDEX_PATH,
    METADATA_PATH,
    MODEL_NAME,
    LLM_MODEL_NAME,
    GEMINI_API_KEY_ENV,
    SYSTEM_PROMPT_PATH,
    TOP_K,
    CONFIDENCE_THRESHOLD,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryProcessor:
    def __init__(self):
        api_key = os.getenv(GEMINI_API_KEY_ENV)
        if not api_key:
            logger.warning(f"{GEMINI_API_KEY_ENV} not found. Query normalization will be skipped.")
            self.model = None
        else:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(LLM_MODEL_NAME)
        
        with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()

    def normalize(self, query: str) -> Dict[str, Any]:
        if not self.model:
            return {"rewritten_query": query, "language": "unknown", "sector_guess": "unknown"}

        prompt = f"{self.system_prompt}\n\nUser Input: \"{query}\""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json",
                )
            )
            return json.loads(response.text)
        except Exception as e:
            logger.error(f"Error normalizing query: {e}")
            return {"rewritten_query": query, "language": "unknown", "sector_guess": "unknown"}

class Chatbot:
    def __init__(self):
        self.processor = QueryProcessor()
        self.embedding_model = SentenceTransformer(MODEL_NAME, device="cpu")
        
        if not FAISS_INDEX_PATH.exists() or not METADATA_PATH.exists():
            logger.error("FAISS index or metadata not found. Please run indexing first.")
            self.index = None
            self.metadata = []
        else:
            self.index = faiss.read_index(str(FAISS_INDEX_PATH))
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

    def _normalize_vector(self, v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def search(self, query: str, sector: Optional[str] = None, language: Optional[str] = None) -> Dict[str, Any]:
        # 1. Normalize query
        norm_result = self.processor.normalize(query)
        rewritten = norm_result.get("rewritten_query", query)
        detected_lang = norm_result.get("language", "unknown")
        detected_sector = norm_result.get("sector_guess", "unknown")

        if rewritten == "unclear":
            return {
                "answer": "I'm sorry, I couldn't understand your request. Could you please rephrase it?",
                "rewritten_query": "unclear",
                "confidence": 0.0,
                "sector": "unknown",
                "language": "unknown"
            }

        if not self.index:
            return {
                "answer": "System is currently unavailable. Please try again later.",
                "rewritten_query": rewritten,
                "confidence": 0.0,
                "sector": detected_sector,
                "language": detected_lang
            }

        # 2. Embed rewritten query
        query_vec = self.embedding_model.encode([rewritten])[0]
        query_vec = self._normalize_vector(query_vec.astype("float32"))

        # 3. Search FAISS
        distances, indices = self.index.search(np.array([query_vec]), TOP_K)
        
        # 4. Filter and rank results
        best_match = None
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1: continue
            
            meta = self.metadata[idx]
            match_score = float(dist)

            # Optional filtering if user provided explicit sector/language
            if sector and meta.get("sector") != sector: continue
            if language and meta.get("language") != language: continue

            if match_score >= CONFIDENCE_THRESHOLD:
                best_match = {
                    "answer": meta["answer"],
                    "confidence": match_score,
                    "sector": meta["sector"],
                    "language": meta["language"],
                    "source_file": meta.get("source_file"),
                }
                break

        if not best_match:
            return {
                "answer": "I don't have enough information to answer that specifically. Please contact the regional office directly.",
                "rewritten_query": rewritten,
                "confidence": distances[0][0] if len(distances[0]) > 0 else 0.0,
                "sector": detected_sector,
                "language": detected_lang
            }

        return {
            **best_match,
            "rewritten_query": rewritten
        }

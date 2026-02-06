import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import faiss
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from config.settings import (
    FAISS_INDEX_PATH,
    METADATA_PATH,
    MODEL_NAME,
    LLM_MODEL_NAME,
    OLLAMA_BASE_URL,
    SYSTEM_PROMPT_PATH,
    TOP_K,
    CONFIDENCE_THRESHOLD,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryProcessor:
    def __init__(self):
        try:
            # Test connection to Ollama
            self.client = ollama.Client(host=OLLAMA_BASE_URL)
            # We don't pull here to avoid startup delay, assuming user has it
            logger.info(f"Connected to Ollama at {OLLAMA_BASE_URL}")
        except Exception as e:
            logger.error(f"Could not connect to Ollama: {e}")
            self.client = None
        
        with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()

    def normalize(self, query: str) -> Dict[str, Any]:
        if not self.client:
            return {"rewritten_query": query, "language": "unknown", "sector_guess": "unknown"}

        try:
            response = self.client.generate(
                model=LLM_MODEL_NAME,
                prompt=f"{self.system_prompt}\n\nUser Input: \"{query}\"",
                format='json',
                stream=False
            )
            return json.loads(response['response'])
        except Exception as e:
            logger.error(f"Error normalizing query with Ollama: {e}")
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

        # LOGGING FOR DEBUGGING
        print(f"\n--- DEBUG SEARCH ---")
        print(f"Original: '{query}'")
        print(f"Rewritten: '{rewritten}' (Lang: {detected_lang})")

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
        
        print("Candidates:")
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1: continue
            
            meta = self.metadata[idx]
            match_score = float(dist)
            print(f" - [{match_score:.4f}] {meta.get('question')} (Sector: {meta.get('sector')})")

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
                # Break early if we found a good match? usually we want the best one, sorted by distance.
                # FAISS returns sorted by distance (descending for IP/Cosine Sim usually)
                if not best_match: # actually this logic is flawed if we want the BEST match above threshold. 
                    pass # We should just iterate and take the first valid one.
        
        # Correct logic: iterate and find first valid one
        final_result = None
        for dist, idx in zip(distances[0], indices[0]):
             if idx == -1: continue
             match_score = float(dist)
             if match_score < CONFIDENCE_THRESHOLD: continue
             
             meta = self.metadata[idx]
             if sector and meta.get("sector") != sector: continue
             if language and meta.get("language") != language: continue
             
             final_result = {
                    "answer": meta["answer"],
                    "confidence": match_score,
                    "sector": meta["sector"],
                    "language": meta["language"],
                    "source_file": meta.get("source_file"),
             }
             break # Take the highest scoring valid match

        print(f"Selected: {final_result['answer'] if final_result else 'None'}\n--------------------\n")

        if final_result:
            return {**final_result, "rewritten_query": rewritten}
            
        return {
            "answer": "I don't have enough information to answer that specifically. Please contact the regional office directly.",
            "rewritten_query": rewritten,
            "confidence": 0.0,
            "sector": detected_sector,
            "language": detected_lang
        }

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

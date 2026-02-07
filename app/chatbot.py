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
from langdetect import detect, DetectorFactory

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
DetectorFactory.seed = 0 # Deterministic language detection

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
        # 1. Detect language reliably using library (better than 1B LLM)
        try:
            lang_code = detect(query)
            # Map specific codes if needed, e.g. 'so'->'om' if it confuses them, but 'en'/'am' are usually good.
            # langdetect supports 'en', 'am', 'so' (Oromo often detects as Somali or Afar in simple classifiers)
            if lang_code == 'so' or lang_code == 'om': 
                detected_lang_name = "Afaan Oromo"
                lang_code_simple = "om"
            elif lang_code == 'am':
                detected_lang_name = "Amharic"
                lang_code_simple = "am"
            else:
                detected_lang_name = "English"
                lang_code_simple = "en"
        except:
            detected_lang_name = "English"
            lang_code_simple = "en"

        if not self.client:
            return {
                "rewritten_query": query, 
                "language": detected_lang_name, 
                "language_code": lang_code_simple,
                "sector_guess": "unknown"
            }

        try:
            # We ask LLM ONLY to rewrite/clarify, passing the detected language to help it.
            prompt = f"""Rewrite the following user query to be clear and concise for a search engine. 
Keep the SAME language as the input ({detected_lang_name}). 
Return result as JSON: {{"rewritten_query": "string"}}

User Input: "{query}" """
            
            response = self.client.generate(
                model=LLM_MODEL_NAME,
                prompt=prompt,
                format='json',
                stream=False
            )
            result = json.loads(response['response'])
            
            return {
                "rewritten_query": result.get("rewritten_query", query),
                "language": detected_lang_name,
                "language_code": lang_code_simple,
                "sector_guess": "general" # Simplified for now
            }
        except Exception as e:
            logger.error(f"Error normalizing query with Ollama: {e}")
            return {
                "rewritten_query": query, 
                "language": detected_lang_name,
                "language_code": lang_code_simple, 
                "sector_guess": "unknown"
            }

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

    def generate_rag_response(self, query: str, context: list[dict], language: str) -> str:
        """Generate a natural language response using retrieved context."""
        # Clean context tokens
        context_text = "\n".join([f"- {c['answer'].replace('(DEMO)', '').strip()}" for c in context])
        
        prompt = f"""You are a helpful government assistant.
Answer the question based ONLY on the facts below.
If the answer is not in the facts, say "I don't know".
Respond in {language}.

Facts:
{context_text}

Question: {query}
Answer:"""

        try:
            response = self.processor.client.generate(
                model=LLM_MODEL_NAME,
                prompt=prompt,
                stream=False
            )
            return response['response']
        except Exception as e:
            logger.error(f"RAG generation failed: {e}")
            return "System error during answer generation."


    def search(self, query: str, sector: Optional[str] = None, language: Optional[str] = None) -> Dict[str, Any]:
        # 1. Normalize query
        norm_result = self.processor.normalize(query)
        rewritten = norm_result.get("rewritten_query", query)
        detected_lang = norm_result.get("language", "English")
        detected_code = norm_result.get("language_code", "en") 
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
        
        # 4. Collect Top Matches (RAG)
        retrieved_context = []
        
        print("Candidates:")
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1: continue
            
            meta = self.metadata[idx]
            match_score = float(dist)
            print(f" - [{match_score:.4f}] {meta.get('question')} (Sector: {meta.get('sector')})")

            # Relaxed filtering: Only strict if score is high, otherwise specific language might be needed
            if sector and meta.get("sector") != sector: continue
            
            # NOTE: For RAG, we might WANT cross-language facts if we can translate, 
            # but for now let's stick to the user's filtered language or detected language
            # if language and meta.get("language") != language: continue 

            # Lower threshold slightly for RAG context retrieval
            if match_score >= 0.35: # Lowered from 0.55 to capture more context
                retrieved_context.append(meta)

        # 5. Generate Response
        if not retrieved_context:
            return {
                "answer": "I don't have enough information to answer that specifically. Please contact the regional office directly.",
                "rewritten_query": rewritten,
                "confidence": 0.0,
                "sector": detected_sector,
                "language": detected_lang
            }
        
        # Call LLM to synthesize answer
        final_answer = self.generate_rag_response(
            query=query, # Original query has the tone/nuance
            context=retrieved_context,
            language=detected_lang
        )

        print(f"Generated RAG Answer: {final_answer}\n--------------------\n")

        return {
            "answer": final_answer,
            "rewritten_query": rewritten,
            "confidence": 1.0, # RAG confidence is synthetic
            "sector": retrieved_context[0]['sector'], # Primary source sector
            "language": detected_lang,
            "source_file": "generated_rag"
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

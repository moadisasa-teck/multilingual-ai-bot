import os
import re
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
        # 1. Detect language
        detected_lang_name = "English"
        lang_code_simple = "en"

        try:
            # Check for Ethiopic characters first (Amharic)
            if re.search(r'[\u1200-\u137F]', query):
                detected_lang_name = "Amharic"
                lang_code_simple = "am"
            else:
                lang_code = detect(query)
                # Map common misclassifications for Afaan Oromo (Oromo uses Latin script similar to Somali/Finnish/Swahili)
                if lang_code in ['om', 'so', 'fi', 'sw']: 
                    detected_lang_name = "Afaan Oromo"
                    lang_code_simple = "om"
                elif lang_code == 'am':
                    detected_lang_name = "Amharic"
                    lang_code_simple = "am"
                else:
                    detected_lang_name = "English"
                    lang_code_simple = "en"
        except Exception:
            # Fallback for "No features in text" or other errors
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

    def generate_rag_response(self, query: str, context: list[dict], language: str, sector: str = "general") -> str:
        """Generate a natural language response using retrieved context."""
        # Clean context tokens
        if not context:
            context_text = "No specific facts found."
        else:
            context_text = "\n".join([f"- {c['answer'].replace('(DEMO)', '').strip()}" for c in context])
        
        prompt = f"""You are a helpful assistant for the Oromia Regional Government ({sector} sector).
Your knowledge is STRICTLY LIMITED to government services.

Instructions:
1. Respond in {language}.
2. FORMATTING IS CRITICAL:
   - Use Markdown for structure.
   - Use clear distinct paragraphs with blank lines between them.
   - Use bullet points (-) for lists or steps.
   - Use bold (**) for headings or key terms.
   - Never output a single large block of text. Break it up.
3. If the user uses a GREETING (hi, hello, thanks), respond politely.
4. For ALL OTHER questions, you must answer based ONLY on the "Context Facts" provided below.
5. If the answer is not in the facts, say "I can only answer questions about Oromia government services."

Context Facts:
{context_text}

User Question: {query}
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
        print(f"DEBUG: search called with query='{query}', sector='{sector}', language='{language}'")
        # 1. Normalize query
        norm_result = self.processor.normalize(query)
        rewritten = norm_result.get("rewritten_query", query)
        detected_lang = norm_result.get("language", "English")
        detected_code = norm_result.get("language_code", "en") 
        detected_sector = norm_result.get("sector_guess", "unknown")

        # Determine target response language (User Preference > Detected)
        target_language = detected_lang
        if language:
            lang_map = {
                "en": "English",
                "am": "Amharic",
                "om": "Afaan Oromo"
            }
            # If language is a code like 'en', map it. If it's already full name, keep it.
            target_language = lang_map.get(language.lower(), language)

        # LOGGING FOR DEBUGGING
        print(f"\n--- DEBUG SEARCH ---")
        print(f"Original: '{query}'")
        print(f"Rewritten: '{rewritten}' (Detected: {detected_lang}) -> Responding in: {target_language}")

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

            # Relaxed filtering: Allow matches from requested sector OR 'general' (for greetings)
            if sector and meta.get("sector") != sector and meta.get("sector") != "general": continue
            
            # NOTE: For RAG, we might WANT cross-language facts if we can translate, 
            # but for now let's stick to the user's filtered language or detected language
            # if language and meta.get("language") != language: continue 

            # Lower threshold slightly for RAG context retrieval
            if match_score >= 0.35: # Lowered from 0.55 to capture more context
                retrieved_context.append(meta)

        # 5. Generate Response (Updated logic: Allow partial conversation even if context is empty)
        # Call LLM to synthesize answer
        final_answer = self.generate_rag_response(
            query=query, # Original query has the tone/nuance
            context=retrieved_context, # Might be empty, prompt must handle it
            language=target_language,
            sector=sector or detected_sector
        )

        print(f"Generated RAG Answer: {final_answer}\n--------------------\n")
        
        # Determine source sector
        primary_sector = "general"
        if retrieved_context:
            primary_sector = retrieved_context[0]['sector']
        elif sector:
            primary_sector = sector
        else:
            primary_sector = detected_sector

        return {
            "answer": final_answer,
            "rewritten_query": rewritten,
            "confidence": 1.0 if retrieved_context else 0.5, # Lower confidence if no context found
            "sector": primary_sector,
            "language": target_language,
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

import re
import json
import logging
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
    CONFIDENCE_THRESHOLD,
    RESPONSE_MODE,
    LLM_REWRITE_LANGUAGES,
    RETRIEVAL_CANDIDATES,
    MAX_SUBQUERIES,
    MAX_FINAL_ANSWERS,
)

load_dotenv()
DetectorFactory.seed = 0

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LANG_CODE_TO_NAME = {
    "en": "English",
    "am": "Amharic",
    "om": "Afaan Oromo",
}

LANG_NAME_TO_CODE = {
    "english": "en",
    "amharic": "am",
    "afaan oromo": "om",
    "oromo": "om",
}

FALLBACK_MESSAGES = {
    "en": "I can only answer questions about Oromia government services. Please rephrase your question.",
    "om": "Tajaajiloota mootummaa Naannoo Oromiyaa qofa irratti deebii kennuu nan danda'a. Maaloo gaaffii kee irra deebi'ii ifa godhi.",
    "am": "ስለ ኦሮሚያ ክልል የመንግስት አገልግሎቶች ብቻ መልስ መስጠት እችላለሁ። እባክዎ ጥያቄዎን እንደገና ያብራሩ።",
}

UNAVAILABLE_MESSAGES = {
    "en": "System is currently unavailable. Please try again later.",
    "om": "Sirni yeroo ammaa hin hojjetu. Maaloo yeroo biraa irra deebi'ii yaali.",
    "am": "ስርዓቱ በአሁኑ ጊዜ አይገኝም። እባክዎ ቆይተው እንደገና ይሞክሩ።",
}

SPLIT_PATTERNS = {
    "en": r"\b(?:and|also|plus)\b|[;?]+",
    "om": r"\b(?:fi|akkasumas)\b|[;?]+",
    "am": r"(?:እና|እንዲሁም|[፣፤፧])",
}

STOPWORDS = {
    "en": {"the", "a", "an", "is", "are", "to", "and", "or", "in", "on", "for", "of", "how", "what", "can", "do", "does", "i", "my", "it"},
    "om": {"fi", "yoo", "ni", "maal", "kan", "akka", "itti", "irratti", "gara", "kee", "na", "si", "isaan", "eeyyee", "miti"},
    "am": {"እና", "እንዴት", "ምን", "ነው", "የ", "በ", "ላይ"},
}

MULTI_INTRO_MESSAGES = {
    "en": "I found the following details:",
    "om": "Odeeffannoon armaan gadii argameera:",
    "am": "የተገኘው መረጃ ይህ ነው፦",
}

CONCISE_PREFIX_MESSAGES = {
    "en": "Here is what I found:",
    "om": "Kunoo odeeffannoo argadhe:",
    "am": "ያገኘሁት መረጃ ይህ ነው፦",
}


class QueryProcessor:
    def __init__(self):
        try:
            self.client = ollama.Client(host=OLLAMA_BASE_URL)
            logger.info(f"Connected to Ollama at {OLLAMA_BASE_URL}")
        except Exception as e:
            logger.error(f"Could not connect to Ollama: {e}")
            self.client = None

        self.system_prompt = ""
        if SYSTEM_PROMPT_PATH.exists():
            with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
                self.system_prompt = f.read()

    def clean_query(self, query: str) -> str:
        return " ".join(query.strip().split())

    def detect_language(self, query: str) -> Dict[str, str]:
        detected_lang_name = "English"
        lang_code_simple = "en"
        try:
            if re.search(r'[\u1200-\u137F]', query):
                detected_lang_name = "Amharic"
                lang_code_simple = "am"
            else:
                lang_code = detect(query)
                if lang_code in ["om", "so", "fi", "sw"]:
                    detected_lang_name = "Afaan Oromo"
                    lang_code_simple = "om"
                elif lang_code == "am":
                    detected_lang_name = "Amharic"
                    lang_code_simple = "am"
        except Exception:
            detected_lang_name = "English"
            lang_code_simple = "en"

        return {
            "language": detected_lang_name,
            "language_code": lang_code_simple,
        }

    def rewrite_query(self, query: str, language_name: str) -> str:
        if not self.client:
            return query

        try:
            prompt = f"""Rewrite the following user query to be clear and concise for a search engine. 
Keep the SAME language as the input ({language_name}). 
Return result as JSON: {{"rewritten_query": "string"}}

User Input: "{query}" """

            response = self.client.generate(
                model=LLM_MODEL_NAME,
                prompt=prompt,
                format="json",
                stream=False
            )
            result = json.loads(response["response"])
            rewritten = self.clean_query(result.get("rewritten_query", query))
            return rewritten or query
        except Exception as e:
            logger.error(f"Error rewriting query with Ollama: {e}")
            return query


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

    def _resolve_target_language_code(self, requested_language: Optional[str], detected_code: str) -> str:
        if not requested_language:
            return detected_code
        normalized = requested_language.strip().lower()
        if normalized in LANG_CODE_TO_NAME:
            return normalized
        return LANG_NAME_TO_CODE.get(normalized, detected_code)

    def _normalize_vector(self, v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def _build_query_variants(self, query: str, lang_code: str, history: Optional[list[dict]] = None) -> list[str]:
        segments = self._split_multi_intent(query, lang_code)
        variants = [query]
        for segment in segments:
            if segment and segment not in variants:
                variants.append(segment)
            if len(variants) - 1 >= MAX_SUBQUERIES:
                break

        for ctx_query in self._history_query_variants(history, lang_code):
            if ctx_query not in variants:
                variants.append(ctx_query)
        return variants

    def _history_query_variants(self, history: Optional[list[dict]], lang_code: str) -> list[str]:
        if not history:
            return []

        user_messages: list[str] = []
        for item in reversed(history):
            role = str(item.get("role", "")).lower()
            content = self.processor.clean_query(str(item.get("content", "")))
            if role != "user" or not content:
                continue
            user_messages.append(content)
            if len(user_messages) >= 2:
                break

        user_messages = list(reversed(user_messages))
        if not user_messages:
            return []

        variants: list[str] = []
        for msg in user_messages:
            if self.processor.detect_language(msg).get("language_code") == lang_code:
                variants.append(msg)
        return variants

    def _split_multi_intent(self, query: str, lang_code: str) -> list[str]:
        pattern = SPLIT_PATTERNS.get(lang_code)
        if not pattern:
            return []
        flags = re.IGNORECASE if lang_code in {"en", "om"} else 0
        parts = [self.processor.clean_query(p) for p in re.split(pattern, query, flags=flags)]
        return [p for p in parts if p]

    def _fallback_message(self, lang_code: str) -> str:
        return FALLBACK_MESSAGES.get(lang_code, FALLBACK_MESSAGES["en"])

    def _unavailable_message(self, lang_code: str) -> str:
        return UNAVAILABLE_MESSAGES.get(lang_code, UNAVAILABLE_MESSAGES["en"])

    def _clean_answer(self, answer: str) -> str:
        return answer.replace("(DEMO)", "").strip()

    def _word_count(self, text: str, lang_code: str) -> int:
        if lang_code == "am":
            return len(re.findall(r"[\u1200-\u137F]+", text))
        return len(re.findall(r"[A-Za-z']+", text))

    def _compose_multi_text(self, answers: list[str], lang_code: str) -> tuple[str, str]:
        concise_prefix = CONCISE_PREFIX_MESSAGES.get(lang_code, CONCISE_PREFIX_MESSAGES["en"])
        multi_intro = MULTI_INTRO_MESSAGES.get(lang_code, MULTI_INTRO_MESSAGES["en"])

        if len(answers) <= 2:
            total_words = sum(self._word_count(answer, lang_code) for answer in answers)
            max_words = max(self._word_count(answer, lang_code) for answer in answers)
            if total_words <= 36 and max_words <= 22:
                return f"{concise_prefix} {' '.join(answers)}", "multi_concise"

        bullet_response = "\n".join([f"- {answer}" for answer in answers])
        return f"{multi_intro}\n\n{bullet_response}", "multi_source"

    def _tokenize_for_overlap(self, text: str, lang_code: str) -> set[str]:
        if lang_code == "am":
            tokens = [token for token in re.findall(r"[\u1200-\u137F]+", text)]
        else:
            tokens = [token.lower() for token in re.findall(r"[a-zA-Z']+", text)]

        if lang_code in {"en", "om"}:
            tokens = [token.lower() for token in tokens]

        stopwords = STOPWORDS.get(lang_code, set())
        return {token for token in tokens if token and len(token) > 1 and token not in stopwords}

    def _has_token_overlap(self, query_text: str, question_text: str, lang_code: str) -> bool:
        query_tokens = self._tokenize_for_overlap(query_text, lang_code)
        if not query_tokens:
            return True
        question_tokens = self._tokenize_for_overlap(question_text, lang_code)
        if not question_tokens:
            return False
        return bool(query_tokens & question_tokens)

    def _collect_ranked_hits(self, query_variants: list[str], requested_sector: Optional[str], target_lang_code: str) -> list[dict]:
        best_by_id: dict[Any, dict] = {}

        for query_text in query_variants:
            query_vec = self.embedding_model.encode([query_text])[0]
            query_vec = self._normalize_vector(query_vec.astype("float32"))
            distances, indices = self.index.search(np.array([query_vec]), RETRIEVAL_CANDIDATES)

            for score, idx in zip(distances[0], indices[0]):
                if idx == -1:
                    continue
                score_val = float(score)
                if score_val < CONFIDENCE_THRESHOLD:
                    continue

                meta = self.metadata[idx]
                if meta.get("language") != target_lang_code:
                    continue

                meta_sector = meta.get("sector")
                if requested_sector and meta_sector not in {requested_sector, "general"}:
                    continue
                if not self._has_token_overlap(query_text, meta.get("question", ""), target_lang_code):
                    continue

                record_id = meta.get("id", idx)
                current = best_by_id.get(record_id)
                if not current or score_val > current["score"]:
                    best_by_id[record_id] = {"score": score_val, "meta": meta}

        ranked = sorted(best_by_id.values(), key=lambda item: item["score"], reverse=True)
        return ranked[:MAX_FINAL_ANSWERS]

    def _compose_extractive_answer(self, ranked_hits: list[dict], target_lang_code: str) -> tuple[str, float, str, Optional[str]]:
        if not ranked_hits:
            return "", 0.0, "unknown", "fallback"

        answers: list[str] = []
        for hit in ranked_hits:
            cleaned = self._clean_answer(hit["meta"].get("answer", ""))
            if cleaned and cleaned not in answers:
                answers.append(cleaned)

        if not answers:
            return "", 0.0, "unknown", "fallback"

        top_meta = ranked_hits[0]["meta"]
        top_score = ranked_hits[0]["score"]
        primary_sector = top_meta.get("sector", "unknown")

        if len(answers) == 1:
            return answers[0], top_score, primary_sector, top_meta.get("source_file")

        formatted, source_file = self._compose_multi_text(answers, target_lang_code)
        return formatted, top_score, primary_sector, source_file

    def generate_rag_response(self, query: str, context: list[dict], language: str, sector: str = "general") -> str:
        if not self.processor.client:
            return self._fallback_message("en")

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

    def search(
        self,
        query: str,
        sector: Optional[str] = None,
        language: Optional[str] = None,
        history: Optional[list[dict]] = None,
    ) -> Dict[str, Any]:
        cleaned_query = self.processor.clean_query(query)
        detected = self.processor.detect_language(cleaned_query)
        detected_lang_code = detected["language_code"]
        target_lang_code = self._resolve_target_language_code(language, detected_lang_code)
        target_language = LANG_CODE_TO_NAME.get(target_lang_code, detected.get("language", "English"))
        use_llm_rewrite = target_lang_code in LLM_REWRITE_LANGUAGES and self.processor.client is not None
        rewritten = cleaned_query
        if use_llm_rewrite:
            rewritten = self.processor.rewrite_query(cleaned_query, target_language)

        if rewritten.lower() == "unclear":
            return {
                "answer": self._fallback_message(target_lang_code),
                "rewritten_query": rewritten,
                "confidence": 0.0,
                "sector": sector or "unknown",
                "language": target_language,
                "source_file": "fallback",
            }

        if not self.index:
            return {
                "answer": self._unavailable_message(target_lang_code),
                "rewritten_query": rewritten,
                "confidence": 0.0,
                "sector": sector or "unknown",
                "language": target_language,
                "source_file": "system_unavailable",
            }

        query_variants = self._build_query_variants(rewritten, target_lang_code, history=history)
        ranked_hits = self._collect_ranked_hits(query_variants, sector, target_lang_code)

        if RESPONSE_MODE == "generative":
            context = [hit["meta"] for hit in ranked_hits]
            final_answer = self.generate_rag_response(
                query=query,
                context=context,
                language=target_language,
                sector=sector or (context[0]["sector"] if context else "general"),
            )
            return {
                "answer": final_answer,
                "rewritten_query": rewritten,
                "confidence": ranked_hits[0]["score"] if ranked_hits else 0.0,
                "sector": context[0]["sector"] if context else (sector or "unknown"),
                "language": target_language,
                "source_file": "generated_rag",
            }

        answer, confidence, primary_sector, source_file = self._compose_extractive_answer(ranked_hits, target_lang_code)
        if not answer:
            return {
                "answer": self._fallback_message(target_lang_code),
                "rewritten_query": rewritten,
                "confidence": 0.0,
                "sector": sector or "unknown",
                "language": target_language,
                "source_file": "fallback",
            }

        return {
            "answer": answer,
            "rewritten_query": rewritten,
            "confidence": confidence,
            "sector": primary_sector,
            "language": target_language,
            "source_file": source_file,
        }

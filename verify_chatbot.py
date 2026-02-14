import os
import sys

# Ensure project root is in PYTHONPATH
sys.path.append(os.getcwd())
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.chatbot import Chatbot


def verify_chatbot() -> None:
    bot = Chatbot()

    test_cases = [
        {
            "label": "Oromo multi-intent (passport docs + time)",
            "query": "Paaspoortii gaafachuuf sanadoonni maalii barbaachisu fi paaspoortii argachuuf yeroo hangamii fudhata?",
            "language": "om",
            "sector": "passport",
        },
        {
            "label": "Amharic multi-intent (lost + damaged passport)",
            "query": "ፓስፖርት ቢጠፋ ምን ላድርግ እና ቢበላሽ ምን ይደረጋል?",
            "language": "am",
            "sector": "passport",
        },
        {
            "label": "English deterministic multi-answer",
            "query": "What documents are required and how long does it take to get a passport?",
            "language": "en",
            "sector": "passport",
        },
        {
            "label": "No-match fallback in Oromo",
            "query": "Qilleensa Marsi fi satalaayitii irratti na gorsi",
            "language": "om",
            "sector": "passport",
        },
    ]

    print("\n--- Deterministic Chatbot Verification ---")
    print(f"Ollama connected: {bot.processor.client is not None}")

    for case in test_cases:
        print(f"\nCase: {case['label']}")
        print(f"User: {case['query']}")
        result = bot.search(
            query=case["query"],
            language=case["language"],
            sector=case["sector"],
        )
        print(f"Rewritten: {result.get('rewritten_query')}")
        print(f"Confidence: {result.get('confidence'):.4f}")
        print(f"Sector: {result.get('sector')}")
        print(f"Language: {result.get('language')}")
        print(f"Source: {result.get('source_file')}")
        print(f"Answer: {result.get('answer')}")


if __name__ == "__main__":
    verify_chatbot()

import os
import sys

# Ensure projects root is in PYTHONPATH
sys.path.append(os.getcwd())

from app.chatbot import Chatbot

def test_chatbot():
    bot = Chatbot()
    
    test_queries = [
        "Hojiiwwan mana fincaanii keessatti raawwataman kam fa’a?",
        "የመኖሪያ ቤት ፍቃድ ለማግኘት ምን ያስፈልጋል?",
        "unclear gibberish text"
    ]
    
    print("\n--- Chatbot Verification ---")
    if not bot.processor.model:
        print("NOTE: LLM normalization is SKIPPED (no API key). Using raw queries for search.")
    
    for q in test_queries:
        print(f"\nUser: {q}")
        result = bot.search(q)
        print(f"Rewritten: {result.get('rewritten_query')}")
        print(f"Confidence: {result.get('confidence'):.4f}")
        print(f"Sector: {result.get('sector')}")
        print(f"Language: {result.get('language')}")
        print(f"Answer: {result.get('answer')}")

if __name__ == "__main__":
    test_chatbot()

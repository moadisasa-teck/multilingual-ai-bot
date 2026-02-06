# Oromia Government AI Chatbot (Demo)

A multilingual demo chatbot for Oromia regional government services.
Supports Afaan Oromo (om), Amharic (am), and English (en).

## Architecture (demo-safe)

User question → multilingual embedding → FAISS vector search → stored answer

No training, no fine-tuning, no generation beyond stored answers.

## Folder structure

```oromia-gov-ai-bot/
├── data/
│   ├── raw/ (CSV per sector + language)
│   └── processed/ (merged JSON)
├── training/ (data prep + indexing)
├── index/ (FAISS index + metadata)
├── app/ (FastAPI API)
├── config/ (settings)
└── README.md```

## CSV schema (exact)

Columns: sector, language, question, answer

Demo answers must include the marker "DEMO".

## Steps

1) Fill raw CSVs under data/raw/ with demo-safe Q/A.
2) Run training/prepare_data.py to validate and merge to data/processed/qa_all.json.
3) Run training/embed_and_index.py to build the FAISS index in index/.

## Safeguards (demo-safe)

- Sector filtering uses stored metadata only.
- Confidence threshold (see config/settings.py) triggers a fallback response.
- No generation beyond stored answers.

## Why this approach

- Works for low-resource languages via multilingual embeddings.
- No model training or fine-tuning (lower risk, faster setup).
- Retrieval-only responses reduce hallucinations.
- Easy to replace demo data with official data later.

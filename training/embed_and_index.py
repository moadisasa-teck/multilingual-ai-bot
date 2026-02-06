"""Embed questions and build a FAISS index (CPU-friendly)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import (
    FAISS_INDEX_PATH,
    METADATA_PATH,
    MODEL_NAME,
    QA_JSON_PATH,
    SUPPORTED_LANGUAGES,
    SUPPORTED_SECTORS,
)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


def _load_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _filter_records(records: list[dict], sectors: list[str], languages: list[str]) -> list[dict]:
    return [
        r
        for r in records
        if r.get("sector") in sectors and r.get("language") in languages
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed QA data and build FAISS index")
    parser.add_argument(
        "--data",
        type=Path,
        default=QA_JSON_PATH,
        help="Input JSON dataset path",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=FAISS_INDEX_PATH,
        help="FAISS index output path",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=METADATA_PATH,
        help="Metadata output path",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help="SentenceTransformers model name",
    )
    parser.add_argument(
        "--sectors",
        nargs="*",
        default=SUPPORTED_SECTORS,
        help="Filter by sector",
    )
    parser.add_argument(
        "--languages",
        nargs="*",
        default=SUPPORTED_LANGUAGES,
        help="Filter by language",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = _load_records(args.data)
    filtered = _filter_records(records, args.sectors, args.languages)

    if not filtered:
        raise ValueError("No records after filtering. Check sectors/languages.")

    model = SentenceTransformer(args.model, device="cpu")
    questions = [r["question"] for r in filtered]
    embeddings = model.encode(questions, batch_size=32, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype="float32")
    embeddings = _normalize(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    args.index_path.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(args.index_path))
    with args.metadata_path.open("w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(f"Indexed {len(filtered)} records")
    print(f"FAISS index: {args.index_path}")
    print(f"Metadata: {args.metadata_path}")


if __name__ == "__main__":
    main()

"""Prepare demo data by validating CSVs and merging into JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from config.settings import (
    DEMO_MARKER,
    DEMO_REQUIRED,
    QA_JSON_PATH,
    RAW_DATA_DIR,
    SUPPORTED_LANGUAGES,
    SUPPORTED_SECTORS,
)

REQUIRED_COLUMNS = ["sector", "language", "question", "answer"]


def _validate_columns(df: pd.DataFrame, file_path: Path) -> None:
    if list(df.columns) != REQUIRED_COLUMNS:
        raise ValueError(
            f"Invalid columns in {file_path}. Expected {REQUIRED_COLUMNS}, got {list(df.columns)}"
        )


def _validate_rows(df: pd.DataFrame, file_path: Path, allow_non_demo: bool) -> None:
    if df.empty:
        raise ValueError(f"No rows found in {file_path}")

    for idx, row in df.iterrows():
        sector = str(row.get("sector", "")).strip()
        language = str(row.get("language", "")).strip()
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()

        if sector not in SUPPORTED_SECTORS:
            raise ValueError(f"Invalid sector '{sector}' in {file_path} (row {idx + 1})")
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Invalid language '{language}' in {file_path} (row {idx + 1})")
        if not question or not answer:
            raise ValueError(f"Empty question/answer in {file_path} (row {idx + 1})")

        if DEMO_REQUIRED and not allow_non_demo:
            if DEMO_MARKER.lower() not in answer.lower():
                raise ValueError(
                    f"Answer must include '{DEMO_MARKER}' marker in {file_path} (row {idx + 1})"
                )


def load_csvs(raw_dir: Path, allow_non_demo: bool) -> list[dict]:
    records: list[dict] = []

    for csv_path in sorted(raw_dir.rglob("*.csv")):
        df = pd.read_csv(csv_path)
        _validate_columns(df, csv_path)
        _validate_rows(df, csv_path, allow_non_demo)

        for _, row in df.iterrows():
            records.append(
                {
                    "sector": str(row["sector"]).strip(),
                    "language": str(row["language"]).strip(),
                    "question": str(row["question"]).strip(),
                    "answer": str(row["answer"]).strip(),
                    "source_file": str(csv_path.relative_to(raw_dir)),
                }
            )

    if not records:
        raise ValueError(f"No CSV records found under {raw_dir}")

    for idx, record in enumerate(records, start=1):
        record["id"] = idx

    return records


def write_json(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate CSVs and merge to JSON")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="Directory containing raw CSV files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=QA_JSON_PATH,
        help="Output JSON path",
    )
    parser.add_argument(
        "--allow-non-demo",
        action="store_true",
        help="Allow answers without the DEMO marker",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_csvs(args.raw_dir, args.allow_non_demo)
    write_json(records, args.output)
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()

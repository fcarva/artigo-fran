#!/usr/bin/env python3
"""Gera saídas de mapa de palavras para análise estatística."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


PT_STOPWORDS = {
    "a", "à", "ao", "aos", "as", "às", "com", "como", "da", "das", "de", "dela", "dele",
    "deles", "demais", "depois", "do", "dos", "e", "é", "ela", "ele", "eles", "em", "entre",
    "era", "eram", "essa", "esse", "esta", "este", "eu", "foi", "foram", "há", "isso", "isto",
    "já", "lhe", "mais", "mas", "me", "mesmo", "meu", "minha", "muito", "na", "nas", "nem",
    "no", "nos", "nós", "o", "os", "ou", "para", "pela", "pelas", "pelo", "pelos", "por",
    "qual", "que", "quem", "se", "sem", "seu", "sua", "são", "sob", "sobre", "também", "te",
    "tem", "tendo", "tenho", "ter", "teu", "tua", "um", "uma", "você", "vocês",
}

EN_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is",
    "it", "its", "of", "on", "that", "the", "to", "was", "were", "will", "with",
}

WORD_PATTERN = re.compile(r"[a-zA-ZÀ-ÿ]+", flags=re.UNICODE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera outputs de mapa de palavras para análise estatística.")
    parser.add_argument("--input", required=True, help="Caminho para o CSV com os artigos")
    parser.add_argument("--output-dir", required=True, help="Diretório onde os arquivos de saída serão salvos")
    parser.add_argument("--delimiter", default=",", help="Delimitador do CSV (padrão: ,)")
    parser.add_argument("--id-column", default="id", help="Nome da coluna de ID")
    parser.add_argument("--text-column", default="text", help="Nome da coluna de texto")
    parser.add_argument("--language", choices=["pt", "en"], default="pt", help="Idioma para stopwords")
    parser.add_argument("--top-words", type=int, default=300, help="Número de palavras para a matriz documento x palavra")
    parser.add_argument("--min-word-length", type=int, default=3, help="Tamanho mínimo das palavras")
    parser.add_argument("--stopwords-file", help="Arquivo de stopwords customizadas (uma por linha)")
    return parser.parse_args()


def load_stopwords(language: str, stopwords_file: str | None) -> set[str]:
    stopwords = set(PT_STOPWORDS if language == "pt" else EN_STOPWORDS)
    if stopwords_file:
        with open(stopwords_file, "r", encoding="utf-8") as file:
            custom = {line.strip().lower() for line in file if line.strip()}
            stopwords.update(custom)
    return stopwords


def tokenize(text: str, stopwords: set[str], min_word_length: int) -> list[str]:
    tokens = [match.group(0).lower() for match in WORD_PATTERN.finditer(text)]
    return [t for t in tokens if len(t) >= min_word_length and t not in stopwords]


def read_articles(
    input_file: Path,
    delimiter: str,
    id_column: str,
    text_column: str,
    stopwords: set[str],
    min_word_length: int,
) -> tuple[list[str], list[list[str]]]:
    ids: list[str] = []
    docs_tokens: list[list[str]] = []

    with input_file.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=delimiter)
        missing = [col for col in [id_column, text_column] if col not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Colunas ausentes no CSV: {', '.join(missing)}")

        for row in reader:
            doc_id = str(row[id_column]).strip()
            text = (row[text_column] or "").strip()
            ids.append(doc_id)
            docs_tokens.append(tokenize(text, stopwords, min_word_length))

    return ids, docs_tokens


def write_word_frequency(output_file: Path, corpus_counter: Counter[str], total_tokens: int) -> None:
    with output_file.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["word", "frequency", "relative_frequency"])
        for word, freq in corpus_counter.most_common():
            rel = freq / total_tokens if total_tokens else 0
            writer.writerow([word, freq, f"{rel:.8f}"])


def write_document_word_matrix(
    output_file: Path,
    doc_ids: list[str],
    docs_tokens: list[list[str]],
    top_words: Iterable[str],
) -> None:
    selected_words = list(top_words)
    with output_file.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", *selected_words])

        for doc_id, tokens in zip(doc_ids, docs_tokens):
            row_counter = Counter(tokens)
            writer.writerow([doc_id, *[row_counter.get(word, 0) for word in selected_words]])


def write_summary(
    output_file: Path,
    num_docs: int,
    total_tokens: int,
    vocabulary_size: int,
    avg_tokens_per_doc: float,
    top_words: list[tuple[str, int]],
) -> None:
    summary = {
        "documents": num_docs,
        "total_tokens": total_tokens,
        "vocabulary_size": vocabulary_size,
        "avg_tokens_per_document": round(avg_tokens_per_doc, 4),
        "top_20_words": [{"word": w, "frequency": f} for w, f in top_words],
    }
    output_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stopwords = load_stopwords(args.language, args.stopwords_file)

    doc_ids, docs_tokens = read_articles(
        input_file=Path(args.input),
        delimiter=args.delimiter,
        id_column=args.id_column,
        text_column=args.text_column,
        stopwords=stopwords,
        min_word_length=args.min_word_length,
    )

    corpus_counter: Counter[str] = Counter()
    for tokens in docs_tokens:
        corpus_counter.update(tokens)

    total_tokens = sum(corpus_counter.values())
    top_words = [word for word, _ in corpus_counter.most_common(args.top_words)]

    write_word_frequency(output_dir / "word_frequency.csv", corpus_counter, total_tokens)
    write_document_word_matrix(output_dir / "document_word_matrix.csv", doc_ids, docs_tokens, top_words)
    write_summary(
        output_dir / "word_map_summary.json",
        num_docs=len(doc_ids),
        total_tokens=total_tokens,
        vocabulary_size=len(corpus_counter),
        avg_tokens_per_doc=(total_tokens / len(doc_ids) if doc_ids else 0),
        top_words=corpus_counter.most_common(20),
    )

    print(f"Arquivos gerados em: {output_dir}")


if __name__ == "__main__":
    main()

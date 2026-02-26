#!/usr/bin/env python3
"""Gera saídas de palavras-chave e mapa de palavras para análise estatística."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
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
TEXT_COLUMN_CANDIDATES = ["text", "texto", "conteudo", "conteúdo", "content", "article", "artigo"]
ID_COLUMN_CANDIDATES = ["id", "article_id", "doc_id", "documento_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gera outputs de palavras-chave e mapa de palavras a partir de CSV de artigos. "
            "Útil para análises estatísticas e criação de keyword maps."
        )
    )
    parser.add_argument("--input", required=True, help="Caminho para o CSV com os artigos")
    parser.add_argument("--output-dir", required=True, help="Diretório onde os arquivos de saída serão salvos")
    parser.add_argument("--delimiter", default=",", help="Delimitador do CSV (padrão: ,)")
    parser.add_argument("--id-column", default="", help="Nome da coluna de ID (opcional)")
    parser.add_argument("--text-column", default="", help="Nome da coluna de texto (opcional)")
    parser.add_argument("--language", choices=["pt", "en"], default="pt", help="Idioma para stopwords")
    parser.add_argument("--top-words", type=int, default=300, help="Número de palavras para a matriz documento x palavra")
    parser.add_argument("--keywords-per-article", type=int, default=15, help="Número de palavras-chave por artigo")
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
    return [token for token in tokens if len(token) >= min_word_length and token not in stopwords]


def choose_column(fieldnames: list[str], explicit_column: str, candidates: list[str], label: str) -> str:
    if explicit_column:
        if explicit_column not in fieldnames:
            raise ValueError(f"Coluna '{explicit_column}' não encontrada para {label}.")
        return explicit_column

    lowered_map = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate in lowered_map:
            return lowered_map[candidate]

    raise ValueError(
        f"Não foi possível detectar coluna de {label}. Use --{label}-column para informar explicitamente."
    )


def read_articles(
    input_file: Path,
    delimiter: str,
    id_column: str,
    text_column: str,
    stopwords: set[str],
    min_word_length: int,
) -> tuple[list[str], list[list[str]], str, str]:
    ids: list[str] = []
    docs_tokens: list[list[str]] = []

    with input_file.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=delimiter)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            raise ValueError("CSV sem cabeçalho.")

        selected_text_column = choose_column(fieldnames, text_column, TEXT_COLUMN_CANDIDATES, "text")

        if id_column:
            selected_id_column = choose_column(fieldnames, id_column, ID_COLUMN_CANDIDATES, "id")
        else:
            try:
                selected_id_column = choose_column(fieldnames, "", ID_COLUMN_CANDIDATES, "id")
            except ValueError:
                selected_id_column = ""

        for idx, row in enumerate(reader, start=1):
            doc_id = str(row[selected_id_column]).strip() if selected_id_column else str(idx)
            text = str(row.get(selected_text_column, "") or "").strip()
            ids.append(doc_id)
            docs_tokens.append(tokenize(text, stopwords, min_word_length))

    return ids, docs_tokens, selected_id_column, selected_text_column


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


def compute_tfidf_keywords(docs_tokens: list[list[str]], keywords_per_article: int) -> list[list[tuple[str, float]]]:
    document_frequency: defaultdict[str, int] = defaultdict(int)
    for tokens in docs_tokens:
        for word in set(tokens):
            document_frequency[word] += 1

    total_docs = len(docs_tokens)
    all_keywords: list[list[tuple[str, float]]] = []

    for tokens in docs_tokens:
        token_count = Counter(tokens)
        doc_len = sum(token_count.values())
        scores: list[tuple[str, float]] = []
        for word, freq in token_count.items():
            tf = freq / doc_len if doc_len else 0
            idf = math.log((1 + total_docs) / (1 + document_frequency[word])) + 1
            scores.append((word, tf * idf))

        scores.sort(key=lambda x: x[1], reverse=True)
        all_keywords.append(scores[:keywords_per_article])

    return all_keywords


def write_article_keywords(output_file: Path, doc_ids: list[str], keywords: list[list[tuple[str, float]]]) -> None:
    with output_file.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "rank", "keyword", "score"])
        for doc_id, items in zip(doc_ids, keywords):
            for rank, (word, score) in enumerate(items, start=1):
                writer.writerow([doc_id, rank, word, f"{score:.8f}"])


def write_summary(
    output_file: Path,
    num_docs: int,
    total_tokens: int,
    vocabulary_size: int,
    avg_tokens_per_doc: float,
    top_words: list[tuple[str, int]],
    id_column_used: str,
    text_column_used: str,
) -> None:
    summary = {
        "documents": num_docs,
        "total_tokens": total_tokens,
        "vocabulary_size": vocabulary_size,
        "avg_tokens_per_document": round(avg_tokens_per_doc, 4),
        "id_column_used": id_column_used or "generated_row_number",
        "text_column_used": text_column_used,
        "top_20_words": [{"word": word, "frequency": freq} for word, freq in top_words],
    }
    output_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stopwords = load_stopwords(args.language, args.stopwords_file)

    doc_ids, docs_tokens, id_column_used, text_column_used = read_articles(
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
    article_keywords = compute_tfidf_keywords(docs_tokens, args.keywords_per_article)

    write_word_frequency(output_dir / "word_frequency.csv", corpus_counter, total_tokens)
    write_document_word_matrix(output_dir / "document_word_matrix.csv", doc_ids, docs_tokens, top_words)
    write_article_keywords(output_dir / "article_keywords.csv", doc_ids, article_keywords)
    write_summary(
        output_dir / "word_map_summary.json",
        num_docs=len(doc_ids),
        total_tokens=total_tokens,
        vocabulary_size=len(corpus_counter),
        avg_tokens_per_doc=(total_tokens / len(doc_ids) if doc_ids else 0),
        top_words=corpus_counter.most_common(20),
        id_column_used=id_column_used,
        text_column_used=text_column_used,
    )

    print(f"Arquivos gerados em: {output_dir}")


if __name__ == "__main__":
    main()

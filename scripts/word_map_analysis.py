#!/usr/bin/env python3
"""Gera saídas de palavras-chave e mapa de palavras para análise estatística."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import struct
import unicodedata
import zlib

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
    parser.add_argument(
        "--delimiter",
        default="auto",
        help="Delimitador do CSV. Use ',' ';' '\\t' ou 'auto' para detectar automaticamente (padrão: auto)",
    )
    parser.add_argument("--id-column", default="", help="Nome da coluna de ID (opcional)")
    parser.add_argument("--text-column", default="", help="Nome da coluna de texto (opcional)")
    parser.add_argument("--language", choices=["pt", "en"], default="pt", help="Idioma para stopwords")
    parser.add_argument("--top-words", type=int, default=300, help="Número de palavras para a matriz documento x palavra")
    parser.add_argument("--keywords-per-article", type=int, default=15, help="Número de palavras-chave por artigo")
    parser.add_argument(
        "--cooccurrence-window",
        type=int,
        default=0,
        help="Se > 0, gera coocorrência de palavras dentro da janela por artigo em keyword_cooccurrence.csv",
    )
    parser.add_argument("--min-word-length", type=int, default=3, help="Tamanho mínimo das palavras")
    parser.add_argument("--stopwords-file", help="Arquivo de stopwords customizadas (uma por linha)")
    parser.add_argument("--wordcloud-png", default="wordcloud.png", help="Nome do PNG da nuvem de palavras")
    return parser.parse_args()


def load_stopwords(language: str, stopwords_file: str | None) -> set[str]:
    stopwords = set(PT_STOPWORDS if language == "pt" else EN_STOPWORDS)
    if stopwords_file:
        with open(stopwords_file, "r", encoding="utf-8") as file:
            custom = {line.strip().lower() for line in file if line.strip()}
            stopwords.update(custom)
    return stopwords


def normalize_delimiter(delimiter: str) -> str:
    if delimiter == r"\t":
        return "\t"
    return delimiter


def detect_delimiter(input_file: Path) -> str:
    with input_file.open("r", encoding="utf-8", newline="") as csv_file:
        sample = csv_file.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except csv.Error:
        return ","


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
) -> tuple[list[str], list[list[str]], str, str, str]:
    ids: list[str] = []
    docs_tokens: list[list[str]] = []

    delimiter_used = detect_delimiter(input_file) if delimiter == "auto" else normalize_delimiter(delimiter)

    with input_file.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=delimiter_used)
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
            row_id = str(row[selected_id_column]).strip() if selected_id_column else ""
            doc_id = row_id or str(idx)
            text = str(row.get(selected_text_column, "") or "").strip()
            ids.append(doc_id)
            docs_tokens.append(tokenize(text, stopwords, min_word_length))

    return ids, docs_tokens, selected_id_column, selected_text_column, delimiter_used


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


def compute_keyword_cooccurrence(docs_tokens: list[list[str]], window: int) -> Counter[tuple[str, str]]:
    cooc_counter: Counter[tuple[str, str]] = Counter()
    if window <= 0:
        return cooc_counter

    for tokens in docs_tokens:
        for i, token_a in enumerate(tokens):
            max_j = min(i + window + 1, len(tokens))
            for j in range(i + 1, max_j):
                token_b = tokens[j]
                if token_a == token_b:
                    continue
                edge = tuple(sorted((token_a, token_b)))
                cooc_counter[edge] += 1

    return cooc_counter


def write_article_keywords(output_file: Path, doc_ids: list[str], keywords: list[list[tuple[str, float]]]) -> None:
    with output_file.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "rank", "keyword", "score"])
        for doc_id, items in zip(doc_ids, keywords):
            for rank, (word, score) in enumerate(items, start=1):
                writer.writerow([doc_id, rank, word, f"{score:.8f}"])


def write_keyword_cooccurrence(output_file: Path, cooc_counter: Counter[tuple[str, str]]) -> None:
    with output_file.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["source", "target", "weight"])
        for (source, target), weight in cooc_counter.most_common():
            writer.writerow([source, target, weight])




def write_wordcloud_png(output_file: Path, corpus_counter: Counter[str], max_words: int = 120) -> None:
    items = corpus_counter.most_common(max_words)

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError:
        # Fallback sem bibliotecas externas: renderização simples em bitmap + PNG via stdlib.
        font = {
            "a": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
            "b": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
            "c": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
            "d": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
            "e": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
            "f": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
            "g": ["01110", "10001", "10000", "10111", "10001", "10001", "01110"],
            "h": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
            "i": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
            "j": ["00111", "00010", "00010", "00010", "10010", "10010", "01100"],
            "k": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
            "l": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
            "m": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
            "n": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
            "o": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
            "p": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
            "q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
            "r": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
            "s": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
            "t": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
            "u": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
            "v": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
            "w": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
            "x": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
            "y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
            "z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
            "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
            "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
            "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
            "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
            "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
            "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
            "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
            "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
            "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
            "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
        }

        def normalize_word(word: str) -> str:
            normalized = unicodedata.normalize("NFKD", word)
            ascii_word = "".join(ch for ch in normalized if not unicodedata.combining(ch))
            return "".join(ch.lower() for ch in ascii_word if ch.isalnum())

        width, height = 1600, 1000
        canvas = bytearray([255] * (width * height * 3))

        def set_px(x: int, y: int, color: tuple[int, int, int]) -> None:
            if 0 <= x < width and 0 <= y < height:
                idx = (y * width + x) * 3
                canvas[idx:idx + 3] = bytes(color)

        def draw_char(x: int, y: int, ch: str, scale: int, color: tuple[int, int, int]) -> int:
            pattern = font.get(ch)
            if not pattern:
                return 4 * scale
            for row, bits in enumerate(pattern):
                for col, bit in enumerate(bits):
                    if bit == "1":
                        for dy in range(scale):
                            for dx in range(scale):
                                set_px(x + col * scale + dx, y + row * scale + dy, color)
            return 6 * scale

        def draw_word(x: int, y: int, word: str, scale: int, color: tuple[int, int, int]) -> None:
            cursor = x
            for ch in word:
                cursor += draw_char(cursor, y, ch, scale, color)

        rng = random.Random(42)
        if not items:
            items = [("sem palavras", 1)]
        max_freq = max(freq for _, freq in items)
        min_freq = min(freq for _, freq in items)

        for word, freq in items[:100]:
            clean = normalize_word(word)
            if not clean:
                continue
            scale = 2 if max_freq == min_freq else int(1 + (freq - min_freq) / (max_freq - min_freq) * 5)
            w_px = len(clean) * 6 * scale
            h_px = 7 * scale
            x = rng.randint(10, max(10, width - w_px - 10))
            y = rng.randint(10, max(10, height - h_px - 10))
            color = (rng.randint(20, 120), rng.randint(40, 140), rng.randint(90, 220))
            draw_word(x, y, clean, scale, color)

        def chunk(chunk_type: bytes, data: bytes) -> bytes:
            crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
            return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)

        signature = b"\x89PNG\r\n\x1a\n"
        ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)

        raw = bytearray()
        row_bytes = width * 3
        for y in range(height):
            raw.append(0)
            start = y * row_bytes
            raw.extend(canvas[start:start + row_bytes])

        idat = zlib.compress(bytes(raw), level=9)
        png_bytes = signature + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
        output_file.write_bytes(png_bytes)
        return

    if not items:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, "Sem palavras para exibir", ha="center", va="center", fontsize=24)
        ax.axis("off")
        fig.savefig(output_file, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    max_freq = items[0][1]
    min_freq = items[-1][1]
    rng = random.Random(42)

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_facecolor("white")
    ax.axis("off")

    for word, freq in items:
        if max_freq == min_freq:
            size = 28
        else:
            size = 12 + (freq - min_freq) / (max_freq - min_freq) * 44

        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.08, 0.92)
        color = (rng.random() * 0.4, rng.random() * 0.5, rng.random() * 0.7)
        ax.text(x, y, word, fontsize=size, color=color, ha="center", va="center", transform=ax.transAxes)

    fig.savefig(output_file, dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    output_file: Path,
    num_docs: int,
    total_tokens: int,
    vocabulary_size: int,
    avg_tokens_per_doc: float,
    top_words: list[tuple[str, int]],
    id_column_used: str,
    text_column_used: str,
    delimiter_used: str,
    cooccurrence_edges: int,
) -> None:
    summary = {
        "documents": num_docs,
        "total_tokens": total_tokens,
        "vocabulary_size": vocabulary_size,
        "avg_tokens_per_document": round(avg_tokens_per_doc, 4),
        "id_column_used": id_column_used or "generated_row_number",
        "text_column_used": text_column_used,
        "delimiter_used": "\\t" if delimiter_used == "\t" else delimiter_used,
        "cooccurrence_edges": cooccurrence_edges,
        "top_20_words": [{"word": word, "frequency": freq} for word, freq in top_words],
    }
    output_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stopwords = load_stopwords(args.language, args.stopwords_file)

    doc_ids, docs_tokens, id_column_used, text_column_used, delimiter_used = read_articles(
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
    cooccurrence = compute_keyword_cooccurrence(docs_tokens, args.cooccurrence_window)

    write_word_frequency(output_dir / "word_frequency.csv", corpus_counter, total_tokens)
    write_document_word_matrix(output_dir / "document_word_matrix.csv", doc_ids, docs_tokens, top_words)
    write_article_keywords(output_dir / "article_keywords.csv", doc_ids, article_keywords)
    if args.cooccurrence_window > 0:
        write_keyword_cooccurrence(output_dir / "keyword_cooccurrence.csv", cooccurrence)

    write_wordcloud_png(output_dir / args.wordcloud_png, corpus_counter)

    write_summary(
        output_dir / "word_map_summary.json",
        num_docs=len(doc_ids),
        total_tokens=total_tokens,
        vocabulary_size=len(corpus_counter),
        avg_tokens_per_doc=(total_tokens / len(doc_ids) if doc_ids else 0),
        top_words=corpus_counter.most_common(20),
        id_column_used=id_column_used,
        text_column_used=text_column_used,
        delimiter_used=delimiter_used,
        cooccurrence_edges=len(cooccurrence),
    )

    print(f"Arquivos gerados em: {output_dir}")


if __name__ == "__main__":
    main()

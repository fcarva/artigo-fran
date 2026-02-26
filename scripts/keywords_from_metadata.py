import argparse
import json
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import pandas as pd


MISSING_VALUE = "nao encontrado"

FIELD_ALIASES = {
    "title": [
        "title",
        "titulo",
        "título",
        "article title",
        "document title",
        "paper title",
    ],
    "authors": [
        "authors",
        "author",
        "autor",
        "autores",
        "author list",
    ],
    "year": [
        "year",
        "ano",
        "publication year",
        "pub year",
        "date",
    ],
    "journal": [
        "journal",
        "source title",
        "revista",
        "periodico",
        "publication",
        "venue",
    ],
    "doi": [
        "doi",
        "digital object identifier",
    ],
    "abstract": [
        "abstract",
        "resumo",
        "summary",
    ],
    "keywords_author": [
        "author keywords",
        "keywords",
        "keyword",
        "palavras-chave",
        "palavras chave",
        "keywords author",
    ],
    "source_url": [
        "source_url",
        "url",
        "link",
        "article url",
        "document url",
        "source",
    ],
}

DOI_REGEX = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE)

STOPWORDS_EN = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "being",
    "between",
    "both",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "during",
    "each",
    "for",
    "from",
    "had",
    "has",
    "have",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "more",
    "most",
    "of",
    "on",
    "or",
    "other",
    "our",
    "out",
    "over",
    "such",
    "than",
    "that",
    "the",
    "their",
    "there",
    "these",
    "this",
    "those",
    "through",
    "to",
    "under",
    "using",
    "use",
    "used",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "within",
}

STOPWORDS_PT = {
    "a",
    "ao",
    "aos",
    "as",
    "com",
    "como",
    "da",
    "das",
    "de",
    "dela",
    "dele",
    "deles",
    "do",
    "dos",
    "e",
    "em",
    "entre",
    "era",
    "essa",
    "esse",
    "esta",
    "este",
    "foi",
    "foram",
    "ha",
    "isso",
    "isto",
    "mais",
    "mas",
    "na",
    "nas",
    "no",
    "nos",
    "o",
    "os",
    "ou",
    "para",
    "pela",
    "pelas",
    "pelo",
    "pelos",
    "por",
    "qual",
    "que",
    "se",
    "sem",
    "ser",
    "seu",
    "seus",
    "sua",
    "suas",
    "tambem",
    "teve",
    "tem",
    "tendo",
    "um",
    "uma",
    "umas",
    "uns",
}

GENERIC_SCI_WORDS = {
    "analysis",
    "analise",
    "article",
    "artigo",
    "background",
    "based",
    "caso",
    "conclusion",
    "conclusions",
    "dados",
    "data",
    "estudo",
    "estudos",
    "evidence",
    "finding",
    "findings",
    "method",
    "methods",
    "objective",
    "objectives",
    "paper",
    "results",
    "resultados",
    "review",
    "sample",
    "study",
}

STOPWORDS_ALL = STOPWORDS_EN | STOPWORDS_PT | GENERIC_SCI_WORDS

MANUAL_KEYWORDS_BY_TITLE = {
    (
        "Nutritional status of children aged 6–59 mo born to mothers treated for severe acute "
        "malnutrition in childhood: an observational Lwiro cohort study in Democratic Republic of Congo"
    ): [
        "childhood malnutrition",
        "intergeneration malnutrition",
        "acute malnutrition",
        "stunting",
        "anemia",
        "maternal and child health",
    ],
    (
        "The effect of childhood stunting and wasting on adolescent cardiovascular diseases risk "
        "and educational achievement in rural Uganda: a retrospective cohort study"
    ): [
        "Child hood undernutrition",
        "blood pressure",
        "schooling",
        "adolescence",
        "Uganda",
    ],
}


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_for_match(text: Any) -> str:
    raw = strip_accents(str(text).strip().lower())
    raw = re.sub(r"[^a-z0-9]+", " ", raw)
    return re.sub(r"\s+", " ", raw).strip()


def normalize_header(name: str) -> str:
    return normalize_for_match(name).replace(" ", "")


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True

    text = str(value).strip()
    if not text:
        return True

    normalized = normalize_for_match(text).replace(" ", "")
    return normalized in {"nan", "none", "null", "naoencontrado", "notfound"}


def clean_text(value: Any) -> str:
    if is_missing(value):
        return ""
    return str(value).strip()


def normalize_doi(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text, flags=re.IGNORECASE)
    match = DOI_REGEX.search(text)
    if match:
        return match.group(0).strip().rstrip(".").lower()
    return text.strip().lower()


def parse_year(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    match = re.search(r"(19|20)\d{2}", text)
    return match.group(0) if match else text


def strip_html(text: Any) -> str:
    value = clean_text(text)
    if not value:
        return ""
    value = re.sub(r"<[^>]+>", " ", value)
    value = unescape(value)
    return re.sub(r"\s+", " ", value).strip()


def first_item(value: Any) -> str:
    if isinstance(value, list) and value:
        return clean_text(value[0])
    return clean_text(value)


def split_keyword_field(value: Any) -> list[str]:
    text = clean_text(value)
    if not text:
        return []

    if ";" in text or "|" in text:
        parts = re.split(r"[;|]", text)
    else:
        parts = text.split(",")
    return [part.strip() for part in parts if part and part.strip()]


def normalize_keyword(value: Any) -> str:
    text = clean_text(value).lower()
    if not text:
        return ""
    text = text.replace("-", " ")
    text = re.sub(r"[^\w\sÀ-ÿ]", " ", text)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 2 or text.isdigit():
        return ""
    return text


def dedupe_keywords(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for raw in values:
        keyword = normalize_keyword(raw)
        if keyword and keyword not in seen:
            seen.add(keyword)
            output.append(keyword)
    return output


def get_manual_keywords_for_title(title: str) -> list[str]:
    target = normalize_for_match(title)
    if not target:
        return []

    for manual_title, keywords in MANUAL_KEYWORDS_BY_TITLE.items():
        if normalize_for_match(manual_title) == target:
            return dedupe_keywords(keywords)
    return []


def extract_keywords_from_abstract(abstract: Any, top_n: int = 10) -> list[str]:
    text = strip_html(abstract).lower()
    if not text:
        return []

    tokens = re.findall(r"[a-zA-ZÀ-ÿ][a-zA-ZÀ-ÿ0-9-]*", text)
    normalized_tokens = [normalize_keyword(token) for token in tokens]
    normalized_tokens = [token for token in normalized_tokens if token]

    if not normalized_tokens:
        return []

    unigram_counter: Counter[str] = Counter()
    phrase_counter: Counter[str] = Counter()

    def is_valid_token(token: str) -> bool:
        if len(token) < 3:
            return False
        if token in STOPWORDS_ALL:
            return False
        if token.isdigit():
            return False
        return True

    for token in normalized_tokens:
        if is_valid_token(token):
            unigram_counter[token] += 1

    for n in (2, 3):
        for start in range(0, len(normalized_tokens) - n + 1):
            chunk = normalized_tokens[start : start + n]
            if all(is_valid_token(token) for token in chunk):
                phrase = " ".join(chunk)
                phrase_counter[phrase] += 1

    scored_terms: list[tuple[str, float, int]] = []
    for term, freq in unigram_counter.items():
        score = float(freq)
        scored_terms.append((term, score, freq))
    for term, freq in phrase_counter.items():
        size = len(term.split())
        score = float(freq) * (1.0 + (size - 1) * 0.35)
        scored_terms.append((term, score, freq))

    scored_terms.sort(key=lambda item: (item[1], item[2], len(item[0])), reverse=True)

    chosen: list[str] = []
    seen: set[str] = set()
    for term, _, _ in scored_terms:
        if term in seen:
            continue
        seen.add(term)
        chosen.append(term)
        if len(chosen) >= top_n:
            break

    return chosen


def similarity_score(left: str, right: str) -> float:
    left_norm = normalize_for_match(left)
    right_norm = normalize_for_match(right)
    if not left_norm or not right_norm:
        return 0.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def choose_best_by_title(
    candidates: list[dict[str, Any]],
    target_title: str,
    target_year: str,
    get_title,
    get_year,
) -> Optional[dict[str, Any]]:
    best_item: Optional[dict[str, Any]] = None
    best_score = 0.0
    target_year_int = int(target_year) if target_year.isdigit() else None

    for candidate in candidates:
        title = clean_text(get_title(candidate))
        if not title:
            continue
        score = similarity_score(target_title, title)

        cand_year = clean_text(get_year(candidate))
        if target_year_int and cand_year.isdigit():
            diff = abs(target_year_int - int(cand_year))
            if diff == 0:
                score += 0.15
            elif diff == 1:
                score += 0.08

        if score > best_score:
            best_item = candidate
            best_score = score

    if best_score < 0.45:
        return None
    return best_item


def find_column(df: pd.DataFrame, aliases: list[str]) -> Optional[str]:
    normalized_columns = {column: normalize_header(column) for column in df.columns}
    alias_norm = [normalize_header(alias) for alias in aliases]

    for alias in alias_norm:
        for column, norm in normalized_columns.items():
            if norm == alias:
                return column

    for alias in alias_norm:
        for column, norm in normalized_columns.items():
            if alias in norm or norm in alias:
                return column

    return None


def map_columns(df: pd.DataFrame) -> dict[str, Optional[str]]:
    mapped: dict[str, Optional[str]] = {}
    for field, aliases in FIELD_ALIASES.items():
        mapped[field] = find_column(df, aliases)
    return mapped


def parse_bib_file(path: Path) -> list[dict[str, str]]:
    content = path.read_text(encoding="utf-8", errors="replace")
    entries = re.split(r"(?=@\w+\s*\{)", content)
    records: list[dict[str, str]] = []

    for entry in entries:
        if "@" not in entry:
            continue
        record = {
            "title": "",
            "authors": "",
            "year": "",
            "journal": "",
            "doi": "",
            "abstract": "",
            "keywords_author": "",
            "source_url": "",
        }
        for field, target in [
            ("title", "title"),
            ("author", "authors"),
            ("year", "year"),
            ("journal", "journal"),
            ("doi", "doi"),
            ("abstract", "abstract"),
            ("keywords", "keywords_author"),
            ("url", "source_url"),
        ]:
            pattern = re.compile(
                rf"{field}\s*=\s*[\{{\"](.+?)[\}}\"],?\s*$",
                flags=re.IGNORECASE | re.MULTILINE,
            )
            match = pattern.search(entry)
            if match:
                value = match.group(1).replace("\n", " ").strip()
                record[target] = value
        if any(clean_text(value) for value in record.values()):
            records.append(record)

    return records


def parse_ris_file(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    current = {
        "title": "",
        "authors": "",
        "year": "",
        "journal": "",
        "doi": "",
        "abstract": "",
        "keywords_author": "",
        "source_url": "",
    }
    authors: list[str] = []
    keywords: list[str] = []

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if " - " not in raw_line:
            continue
        tag = raw_line[:2].strip().upper()
        value = raw_line[6:].strip()
        if tag == "TY":
            current = {key: "" for key in current}
            authors = []
            keywords = []
        elif tag in {"TI", "T1"}:
            current["title"] = value
        elif tag == "AU":
            authors.append(value)
        elif tag in {"PY", "Y1"}:
            current["year"] = value
        elif tag in {"JO", "JF", "T2"}:
            current["journal"] = value
        elif tag == "DO":
            current["doi"] = value
        elif tag == "AB":
            current["abstract"] = value
        elif tag == "KW":
            keywords.append(value)
        elif tag == "UR":
            current["source_url"] = value
        elif tag == "ER":
            current["authors"] = "; ".join(authors)
            current["keywords_author"] = "; ".join(keywords)
            if any(clean_text(value) for value in current.values()):
                records.append(current.copy())

    return records


def parse_txt_file(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = re.sub(r"^\s*\d+[.)-]?\s*", "", line).strip()
        if not stripped:
            continue

        record = {
            "title": "",
            "authors": "",
            "year": "",
            "journal": "",
            "doi": "",
            "abstract": "",
            "keywords_author": "",
            "source_url": "",
        }
        doi_match = DOI_REGEX.search(stripped)
        if doi_match:
            record["doi"] = doi_match.group(0)
        elif stripped.lower().startswith("http"):
            record["source_url"] = stripped
        else:
            record["title"] = stripped
            year_match = re.search(r"(19|20)\d{2}", stripped)
            if year_match:
                record["year"] = year_match.group(0)

        records.append(record)

    return records


def load_extra_records(paths: list[str]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        suffix = path.suffix.lower()
        if suffix == ".bib":
            records.extend(parse_bib_file(path))
        elif suffix == ".ris":
            records.extend(parse_ris_file(path))
        else:
            records.extend(parse_txt_file(path))
    return records


def keyword_list_to_text(values: list[str]) -> str:
    return "; ".join([value for value in values if clean_text(value)])


def parse_crossref_message(message: dict[str, Any]) -> dict[str, str]:
    authors = []
    for author in message.get("author", []):
        given = clean_text(author.get("given"))
        family = clean_text(author.get("family"))
        name = clean_text(author.get("name"))
        full_name = " ".join([part for part in [given, family] if part]).strip()
        if full_name:
            authors.append(full_name)
        elif name:
            authors.append(name)

    year = ""
    for field in ["issued", "published-print", "published-online", "created"]:
        block = message.get(field, {})
        date_parts = block.get("date-parts", [])
        if date_parts and date_parts[0]:
            year = str(date_parts[0][0])
            break

    keywords = [clean_text(item) for item in message.get("subject", []) if clean_text(item)]
    return {
        "title": first_item(message.get("title")),
        "authors": "; ".join(authors),
        "year": year,
        "journal": first_item(message.get("container-title")),
        "doi": normalize_doi(message.get("DOI")),
        "abstract": strip_html(message.get("abstract")),
        "keywords_author": keyword_list_to_text(keywords),
        "source_url": clean_text(message.get("URL")),
    }


def reconstruct_openalex_abstract(index: Any) -> str:
    if not isinstance(index, dict):
        return ""
    ordered: list[tuple[int, str]] = []
    for word, positions in index.items():
        if not isinstance(positions, list):
            continue
        for pos in positions:
            if isinstance(pos, int):
                ordered.append((pos, word))
    if not ordered:
        return ""
    ordered.sort(key=lambda item: item[0])
    return " ".join([word for _, word in ordered])


def parse_openalex_work(work: dict[str, Any]) -> dict[str, str]:
    authors = []
    for authorship in work.get("authorships", []):
        author = authorship.get("author", {})
        name = clean_text(author.get("display_name"))
        if name:
            authors.append(name)

    keywords = []
    for keyword in work.get("keywords", []):
        name = clean_text(keyword.get("display_name"))
        if name:
            keywords.append(name)
    if not keywords:
        for concept in work.get("concepts", []):
            name = clean_text(concept.get("display_name"))
            score = concept.get("score", 0)
            if name and isinstance(score, (int, float)) and score >= 0.35:
                keywords.append(name)
        keywords = keywords[:10]

    doi = clean_text(work.get("doi"))
    doi = normalize_doi(doi)
    primary_location = work.get("primary_location", {}) or {}
    source = primary_location.get("source", {}) or {}
    return {
        "title": clean_text(work.get("display_name")),
        "authors": "; ".join(authors),
        "year": parse_year(work.get("publication_year")),
        "journal": clean_text(source.get("display_name")),
        "doi": doi,
        "abstract": reconstruct_openalex_abstract(work.get("abstract_inverted_index")),
        "keywords_author": keyword_list_to_text(keywords),
        "source_url": clean_text(primary_location.get("landing_page_url") or work.get("id")),
    }


def parse_semantic_scholar_item(item: dict[str, Any]) -> dict[str, str]:
    authors = []
    for author in item.get("authors", []):
        name = clean_text(author.get("name"))
        if name:
            authors.append(name)

    external_ids = item.get("externalIds", {}) or {}
    doi = normalize_doi(external_ids.get("DOI"))

    keywords = []
    for field in item.get("fieldsOfStudy", []) or []:
        value = clean_text(field)
        if value:
            keywords.append(value)

    return {
        "title": clean_text(item.get("title")),
        "authors": "; ".join(authors),
        "year": parse_year(item.get("year")),
        "journal": clean_text(item.get("venue")),
        "doi": doi,
        "abstract": strip_html(item.get("abstract")),
        "keywords_author": keyword_list_to_text(keywords),
        "source_url": clean_text(item.get("url")),
    }


class MetaParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.meta: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag.lower() != "meta":
            return
        attr_dict = {
            key.lower(): clean_text(value)
            for key, value in attrs
            if key is not None and value is not None
        }
        key = attr_dict.get("name") or attr_dict.get("property")
        content = attr_dict.get("content")
        if key and content and key.lower() not in self.meta:
            self.meta[key.lower()] = content


@dataclass
class FetchResult:
    data: dict[str, str]
    error: str = ""


class MetadataFetcher:
    def __init__(self, sleep_seconds: float = 1.0, timeout: int = 25):
        self.sleep_seconds = sleep_seconds
        self.timeout = timeout
        self.last_call = 0.0
        self.cache: dict[str, Any] = {}

    def _respect_rate_limit(self) -> None:
        elapsed = time.time() - self.last_call
        if elapsed < self.sleep_seconds:
            time.sleep(self.sleep_seconds - elapsed)
        self.last_call = time.time()

    def _request_json(self, url: str, headers: Optional[dict[str, str]] = None) -> FetchResult:
        if url in self.cache:
            return FetchResult(data=self.cache[url])
        request_headers = {
            "User-Agent": "keyword-extractor/1.0 (mailto:local@localhost)",
            "Accept": "application/json",
        }
        if headers:
            request_headers.update(headers)

        try:
            self._respect_rate_limit()
            request = Request(url, headers=request_headers)
            with urlopen(request, timeout=self.timeout) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                raw = response.read().decode(charset, errors="replace")
            data = json.loads(raw)
            self.cache[url] = data
            return FetchResult(data=data)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            return FetchResult(data={}, error=str(exc))

    def _request_text(self, url: str) -> FetchResult:
        cache_key = f"text::{url}"
        if cache_key in self.cache:
            return FetchResult(data={"text": self.cache[cache_key]})
        headers = {
            "User-Agent": "keyword-extractor/1.0 (mailto:local@localhost)",
        }
        try:
            self._respect_rate_limit()
            request = Request(url, headers=headers)
            with urlopen(request, timeout=self.timeout) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                raw = response.read().decode(charset, errors="replace")
            self.cache[cache_key] = raw
            return FetchResult(data={"text": raw})
        except (HTTPError, URLError, TimeoutError) as exc:
            return FetchResult(data={}, error=str(exc))

    def crossref_by_doi(self, doi: str) -> FetchResult:
        normalized_doi = normalize_doi(doi)
        if not normalized_doi:
            return FetchResult(data={})
        url = f"https://api.crossref.org/works/{quote(normalized_doi, safe='')}"
        response = self._request_json(url)
        if response.error:
            return response
        message = response.data.get("message", {})
        return FetchResult(data=parse_crossref_message(message))

    def crossref_by_title(self, title: str, year: str) -> FetchResult:
        if is_missing(title):
            return FetchResult(data={})
        params = {
            "query.title": title,
            "rows": 5,
        }
        url = "https://api.crossref.org/works?" + urlencode(params)
        response = self._request_json(url)
        if response.error:
            return response
        items = response.data.get("message", {}).get("items", [])
        if not isinstance(items, list):
            return FetchResult(data={})
        best = choose_best_by_title(
            items,
            target_title=title,
            target_year=year,
            get_title=lambda item: first_item(item.get("title")),
            get_year=lambda item: parse_crossref_message(item).get("year", ""),
        )
        if not best:
            return FetchResult(data={})
        return FetchResult(data=parse_crossref_message(best))

    def openalex_by_doi(self, doi: str) -> FetchResult:
        normalized_doi = normalize_doi(doi)
        if not normalized_doi:
            return FetchResult(data={})
        url = (
            "https://api.openalex.org/works?"
            + urlencode({"filter": f"doi:{normalized_doi}", "per-page": 1})
        )
        response = self._request_json(url)
        if response.error:
            return response
        results = response.data.get("results", [])
        if not results:
            return FetchResult(data={})
        return FetchResult(data=parse_openalex_work(results[0]))

    def semantic_scholar_by_title(self, title: str, year: str) -> FetchResult:
        if is_missing(title):
            return FetchResult(data={})
        params = {
            "query": title,
            "limit": 5,
            "fields": "title,year,authors,venue,abstract,externalIds,url,fieldsOfStudy",
        }
        url = "https://api.semanticscholar.org/graph/v1/paper/search?" + urlencode(params)
        response = self._request_json(url)
        if response.error:
            return response
        data = response.data.get("data", [])
        if not isinstance(data, list):
            return FetchResult(data={})
        best = choose_best_by_title(
            data,
            target_title=title,
            target_year=year,
            get_title=lambda item: clean_text(item.get("title")),
            get_year=lambda item: parse_year(item.get("year")),
        )
        if not best:
            return FetchResult(data={})
        return FetchResult(data=parse_semantic_scholar_item(best))

    def pubmed_by_title(self, title: str, year: str) -> FetchResult:
        if is_missing(title):
            return FetchResult(data={})

        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + urlencode(
            {
                "db": "pubmed",
                "retmode": "json",
                "retmax": 5,
                "term": f"{title}[Title]",
            }
        )
        search_result = self._request_json(search_url)
        if search_result.error:
            return search_result
        id_list = search_result.data.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return FetchResult(data={})

        summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?" + urlencode(
            {
                "db": "pubmed",
                "retmode": "json",
                "id": ",".join(id_list),
            }
        )
        summary_result = self._request_json(summary_url)
        if summary_result.error:
            return summary_result
        summary_data = summary_result.data.get("result", {})

        candidates = []
        for pmid in id_list:
            summary_item = summary_data.get(pmid, {})
            if summary_item:
                candidates.append(
                    {
                        "pmid": pmid,
                        "title": summary_item.get("title"),
                        "pubdate": summary_item.get("pubdate", ""),
                    }
                )

        best = choose_best_by_title(
            candidates,
            target_title=title,
            target_year=year,
            get_title=lambda item: clean_text(item.get("title")),
            get_year=lambda item: parse_year(item.get("pubdate")),
        )
        if not best:
            return FetchResult(data={})

        pmid = best["pmid"]
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?" + urlencode(
            {
                "db": "pubmed",
                "id": pmid,
                "retmode": "xml",
            }
        )
        fetched = self._request_text(fetch_url)
        if fetched.error:
            return FetchResult(data={}, error=fetched.error)

        xml_text = fetched.data.get("text", "")
        if not xml_text:
            return FetchResult(data={})

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            return FetchResult(data={}, error=str(exc))

        article = root.find(".//PubmedArticle")
        if article is None:
            return FetchResult(data={})

        def xml_node_text(node: Optional[ET.Element]) -> str:
            if node is None:
                return ""
            return " ".join([piece.strip() for piece in node.itertext() if piece.strip()])

        title_value = xml_node_text(article.find(".//ArticleTitle"))
        abstract_parts = [xml_node_text(node) for node in article.findall(".//Abstract/AbstractText")]
        abstract_value = " ".join([part for part in abstract_parts if part])
        journal_value = xml_node_text(article.find(".//Journal/Title"))
        year_value = xml_node_text(article.find(".//PubDate/Year"))
        if not year_value:
            medline_date = xml_node_text(article.find(".//PubDate/MedlineDate"))
            year_value = parse_year(medline_date)

        authors = []
        for author in article.findall(".//AuthorList/Author"):
            last_name = xml_node_text(author.find("./LastName"))
            fore_name = xml_node_text(author.find("./ForeName"))
            collective_name = xml_node_text(author.find("./CollectiveName"))
            full_name = " ".join([piece for piece in [fore_name, last_name] if piece]).strip()
            if full_name:
                authors.append(full_name)
            elif collective_name:
                authors.append(collective_name)

        doi = ""
        for article_id in article.findall(".//ArticleIdList/ArticleId"):
            if article_id.attrib.get("IdType", "").lower() == "doi":
                doi = xml_node_text(article_id)
                break

        keywords = []
        for keyword in article.findall(".//KeywordList/Keyword"):
            value = xml_node_text(keyword)
            if value:
                keywords.append(value)

        return FetchResult(
            data={
                "title": title_value,
                "authors": "; ".join(authors),
                "year": parse_year(year_value),
                "journal": journal_value,
                "doi": normalize_doi(doi),
                "abstract": abstract_value,
                "keywords_author": keyword_list_to_text(keywords),
                "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            }
        )

    def scrape_by_url(self, url: str) -> FetchResult:
        if is_missing(url):
            return FetchResult(data={})
        fetched = self._request_text(url)
        if fetched.error:
            return fetched
        html_text = fetched.data.get("text", "")
        if not html_text:
            return FetchResult(data={})

        parser = MetaParser()
        parser.feed(html_text)
        meta = parser.meta
        candidates = {
            "title": [
                "citation_title",
                "dc.title",
                "og:title",
                "twitter:title",
            ],
            "doi": [
                "citation_doi",
                "dc.identifier",
                "prism.doi",
            ],
            "journal": [
                "citation_journal_title",
                "prism.publicationname",
                "dc.source",
            ],
            "abstract": [
                "description",
                "dc.description",
                "og:description",
                "citation_abstract",
            ],
            "keywords_author": [
                "citation_keywords",
                "keywords",
                "dc.subject",
            ],
        }
        parsed = {
            "title": "",
            "authors": "",
            "year": "",
            "journal": "",
            "doi": "",
            "abstract": "",
            "keywords_author": "",
            "source_url": clean_text(url),
        }

        for field, keys in candidates.items():
            for key in keys:
                if key in meta and clean_text(meta[key]):
                    parsed[field] = clean_text(meta[key])
                    break

        parsed["doi"] = normalize_doi(parsed["doi"])
        return FetchResult(data=parsed)


def merge_missing_fields(base: dict[str, str], candidate: dict[str, str]) -> None:
    for field in ["title", "authors", "year", "journal", "doi", "abstract", "keywords_author", "source_url"]:
        if is_missing(base.get(field, "")) and not is_missing(candidate.get(field, "")):
            value = clean_text(candidate.get(field))
            if field == "doi":
                value = normalize_doi(value)
            if field == "year":
                value = parse_year(value)
            base[field] = value


def get_missing_fields(record: dict[str, str]) -> list[str]:
    fields = ["title", "authors", "year", "journal", "doi", "abstract", "keywords_author"]
    return [field for field in fields if is_missing(record.get(field))]


def should_fetch_more(record: dict[str, str]) -> bool:
    needed = ["doi", "journal", "abstract", "keywords_author"]
    return any(is_missing(record.get(field)) for field in needed)


def enrich_record(
    record: dict[str, str],
    fetcher: MetadataFetcher,
    skip_api: bool,
    warning_set: set[str],
) -> tuple[dict[str, str], list[str]]:
    used_sources: list[str] = []
    if skip_api:
        return record, used_sources

    def apply_result(source_name: str, result: FetchResult) -> None:
        if result.error:
            warning_set.add(f"{source_name}: {result.error}")
            return
        if result.data:
            before = record.copy()
            merge_missing_fields(record, result.data)
            if record != before:
                used_sources.append(source_name)

    if not is_missing(record.get("doi")):
        apply_result("crossref_doi", fetcher.crossref_by_doi(record["doi"]))
        if not is_missing(record.get("doi")) and should_fetch_more(record):
            apply_result("openalex_doi", fetcher.openalex_by_doi(record["doi"]))
    else:
        apply_result("crossref_title", fetcher.crossref_by_title(record.get("title", ""), record.get("year", "")))

    if should_fetch_more(record):
        apply_result("pubmed_title", fetcher.pubmed_by_title(record.get("title", ""), record.get("year", "")))

    if should_fetch_more(record):
        apply_result(
            "semantic_scholar_title",
            fetcher.semantic_scholar_by_title(record.get("title", ""), record.get("year", "")),
        )

    if should_fetch_more(record) and not is_missing(record.get("doi")):
        apply_result("openalex_doi", fetcher.openalex_by_doi(record["doi"]))

    if should_fetch_more(record) and not is_missing(record.get("source_url")):
        apply_result("scrape_url", fetcher.scrape_by_url(record["source_url"]))

    if is_missing(record.get("keywords_author")):
        manual_keywords = get_manual_keywords_for_title(record.get("title", ""))
        if manual_keywords:
            record["keywords_author"] = keyword_list_to_text(manual_keywords)
            used_sources.append("manual_keywords")

    if is_missing(record.get("doi")) and not is_missing(record.get("title")):
        warning_set.add(f'Artigo "{record.get("title", "")}" - DOI nao encontrado; titulo usado como fallback')
    if is_missing(record.get("abstract")) and is_missing(record.get("keywords_author")):
        warning_set.add(f'Artigo "{record.get("title", "")}" - abstract indisponivel e sem keywords de autor')

    return record, used_sources


def collect_article_keywords(record: dict[str, str]) -> tuple[list[str], str]:
    author_keywords = dedupe_keywords(split_keyword_field(record.get("keywords_author", "")))
    if author_keywords:
        return author_keywords, "author_keywords"

    if not is_missing(record.get("abstract")):
        extracted = dedupe_keywords(extract_keywords_from_abstract(record.get("abstract"), top_n=10))
        if extracted:
            return extracted, "abstract_extraction"

    return [], ""


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Extrai keywords de artigos com base em dados_final.csv e enriquecimento externo.",
    )
    parser.add_argument(
        "--input",
        default=str(project_dir / "data" / "raw" / "dados_final.csv"),
        help="CSV principal com artigos.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_dir / "data" / "processed"),
        help="Diretorio de saida.",
    )
    parser.add_argument(
        "--extra-input",
        nargs="*",
        default=[],
        help="Arquivos extras (.bib/.ris/.txt) com referencias adicionais.",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=0,
        help="Limita o numero de artigos processados (0 = todos).",
    )
    parser.add_argument(
        "--skip-api",
        action="store_true",
        help="Nao chama APIs externas; usa apenas dados locais.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.0,
        help="Intervalo entre requisicoes externas.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=25,
        help="Timeout de requisicao HTTP em segundos.",
    )
    return parser.parse_args()


def build_base_records(df: pd.DataFrame, mapped_columns: dict[str, Optional[str]]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for _, row in df.iterrows():
        record = {
            "title": clean_text(row[mapped_columns["title"]]) if mapped_columns["title"] else "",
            "authors": clean_text(row[mapped_columns["authors"]]) if mapped_columns["authors"] else "",
            "year": parse_year(row[mapped_columns["year"]]) if mapped_columns["year"] else "",
            "journal": clean_text(row[mapped_columns["journal"]]) if mapped_columns["journal"] else "",
            "doi": normalize_doi(row[mapped_columns["doi"]]) if mapped_columns["doi"] else "",
            "abstract": strip_html(row[mapped_columns["abstract"]]) if mapped_columns["abstract"] else "",
            "keywords_author": clean_text(row[mapped_columns["keywords_author"]]) if mapped_columns["keywords_author"] else "",
            "source_url": clean_text(row[mapped_columns["source_url"]]) if mapped_columns["source_url"] else "",
        }
        records.append(record)
    return records


def ensure_missing_marker(value: str) -> str:
    return value if not is_missing(value) else MISSING_VALUE


def print_input_summary(df: pd.DataFrame, mapped_columns: dict[str, Optional[str]]) -> None:
    print("=== RESUMO DO INPUT ===")
    print(f"Total de linhas: {len(df)}")
    print(f"Colunas disponiveis: {', '.join([str(column) for column in df.columns])}")
    print("\nAmostra inicial (3 primeiras linhas):")
    print(df.head(3).to_string(index=False))
    print("\nMapeamento semantico de colunas:")
    for field in ["title", "authors", "year", "journal", "doi", "abstract", "keywords_author", "source_url"]:
        print(f"  - {field}: {mapped_columns.get(field) or '(nao mapeada)'}")
    print()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {input_path}")

    output_dir = Path(args.output_dir) if args.output_dir else input_path.resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, encoding="utf-8")
    mapped_columns = map_columns(df)
    print_input_summary(df, mapped_columns)

    records = build_base_records(df, mapped_columns)
    extra_records = load_extra_records(args.extra_input)
    records.extend(extra_records)

    if args.max_articles and args.max_articles > 0:
        records = records[: args.max_articles]

    fetcher = MetadataFetcher(sleep_seconds=args.sleep_seconds, timeout=args.timeout)

    warning_set: set[str] = set()
    status_rows: list[dict[str, Any]] = []
    keyword_rows: list[dict[str, Any]] = []

    author_keyword_articles = 0
    abstract_keyword_articles = 0
    no_data_articles = 0

    for index, record in enumerate(records, start=1):
        missing_before = get_missing_fields(record)
        enriched_record, used_sources = enrich_record(
            record=record,
            fetcher=fetcher,
            skip_api=args.skip_api,
            warning_set=warning_set,
        )
        missing_after = get_missing_fields(enriched_record)

        keywords, keyword_source = collect_article_keywords(enriched_record)
        if keyword_source == "author_keywords":
            author_keyword_articles += 1
        elif keyword_source == "abstract_extraction":
            abstract_keyword_articles += 1
        else:
            no_data_articles += 1

        for keyword in keywords:
            keyword_rows.append(
                {
                    "keyword": keyword,
                    "source": keyword_source,
                    "article_doi": ensure_missing_marker(enriched_record.get("doi", "")),
                    "article_title": ensure_missing_marker(enriched_record.get("title", "")),
                    "year": ensure_missing_marker(parse_year(enriched_record.get("year", ""))),
                }
            )

        status_rows.append(
            {
                "article_index": index,
                "article_title": ensure_missing_marker(enriched_record.get("title", "")),
                "missing_before": "; ".join(missing_before) if missing_before else "none",
                "missing_after": "; ".join(missing_after) if missing_after else "none",
                "external_sources_used": "; ".join(used_sources) if used_sources else "none",
            }
        )

    keyword_counter = Counter(row["keyword"] for row in keyword_rows)
    for row in keyword_rows:
        row["frequency"] = keyword_counter[row["keyword"]]

    columns_long = ["keyword", "frequency", "source", "article_doi", "article_title", "year"]
    if keyword_rows:
        keywords_articles_df = pd.DataFrame(keyword_rows)[columns_long]
        keywords_articles_df = keywords_articles_df.sort_values(
            by=["frequency", "keyword", "article_title"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
    else:
        keywords_articles_df = pd.DataFrame(columns=columns_long)

    frequency_rows = [
        {"keyword": keyword, "frequency": frequency}
        for keyword, frequency in keyword_counter.items()
        if frequency >= 2
    ]
    keywords_frequency_df = pd.DataFrame(frequency_rows, columns=["keyword", "frequency"])
    if not keywords_frequency_df.empty:
        keywords_frequency_df = keywords_frequency_df.sort_values(
            by=["frequency", "keyword"],
            ascending=[False, True],
        ).reset_index(drop=True)

    status_df = pd.DataFrame(status_rows)

    keywords_articles_path = output_dir / "keywords_articles.csv"
    keywords_frequency_path = output_dir / "keywords_frequency.csv"
    status_path = output_dir / "article_status.csv"

    keywords_articles_df.to_csv(keywords_articles_path, index=False, encoding="utf-8")
    keywords_frequency_df.to_csv(keywords_frequency_path, index=False, encoding="utf-8")
    status_df.to_csv(status_path, index=False, encoding="utf-8")

    print("=== RELATORIO DE EXTRACAO ===")
    print(f"Total de artigos processados: {len(records)}")
    print(f"Artigos com keywords dos autores: {author_keyword_articles}")
    print(f"Artigos com keywords extraidas do abstract: {abstract_keyword_articles}")
    print(f"Artigos sem dados encontrados: {no_data_articles}")
    print()
    print(f"Total de keywords unicas extraidas: {len(keyword_counter)}")
    print("Top 10 keywords mais frequentes:")
    for index, (keyword, frequency) in enumerate(keyword_counter.most_common(10), start=1):
        print(f"  {index}. {keyword} - {frequency} ocorrencias")

    print()
    print("Arquivos gerados:")
    print(f"  OK {keywords_articles_path}")
    print(f"  OK {keywords_frequency_path}")
    print(f"  OK {status_path}")

    if warning_set:
        print()
        print("Avisos:")
        sorted_warnings = sorted(warning_set)
        preview_limit = 30
        for warning in sorted_warnings[:preview_limit]:
            print(f"  WARN {warning}")
        remaining = len(sorted_warnings) - preview_limit
        if remaining > 0:
            print(f"  ... e mais {remaining} avisos")


if __name__ == "__main__":
    main()

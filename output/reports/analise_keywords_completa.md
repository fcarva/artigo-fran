# Keyword Analysis Report

## 1. Scope and Data Provenance
- Input file: `artigo-fran\data\processed\keywords_articles.csv`
- Generation timestamp: **2026-02-26 19:57 UTC**
- Total keyword occurrences: **2070**
- Unique keywords: **361**

## 2. Methods (Academic Reproducibility)
- Keyword normalization: lowercase, punctuation stripped, hyphen harmonized.
- Minimum frequency threshold: **1**.
- Minimum keyword length: **3** characters.
- Wordcloud: frequency-based (`max_words=500`, random_state=42).
- Top-keyword chart: **top 50** terms + Pareto concentration curve.

## 3. Core Findings
- Top 1 share: **8.79%**
- Top 5 cumulative share: **28.16%**
- Top 10 cumulative share: **40.48%**
- Top 20 cumulative share: **56.72%**
- Rank needed to reach 80% cumulative share: **64**
- Singleton keywords (frequency=1): **221** (61.22% of vocabulary)

## 4. Concentration Diagnostics
- Herfindahl-Hirschman Index (HHI): **0.025210**
- Normalized Shannon entropy: **0.770764**
- Gini coefficient (keyword frequency distribution): **0.735228**

Interpretation guide:
- Higher HHI and Gini indicate stronger thematic concentration.
- Higher normalized entropy indicates broader thematic dispersion.

## 5. Source Composition
- `author_keywords`: **361** keywords

## 6. Yearly Highlights
| Year | Top Keyword | Frequency |
|---:|---|---:|
| 1999 | geography | 1 |
| 2002 | medicine | 1 |
| 2005 | medicine | 3 |
| 2006 | medicine | 1 |
| 2008 | malnutrition | 1 |
| 2009 | medicine | 2 |
| 2010 | medicine | 2 |
| 2011 | medicine | 1 |
| 2012 | medicine | 3 |
| 2013 | medicine | 2 |
| 2014 | medicine | 5 |
| 2015 | medicine | 8 |
| 2016 | medicine | 5 |
| 2017 | medicine | 11 |
| 2018 | medicine | 9 |
| 2019 | medicine | 13 |
| 2020 | medicine | 16 |
| 2021 | medicine | 18 |
| 2022 | medicine | 16 |
| 2023 | medicine | 30 |
| 2024 | medicine | 17 |
| 2025 | medicine | 19 |

## 7. Top Keywords Table

| Rank | Keyword | Frequency | Share (%) | Cum. Share (%) | Dominant Source |
|---:|---|---:|---:|---:|---|
| 1 | medicine | 182 | 8.79 | 8.79 | author_keywords |
| 2 | environmental health | 126 | 6.09 | 14.88 | author_keywords |
| 3 | malnutrition | 123 | 5.94 | 20.82 | author_keywords |
| 4 | geography | 78 | 3.77 | 24.59 | author_keywords |
| 5 | demography | 74 | 3.57 | 28.16 | author_keywords |
| 6 | pediatrics | 56 | 2.71 | 30.87 | author_keywords |
| 7 | population | 54 | 2.61 | 33.48 | author_keywords |
| 8 | economic growth | 53 | 2.56 | 36.04 | author_keywords |
| 9 | economics | 52 | 2.51 | 38.55 | author_keywords |
| 10 | overweight | 40 | 1.93 | 40.48 | author_keywords |
| 11 | underweight | 39 | 1.88 | 42.37 | author_keywords |
| 12 | socioeconomics | 38 | 1.84 | 44.20 | author_keywords |
| 13 | food security | 35 | 1.69 | 45.89 | author_keywords |
| 14 | wasting | 35 | 1.69 | 47.58 | author_keywords |
| 15 | body mass index | 34 | 1.64 | 49.23 | author_keywords |
| 16 | public health | 34 | 1.64 | 50.87 | author_keywords |
| 17 | agriculture | 33 | 1.59 | 52.46 | author_keywords |
| 18 | logistic regression | 31 | 1.50 | 53.96 | author_keywords |
| 19 | anthropometry | 29 | 1.40 | 55.36 | author_keywords |
| 20 | developing country | 28 | 1.35 | 56.72 | author_keywords |
| 21 | internal medicine | 28 | 1.35 | 58.07 | author_keywords |
| 22 | political science | 26 | 1.26 | 59.32 | author_keywords |
| 23 | cross sectional study | 24 | 1.16 | 60.48 | author_keywords |
| 24 | severe acute malnutrition | 21 | 1.01 | 61.50 | author_keywords |
| 25 | psychological intervention | 19 | 0.92 | 62.42 | author_keywords |
| 26 | nursing | 18 | 0.87 | 63.28 | author_keywords |
| 27 | socioeconomic status | 18 | 0.87 | 64.15 | author_keywords |
| 28 | development economics | 17 | 0.82 | 64.98 | author_keywords |
| 29 | gerontology | 17 | 0.82 | 65.80 | author_keywords |
| 30 | sociology | 16 | 0.77 | 66.57 | author_keywords |

## 8. Generated Files
- Wordcloud (PNG): `nuvem_palavras_academica.png`
- Top keywords chart (PNG): `grafico_top_keywords_academico.png`
- Full frequency table: `tabela_keywords_completa.csv`
- Top frequency table: `tabela_keywords_top50.csv`

## 9. Limitations
- Keyword quality depends on metadata completeness and source indexing practices.
- API-enriched keywords may over-represent broad subject labels in some databases.
- For inferential analysis, complement this report with full-text topic modeling.

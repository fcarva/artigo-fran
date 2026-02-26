# Keyword Analysis Pipeline (Academic Style)

This project processes article metadata, builds keyword datasets, and generates publication-ready outputs:

- enriched keyword tables
- academic wordcloud
- top-keyword + Pareto concentration chart
- extended Markdown report with concentration diagnostics (HHI, Gini, entropy)

## Folder Organization

```text
artigo-fran/
  data/
    raw/
      dados_final.csv
      dados_exemplo_100_artigos.csv
    processed/
      keywords_articles.csv
      keywords_frequency.csv
      article_status.csv
  output/
    figures/
      nuvem_palavras_academica.png
      nuvem_palavras_academica.svg
      grafico_top_keywords_academico.png
      grafico_top_keywords_academico.svg
    tables/
      tabela_keywords_completa.csv
      tabela_keywords_top40.csv
    reports/
      analise_keywords_completa.md
      resumo_metricas_keywords.json
  scripts/
    keywords_from_metadata.py
  gerar_nuvem_palavras.py
```

## 1) Build Keyword Tables (metadata + enrichment)

```bash
python artigo-fran/scripts/keywords_from_metadata.py \
  --input artigo-fran/data/raw/dados_final.csv \
  --output-dir artigo-fran/data/processed
```

## 2) Generate Academic Visuals + Complete Analysis

```bash
python artigo-fran/gerar_nuvem_palavras.py \
  --input artigo-fran/data/processed/keywords_articles.csv \
  --output-root artigo-fran/output \
  --title "Keyword Cloud - dados_final.csv" \
  --top-n 40 \
  --max-words 500 \
  --min-frequency 1
```

## Optional Arguments

- `--exclude-terms "term1,term2,..."`: remove generic terms from analysis
- `--legacy-output-dir artigo-fran`: also write legacy file names in root folder

## Academic Good Practices Included

- explicit normalization and filtering rules
- reproducible wordcloud (`random_state=42`)
- top-keyword chart plus Pareto concentration curve
- concentration metrics in report:
  - Top-k cumulative shares
  - HHI
  - Gini coefficient
  - normalized Shannon entropy
- structured outputs in figures/tables/reports folders

# Análise de Mapa de Palavras para Artigos

Este repositório contém um script para transformar textos de artigos em saídas úteis para análise estatística.

## O que o script gera

A partir de um arquivo com os artigos, o script produz:

1. `word_frequency.csv`: frequência absoluta e relativa de cada palavra.
2. `document_word_matrix.csv`: matriz documento x palavra (contagem), limitada às palavras mais frequentes.
3. `word_map_summary.json`: resumo estatístico geral (nº de documentos, tokens, vocabulário, etc.).

## Formato de entrada

Arquivo CSV com pelo menos estas colunas:

- `id`: identificador do artigo
- `text`: texto do artigo

Exemplo:

```csv
id,text
1,"Este é um artigo de exemplo"
2,"Outro artigo para análise de palavras"
```

## Como executar

```bash
python3 scripts/word_map_analysis.py \
  --input artigos.csv \
  --output-dir output \
  --language pt \
  --top-words 300
```

### Parâmetros opcionais

- `--delimiter` (padrão: `,`): delimitador do CSV.
- `--min-word-length` (padrão: `3`): tamanho mínimo das palavras.
- `--top-words` (padrão: `300`): número de palavras para matriz documento x palavra.
- `--stopwords-file`: caminho para arquivo de stopwords customizadas (uma por linha).
- `--id-column` (padrão: `id`): nome da coluna de ID.
- `--text-column` (padrão: `text`): nome da coluna de texto.

## Exemplo para 8 mil artigos

Se você possui 8.000 artigos no arquivo `artigos_8000.csv`:

```bash
python3 scripts/word_map_analysis.py --input artigos_8000.csv --output-dir output_8000 --language pt --top-words 500
```

Depois, você pode abrir os CSVs em R, Python, Power BI, Excel ou outro software estatístico.

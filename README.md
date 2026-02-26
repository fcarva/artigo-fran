# Keyword map para artigos (CSV)

Este projeto gera uma base pronta para análise estatística de palavras-chave e para criação de **keyword map**.

## Entradas esperadas

Arquivo CSV com uma coluna de texto (ex.: `text`, `texto`, `conteudo`, `content`) e opcionalmente uma coluna de ID (`id`, `article_id`, etc.).

Se não houver ID, o script cria um ID sequencial automaticamente.

## Saídas geradas

Ao executar o script, ele cria os arquivos abaixo:

1. `word_frequency.csv`  
   Frequência absoluta e relativa de cada palavra no corpus.
2. `document_word_matrix.csv`  
   Matriz documento x palavra (contagem), usando as palavras mais frequentes.
3. `article_keywords.csv`  
   Lista de palavras-chave por artigo (`id`, `rank`, `keyword`, `score`) via TF-IDF.
4. `word_map_summary.json`  
   Resumo estatístico do processamento (nº de documentos, tokens, vocabulário e top palavras).

## Como executar

```bash
python3 scripts/word_map_analysis.py \
  --input dados_exemplo_100_artigos.csv \
  --output-dir output_keywords \
  --language pt \
  --top-words 300 \
  --keywords-per-article 15
```

## Parâmetros úteis

- `--delimiter` (padrão `,`)
- `--text-column` (opcional; se não passar, tenta detectar automaticamente)
- `--id-column` (opcional; se não passar, tenta detectar automaticamente)
- `--top-words` (padrão `300`): tamanho do vocabulário da matriz documento x palavra
- `--keywords-per-article` (padrão `15`): quantas palavras-chave por artigo
- `--min-word-length` (padrão `3`)
- `--stopwords-file`: arquivo de stopwords customizadas (uma por linha)

## Exemplo para seus 83 artigos

```bash
python3 scripts/word_map_analysis.py --input seus_83_artigos.csv --output-dir output_83 --language pt --keywords-per-article 20
```

Depois, use `article_keywords.csv` para construir visualmente seu mapa de palavras-chave (Power BI, Python, R, Gephi etc.).

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter
from wordcloud import WordCloud


def split_keywords(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    text = text.replace(";", ",")
    parts = [item.strip() for item in text.split(",")]
    return [item for item in parts if item]


def normalize_header(name: str) -> str:
    return (
        str(name)
        .strip()
        .casefold()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )


def detect_column(df: pd.DataFrame, aliases: list[str]) -> str:
    normalized = {col: normalize_header(col) for col in df.columns}
    alias_norm = [normalize_header(alias) for alias in aliases]

    for alias in alias_norm:
        for col, norm in normalized.items():
            if norm == alias:
                return col

    for alias in alias_norm:
        for col, norm in normalized.items():
            if alias in norm or norm in alias:
                return col

    raise KeyError(f"Column not found for aliases: {aliases}")


def detect_optional_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    try:
        return detect_column(df, aliases)
    except KeyError:
        return None


def normalize_keyword(value: Any) -> str:
    text = str(value).strip().lower()
    text = text.replace("-", " ")
    text = re.sub(r"[^\w\sA-Za-z0-9]", " ", text)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_year(value: Any) -> str:
    text = str(value).strip()
    match = re.search(r"(19|20)\d{2}", text)
    return match.group(0) if match else ""


def shorten_label(text: str, max_chars: int = 52) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def resolve_default_input(base_dir: Path) -> Path:
    candidates = [
        base_dir / "data" / "processed" / "keywords_articles.csv",
        base_dir / "keywords_articles.csv",
        base_dir / "dados_exemplo_100_artigos.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Generate academic-quality keyword visuals, tables, and report.",
    )
    parser.add_argument(
        "--input",
        default=str(resolve_default_input(base_dir)),
        help="Input CSV (keywords_articles.csv, keywords_frequency.csv, or keyword list CSV).",
    )
    parser.add_argument(
        "--output-root",
        default=str(base_dir / "output"),
        help="Root output directory (creates figures/tables/reports).",
    )
    parser.add_argument(
        "--title",
        default="Keyword Cloud - Academic Summary",
        help="Wordcloud title.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=40,
        help="Number of top keywords in bar chart.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=500,
        help="Maximum words rendered in wordcloud.",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=1,
        help="Minimum global frequency for a keyword to be kept.",
    )
    parser.add_argument(
        "--min-keyword-length",
        type=int,
        default=3,
        help="Minimum keyword length after normalization.",
    )
    parser.add_argument(
        "--exclude-terms",
        default="",
        help="Comma-separated keywords to exclude after normalization.",
    )
    parser.add_argument(
        "--legacy-output-dir",
        default="",
        help="Optional directory to also write legacy file names.",
    )
    return parser.parse_args()


def build_frequency_payload(
    df: pd.DataFrame,
    min_keyword_length: int,
    exclude_terms: set[str],
) -> dict[str, Any]:
    keyword_col = detect_column(df, ["keyword", "keywords", "palavraschave", "palavras-chave"])
    frequency_col = detect_optional_column(df, ["frequency", "frequencia"])
    source_col = detect_optional_column(df, ["source", "origem"])
    year_col = detect_optional_column(df, ["year", "ano", "publicationyear"])
    doi_col = detect_optional_column(df, ["article_doi", "doi"])
    title_col = detect_optional_column(df, ["article_title", "title", "titulo"])

    non_null_kw = df[keyword_col].dropna().astype(str).str.strip()
    is_aggregated = (
        frequency_col is not None
        and len(non_null_kw) > 0
        and non_null_kw.nunique(dropna=True) == len(non_null_kw)
        and source_col is None
    )

    keyword_counter: Counter[str] = Counter()
    variant_counter: defaultdict[str, Counter[str]] = defaultdict(Counter)
    source_counter: defaultdict[str, Counter[str]] = defaultdict(Counter)
    article_coverage: defaultdict[str, set[str]] = defaultdict(set)
    year_counter: Counter[str] = Counter()
    year_keyword_counter: defaultdict[str, Counter[str]] = defaultdict(Counter)

    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        row_source = str(row_dict.get(source_col, "not_reported")).strip() if source_col else "not_reported"
        row_source = row_source if row_source else "not_reported"

        row_year = parse_year(row_dict.get(year_col, "")) if year_col else ""

        article_key = ""
        row_doi = str(row_dict.get(doi_col, "")).strip() if doi_col else ""
        row_title = str(row_dict.get(title_col, "")).strip() if title_col else ""
        if row_doi:
            article_key = row_doi.lower()
        elif row_title:
            article_key = row_title.lower()

        raw_value = row_dict.get(keyword_col)
        terms = split_keywords(raw_value)
        if not terms:
            continue

        for raw_term in terms:
            normalized = normalize_keyword(raw_term)
            if not normalized:
                continue
            if len(normalized) < min_keyword_length:
                continue
            if normalized in exclude_terms:
                continue

            weight = 1
            if is_aggregated and frequency_col:
                try:
                    weight = int(float(row_dict.get(frequency_col, 0)))
                except (TypeError, ValueError):
                    weight = 0
            if weight <= 0:
                continue

            keyword_counter[normalized] += weight
            variant_counter[normalized][normalized] += weight
            source_counter[normalized][row_source] += weight

            if article_key and not is_aggregated:
                article_coverage[normalized].add(article_key)

            if row_year:
                year_counter[row_year] += weight
                year_keyword_counter[row_year][normalized] += weight

    display_counter: Counter[str] = Counter()
    keyword_metadata: dict[str, dict[str, Any]] = {}
    for normalized, count in keyword_counter.items():
        display = variant_counter[normalized].most_common(1)[0][0]
        display_counter[display] = count

        dominant_source = "not_reported"
        if source_counter[normalized]:
            dominant_source = source_counter[normalized].most_common(1)[0][0]

        keyword_metadata[display] = {
            "dominant_source": dominant_source,
            "article_count": len(article_coverage[normalized]) if article_coverage[normalized] else None,
        }

    return {
        "frequencies": display_counter,
        "metadata": keyword_metadata,
        "year_counter": year_counter,
        "year_keyword_counter": year_keyword_counter,
        "source_counter": source_counter,
    }


def build_frequency_table(
    frequencies: Counter,
    metadata: dict[str, dict[str, Any]],
    min_frequency: int,
) -> pd.DataFrame:
    rows = []
    filtered_items = [
        (keyword, freq)
        for keyword, freq in frequencies.items()
        if freq >= min_frequency
    ]
    filtered_items.sort(key=lambda item: (-item[1], item[0]))

    total = sum(freq for _, freq in filtered_items)
    cumulative = 0.0

    for rank, (keyword, freq) in enumerate(filtered_items, start=1):
        share = (freq / total) * 100 if total else 0.0
        cumulative += share
        meta = metadata.get(keyword, {})
        rows.append(
            {
                "rank": rank,
                "keyword": keyword,
                "frequency": int(freq),
                "share_pct": round(share, 4),
                "cumulative_share_pct": round(cumulative, 4),
                "dominant_source": meta.get("dominant_source", "not_reported"),
                "article_count": meta.get("article_count"),
            }
        )

    return pd.DataFrame(rows)


def gini_coefficient(values: list[int]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    n = len(sorted_values)
    total = float(sum(sorted_values))
    if total == 0:
        return 0.0
    cumulative = 0.0
    for index, value in enumerate(sorted_values, start=1):
        cumulative += (2 * index - n - 1) * value
    return cumulative / (n * total)


def compute_summary_metrics(table_df: pd.DataFrame) -> dict[str, Any]:
    total = int(table_df["frequency"].sum())
    unique = int(len(table_df))
    top1 = float(table_df.iloc[0]["share_pct"]) if unique else 0.0
    top5 = float(table_df.head(5)["share_pct"].sum()) if unique else 0.0
    top10 = float(table_df.head(10)["share_pct"].sum()) if unique else 0.0
    top20 = float(table_df.head(20)["share_pct"].sum()) if unique else 0.0

    probs = [float(freq) / total for freq in table_df["frequency"].tolist()] if total else []
    hhi = sum(p * p for p in probs)

    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log(p)
    entropy_norm = entropy / math.log(unique) if unique > 1 else 0.0

    gini = gini_coefficient(table_df["frequency"].tolist())
    singleton_count = int((table_df["frequency"] == 1).sum())

    rank_80 = None
    over_80 = table_df[table_df["cumulative_share_pct"] >= 80]
    if not over_80.empty:
        rank_80 = int(over_80.iloc[0]["rank"])

    return {
        "total_occurrences": total,
        "unique_keywords": unique,
        "top1_share_pct": round(top1, 4),
        "top5_share_pct": round(top5, 4),
        "top10_share_pct": round(top10, 4),
        "top20_share_pct": round(top20, 4),
        "hhi": round(hhi, 6),
        "entropy_normalized": round(entropy_norm, 6),
        "gini": round(gini, 6),
        "singletons": singleton_count,
        "singleton_share_pct": round((singleton_count / unique) * 100 if unique else 0.0, 4),
        "rank_for_80pct": rank_80,
    }


def save_wordcloud_figure(
    frequencies: Counter,
    output_png: Path,
    output_svg: Path,
    title: str,
    max_words: int,
) -> None:
    wordcloud = WordCloud(
        width=3200,
        height=1900,
        background_color="white",
        colormap="viridis",
        max_words=max_words,
        collocations=False,
        relative_scaling=0.25,
        prefer_horizontal=0.92,
        random_state=42,
        min_font_size=8,
    ).generate_from_frequencies(frequencies)

    fig, ax = plt.subplots(figsize=(15, 9), dpi=350)
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=17, fontweight="semibold", loc="left", pad=12)
    ax.text(
        0.0,
        -0.05,
        "Method: frequency-based cloud, normalized keywords, random_state=42, viridis palette.",
        transform=ax.transAxes,
        fontsize=9,
        color="#444444",
        va="top",
    )

    fig.savefig(output_png, dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(output_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_top_keywords_graph(
    table_df: pd.DataFrame,
    output_png: Path,
    output_svg: Path,
    top_n: int,
) -> None:
    plot_df = table_df.head(top_n).copy()
    if plot_df.empty:
        return

    plot_df = plot_df.iloc[::-1].reset_index(drop=True)
    labels = [shorten_label(term, max_chars=50) for term in plot_df["keyword"].tolist()]

    fig = plt.figure(figsize=(18, 9), dpi=350, constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[3.2, 2.0], wspace=0.3)
    ax_left = fig.add_subplot(grid[0, 0])
    ax_right = fig.add_subplot(grid[0, 1])

    colors = plt.cm.cividis([0.25 + 0.65 * (i / max(1, len(plot_df) - 1)) for i in range(len(plot_df))])
    ax_left.barh(labels, plot_df["frequency"].tolist(), color=colors)
    ax_left.set_title(f"Top {top_n} keywords (absolute frequency)", fontsize=13, pad=10)
    ax_left.set_xlabel("Frequency")
    ax_left.tick_params(axis="y", labelsize=8)
    ax_left.grid(axis="x", alpha=0.25, linewidth=0.8)
    ax_left.set_axisbelow(True)
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)

    max_freq = max(plot_df["frequency"].tolist())
    for idx, row in enumerate(plot_df.itertuples(index=False)):
        ax_left.text(
            row.frequency + max_freq * 0.01,
            idx,
            f"{row.share_pct:.2f}%",
            va="center",
            fontsize=7,
            color="#333333",
        )

    ranks = table_df["rank"].tolist()
    cumulative = table_df["cumulative_share_pct"].tolist()
    ax_right.plot(ranks, cumulative, color="#1b4965", linewidth=2.2)
    ax_right.fill_between(ranks, cumulative, color="#5fa8d3", alpha=0.25)
    ax_right.set_title("Cumulative concentration (Pareto)", fontsize=13, pad=10)
    ax_right.set_xlabel("Keyword rank")
    ax_right.set_ylabel("Cumulative share (%)")
    ax_right.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax_right.set_ylim(0, 100)
    ax_right.grid(alpha=0.25, linewidth=0.8)

    if len(ranks) > 60:
        ax_right.set_xscale("log")

    rank_80 = None
    over_80 = table_df[table_df["cumulative_share_pct"] >= 80]
    if not over_80.empty:
        rank_80 = int(over_80.iloc[0]["rank"])
        ax_right.axhline(80, color="#8d99ae", linestyle="--", linewidth=1.2)
        ax_right.axvline(rank_80, color="#8d99ae", linestyle="--", linewidth=1.2)
        ax_right.text(
            rank_80,
            83,
            f"80% at rank {rank_80}",
            fontsize=8,
            color="#4a4a4a",
            ha="left",
        )

    fig.savefig(output_png, dpi=500, bbox_inches="tight", facecolor="white")
    fig.savefig(output_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def to_markdown_table(table_df: pd.DataFrame, top_n: int) -> str:
    lines = [
        "| Rank | Keyword | Frequency | Share (%) | Cum. Share (%) | Dominant Source |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for row in table_df.head(top_n).itertuples(index=False):
        lines.append(
            f"| {row.rank} | {row.keyword} | {row.frequency} | {row.share_pct:.2f} | {row.cumulative_share_pct:.2f} | {row.dominant_source} |"
        )
    return "\n".join(lines)


def build_yearly_highlights(year_keyword_counter: dict[str, Counter[str]]) -> list[dict[str, Any]]:
    rows = []
    for year, counter in sorted(year_keyword_counter.items()):
        if not counter:
            continue
        keyword, freq = counter.most_common(1)[0]
        rows.append({"year": year, "top_keyword": keyword, "frequency": int(freq)})
    return rows


def save_analysis_markdown(
    table_df: pd.DataFrame,
    summary: dict[str, Any],
    yearly_highlights: list[dict[str, Any]],
    output_path: Path,
    input_path: Path,
    figure_wordcloud_png: Path,
    figure_top_png: Path,
    table_full_path: Path,
    table_top_path: Path,
    params: dict[str, Any],
) -> None:
    utc_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    source_mix = table_df["dominant_source"].value_counts().to_dict()
    source_lines = "\n".join(
        [f"- `{source}`: **{count}** keywords" for source, count in source_mix.items()]
    )
    if not source_lines:
        source_lines = "- Source metadata not available in the input table."

    yearly_block = ""
    if yearly_highlights:
        yearly_table = [
            "| Year | Top Keyword | Frequency |",
            "|---:|---|---:|",
        ]
        for row in yearly_highlights:
            yearly_table.append(f"| {row['year']} | {row['top_keyword']} | {row['frequency']} |")
        yearly_block = "\n".join(yearly_table)
    else:
        yearly_block = "Year metadata was not available for trend decomposition."

    markdown = (
        "# Keyword Analysis Report\n\n"
        "## 1. Scope and Data Provenance\n"
        f"- Input file: `{input_path}`\n"
        f"- Generation timestamp: **{utc_now}**\n"
        f"- Total keyword occurrences: **{summary['total_occurrences']}**\n"
        f"- Unique keywords: **{summary['unique_keywords']}**\n\n"
        "## 2. Methods (Academic Reproducibility)\n"
        f"- Keyword normalization: lowercase, punctuation stripped, hyphen harmonized.\n"
        f"- Minimum frequency threshold: **{params['min_frequency']}**.\n"
        f"- Minimum keyword length: **{params['min_keyword_length']}** characters.\n"
        f"- Wordcloud: frequency-based (`max_words={params['max_words']}`, random_state=42).\n"
        f"- Top-keyword chart: **top {params['top_n']}** terms + Pareto concentration curve.\n\n"
        "## 3. Core Findings\n"
        f"- Top 1 share: **{summary['top1_share_pct']:.2f}%**\n"
        f"- Top 5 cumulative share: **{summary['top5_share_pct']:.2f}%**\n"
        f"- Top 10 cumulative share: **{summary['top10_share_pct']:.2f}%**\n"
        f"- Top 20 cumulative share: **{summary['top20_share_pct']:.2f}%**\n"
        f"- Rank needed to reach 80% cumulative share: **{summary['rank_for_80pct']}**\n"
        f"- Singleton keywords (frequency=1): **{summary['singletons']}** ({summary['singleton_share_pct']:.2f}% of vocabulary)\n\n"
        "## 4. Concentration Diagnostics\n"
        f"- Herfindahl-Hirschman Index (HHI): **{summary['hhi']:.6f}**\n"
        f"- Normalized Shannon entropy: **{summary['entropy_normalized']:.6f}**\n"
        f"- Gini coefficient (keyword frequency distribution): **{summary['gini']:.6f}**\n\n"
        "Interpretation guide:\n"
        "- Higher HHI and Gini indicate stronger thematic concentration.\n"
        "- Higher normalized entropy indicates broader thematic dispersion.\n\n"
        "## 5. Source Composition\n"
        f"{source_lines}\n\n"
        "## 6. Yearly Highlights\n"
        f"{yearly_block}\n\n"
        "## 7. Top Keywords Table\n\n"
        f"{to_markdown_table(table_df, top_n=30)}\n\n"
        "## 8. Generated Files\n"
        f"- Wordcloud (PNG): `{figure_wordcloud_png.name}`\n"
        f"- Top keywords chart (PNG): `{figure_top_png.name}`\n"
        f"- Full frequency table: `{table_full_path.name}`\n"
        f"- Top frequency table: `{table_top_path.name}`\n"
        "\n"
        "## 9. Limitations\n"
        "- Keyword quality depends on metadata completeness and source indexing practices.\n"
        "- API-enriched keywords may over-represent broad subject labels in some databases.\n"
        "- For inferential analysis, complement this report with full-text topic modeling.\n"
    )

    output_path.write_text(markdown, encoding="utf-8")


def write_summary_json(summary: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def maybe_write_legacy_outputs(
    legacy_dir: Path,
    wordcloud_png: Path,
    top_png: Path,
    table_full: Path,
    report_md: Path,
) -> None:
    legacy_dir.mkdir(parents=True, exist_ok=True)
    (legacy_dir / "nuvem_palavras.png").write_bytes(wordcloud_png.read_bytes())
    (legacy_dir / "grafico_top_keywords.png").write_bytes(top_png.read_bytes())
    (legacy_dir / "tabela_nuvem_palavras.csv").write_bytes(table_full.read_bytes())
    (legacy_dir / "analise_nuvem_palavras.md").write_bytes(report_md.read_bytes())


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_root = Path(args.output_root)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_figures = output_root / "figures"
    output_tables = output_root / "tables"
    output_reports = output_root / "reports"
    output_figures.mkdir(parents=True, exist_ok=True)
    output_tables.mkdir(parents=True, exist_ok=True)
    output_reports.mkdir(parents=True, exist_ok=True)

    exclude_terms = {
        normalize_keyword(term)
        for term in split_keywords(args.exclude_terms)
        if normalize_keyword(term)
    }

    df = pd.read_csv(input_path, encoding="utf-8")
    payload = build_frequency_payload(
        df=df,
        min_keyword_length=max(1, args.min_keyword_length),
        exclude_terms=exclude_terms,
    )

    table_df = build_frequency_table(
        frequencies=payload["frequencies"],
        metadata=payload["metadata"],
        min_frequency=max(1, args.min_frequency),
    )

    if table_df.empty:
        raise ValueError("No keywords found after applying filters.")

    summary = compute_summary_metrics(table_df)

    wordcloud_png = output_figures / "nuvem_palavras_academica.png"
    wordcloud_svg = output_figures / "nuvem_palavras_academica.svg"
    top_graph_png = output_figures / "grafico_top_keywords_academico.png"
    top_graph_svg = output_figures / "grafico_top_keywords_academico.svg"

    table_full_path = output_tables / "tabela_keywords_completa.csv"
    top_n_effective = max(5, args.top_n)
    table_top_path = output_tables / f"tabela_keywords_top{top_n_effective}.csv"
    report_md_path = output_reports / "analise_keywords_completa.md"
    summary_json_path = output_reports / "resumo_metricas_keywords.json"

    freq_counter = Counter(dict(zip(table_df["keyword"], table_df["frequency"])))
    save_wordcloud_figure(
        frequencies=freq_counter,
        output_png=wordcloud_png,
        output_svg=wordcloud_svg,
        title=args.title,
        max_words=max(20, args.max_words),
    )
    save_top_keywords_graph(
        table_df=table_df,
        output_png=top_graph_png,
        output_svg=top_graph_svg,
        top_n=top_n_effective,
    )

    table_df.to_csv(table_full_path, index=False, encoding="utf-8-sig")
    table_df.head(top_n_effective).to_csv(table_top_path, index=False, encoding="utf-8-sig")

    yearly_highlights = build_yearly_highlights(payload["year_keyword_counter"])
    save_analysis_markdown(
        table_df=table_df,
        summary=summary,
        yearly_highlights=yearly_highlights,
        output_path=report_md_path,
        input_path=input_path,
        figure_wordcloud_png=wordcloud_png,
        figure_top_png=top_graph_png,
        table_full_path=table_full_path,
        table_top_path=table_top_path,
        params={
            "min_frequency": max(1, args.min_frequency),
            "min_keyword_length": max(1, args.min_keyword_length),
            "max_words": max(20, args.max_words),
            "top_n": top_n_effective,
        },
    )

    write_summary_json(summary, summary_json_path)

    if args.legacy_output_dir:
        maybe_write_legacy_outputs(
            legacy_dir=Path(args.legacy_output_dir),
            wordcloud_png=wordcloud_png,
            top_png=top_graph_png,
            table_full=table_full_path,
            report_md=report_md_path,
        )

    print(f"Input used: {input_path}")
    print(f"Wordcloud (PNG): {wordcloud_png}")
    print(f"Wordcloud (SVG): {wordcloud_svg}")
    print(f"Top graph (PNG): {top_graph_png}")
    print(f"Top graph (SVG): {top_graph_svg}")
    print(f"Full table: {table_full_path}")
    print(f"Top table: {table_top_path}")
    print(f"Report: {report_md_path}")
    print(f"Summary JSON: {summary_json_path}")
    if args.legacy_output_dir:
        print(f"Legacy outputs copied to: {args.legacy_output_dir}")

    print("\nKey metrics:")
    print(f"  Total occurrences: {summary['total_occurrences']}")
    print(f"  Unique keywords: {summary['unique_keywords']}")
    print(f"  Top 10 cumulative share: {summary['top10_share_pct']:.2f}%")
    print(f"  HHI: {summary['hhi']:.6f}")
    print(f"  Gini: {summary['gini']:.6f}")


if __name__ == "__main__":
    main()

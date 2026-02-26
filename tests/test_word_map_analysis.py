import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class WordMapAnalysisCLITest(unittest.TestCase):
    def test_generates_outputs_with_auto_delimiter_and_fallback_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            input_file = base / "artigos.csv"
            output_dir = base / "output"

            input_file.write_text(
                "texto;id\n"
                "analise de dados e ciencia;\n"
                "mapa de palavras para artigo;A2\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    "python3",
                    "scripts/word_map_analysis.py",
                    "--input",
                    str(input_file),
                    "--output-dir",
                    str(output_dir),
                    "--language",
                    "pt",
                    "--delimiter",
                    "auto",
                    "--keywords-per-article",
                    "3",
                    "--cooccurrence-window",
                    "2",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("Arquivos gerados em", result.stdout)

            expected_files = [
                "word_frequency.csv",
                "document_word_matrix.csv",
                "article_keywords.csv",
                "word_map_summary.json",
                "keyword_cooccurrence.csv",
                "wordcloud.png",
            ]
            for name in expected_files:
                self.assertTrue((output_dir / name).exists(), f"Arquivo não gerado: {name}")

            with (output_dir / "article_keywords.csv").open("r", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertTrue(rows)
            # Primeiro id veio vazio e deve ser substituído por índice de linha.
            self.assertEqual(rows[0]["id"], "1")

            summary = json.loads((output_dir / "word_map_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["delimiter_used"], ";")
            self.assertEqual(summary["documents"], 2)
            self.assertGreaterEqual(summary["cooccurrence_edges"], 0)


if __name__ == "__main__":
    unittest.main()

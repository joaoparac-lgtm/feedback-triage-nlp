"""Interface de linha de comando.

Exemplos:
    python -m src.cli generate --n 300
    python -m src.cli classify --input data/sample_feedbacks.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .classifier import TextClassifier
from .config import SAMPLE_CSV, Settings
from .data_processing import (
    COL_CATEGORY,
    COL_TEXT,
    CSVLoadError,
    classify_dataframe,
    generate_sample_data,
    load_csv,
    to_hf_dataset,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Triagem de feedbacks com Hugging Face")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Gera um CSV sintético de exemplo")
    gen.add_argument("--n", type=int, default=200)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--output", type=Path, default=SAMPLE_CSV)

    cls = sub.add_parser("classify", help="Classifica um CSV em lote")
    cls.add_argument("--input", type=Path, default=SAMPLE_CSV)
    cls.add_argument("--output", type=Path, default=None)
    cls.add_argument("--text-column", default=COL_TEXT)
    cls.add_argument(
        "--save-hf-dataset",
        type=Path,
        default=None,
        help="Também salva o resultado como datasets.Dataset (Arrow)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = _build_parser().parse_args(argv)

    if args.command == "generate":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        generate_sample_data(args.n, args.seed).to_csv(args.output, index=False)
        print(f"✔ {args.n} feedbacks gerados em {args.output}")
        return 0

    try:
        settings = Settings.from_env()
        df = load_csv(args.input)
        out = classify_dataframe(
            df,
            TextClassifier(settings),
            text_column=args.text_column,
            chunk_size=settings.chunk_rows,
            low_confidence_threshold=settings.low_confidence_threshold,
            progress_callback=lambda done, total: print(f"  {done}/{total}", end="\r"),
        )
    except (CSVLoadError, RuntimeError, ValueError) as exc:
        print(f"✖ {exc}", file=sys.stderr)
        return 1

    output = args.output or args.input.with_name(f"{args.input.stem}_classificado.csv")
    out.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"\n✔ Resultado salvo em {output}")
    print(out[COL_CATEGORY].value_counts(dropna=False).to_string())
    if args.save_hf_dataset:
        to_hf_dataset(out).save_to_disk(str(args.save_hf_dataset))
        print(f"✔ Dataset Arrow salvo em {args.save_hf_dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from .analysis import LineupOwnershipAnalyzer
from .projection_pipeline import ProjectionPipeline


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MLB DFS projection pipeline")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("pipeline_artifacts"),
        help="Directory for cached data, features, and models.",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=[2022, 2023, 2024],
        help="Seasons to include in the training corpus.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Collect data and train projection models")
    train_parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore cached data and redownload everything.",
    )

    project_parser = subparsers.add_parser("project", help="Generate projections for a DraftKings slate")
    project_parser.add_argument("--slate", type=Path, required=True, help="Path to DraftKings export CSV")
    project_parser.add_argument("--date", type=_parse_date, required=True, help="Slate date (YYYY-MM-DD)")
    project_parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/projections.csv"),
        help="Destination CSV for projections.",
    )
    project_parser.add_argument(
        "--template-output",
        type=Path,
        help="Optional path for a projection CSV formatted like the projection template.",
    )

    analyze_parser = subparsers.add_parser("analyze", help="Evaluate lineup ownership metrics")
    analyze_parser.add_argument("--lineups", type=Path, required=True, help="CSV containing lineups with ownership")
    analyze_parser.add_argument(
        "--lineup-col",
        default="lineup_id",
        help="Column identifying individual lineups (default: lineup_id).",
    )
    analyze_parser.add_argument(
        "--ownership-col",
        default="ownership",
        help="Column containing ownership percentages or probabilities.",
    )
    analyze_parser.add_argument(
        "--salary-col",
        default=None,
        help="Optional column containing player salaries to sum per lineup.",
    )
    analyze_parser.add_argument(
        "--output",
        type=Path,
        help="Optional destination for the lineup analysis CSV.",
    )
    analyze_parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of lineups to display in the console (0 prints all).",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    pipeline = ProjectionPipeline(base_dir=args.data_dir, seasons=args.seasons)

    if args.command == "train":
        metrics = pipeline.train(force_refresh=args.force_refresh)
        print("Training complete:")
        for model_name, metric in metrics.items():
            print(
                f"  {model_name}: r2={metric['r2']:.3f}, RMSE={metric['rmse']:.2f}, MAE={metric['mae']:.2f}"
            )
    elif args.command == "project":
        pipeline.build_training_corpus(force_refresh=False)
        output, template_df = pipeline.generate_projections(
            args.slate, args.date, args.output, args.template_output
        )
        if args.output:
            print(f"Wrote {len(output)} player projections to {args.output}")
        if args.template_output:
            print(f"Wrote template-formatted projections to {args.template_output}")
    elif args.command == "analyze":
        analyzer = LineupOwnershipAnalyzer(
            lineup_col=args.lineup_col,
            ownership_col=args.ownership_col,
            salary_col=args.salary_col,
        )
        metrics = analyzer.analyze_file(args.lineups)
        metrics = metrics.sort_values("ownership_product", ascending=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            metrics.to_csv(args.output, index=False)
            print(f"Wrote lineup analysis to {args.output}")
        top_n = args.top if args.top is not None else 10
        if top_n == 0:
            display = metrics
        else:
            display = metrics.head(top_n)
        pd.set_option("display.float_format", lambda x: f"{x:.6f}")
        print(display.to_string(index=False))
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()

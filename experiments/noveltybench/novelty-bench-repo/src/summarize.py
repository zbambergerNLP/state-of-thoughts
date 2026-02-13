import argparse
import json
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging for summarize script."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    else:
        root_logger.setLevel(numeric_level)
    logger.setLevel(numeric_level)


def summarize(df: pd.DataFrame) -> dict:
    summary = {}

    summary["mean_distinct"] = np.mean(df["partition_scores"].map(len))
    summary["mean_utility"] = np.mean(df["utility"])

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-dir", help="Directory containing evaluation files", required=True
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("NOVELTY_BENCH_LOG_LEVEL", "INFO"),
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging level for summarization run",
    )
    args = parser.parse_args()
    configure_logging(args.log_level)

    eval_dir = args.eval_dir
    scores_path = os.path.join(eval_dir, "scores.jsonl")
    logger.info(f"Loading scores from {scores_path}")
    df = pd.read_json(os.path.join(eval_dir, "scores.jsonl"), lines=True)
    summary = summarize(df)
    summary_path = os.path.join(eval_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()

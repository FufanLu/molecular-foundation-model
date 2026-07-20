"""Aggregate compatible cross-sensory fold metrics into JSON and Markdown.

Example:
    python scripts/aggregate_cross_sensory.py \
      --metrics outputs/v3_d/fold0_metrics.json outputs/v3_d/fold1_metrics.json \
      --output-dir reports/v3_d_2fold
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, nargs="+", required=True, help="Per-fold metrics JSON files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for summary.json and summary.md.")
    return parser.parse_args()


def read_metrics(paths: list[Path]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    seen_folds: set[int] = set()
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            fold = int(payload["split"]["test_fold"])
            payload["test"]["odor"]["macro"]
            payload["test"]["taste"]["macro"]
            payload["test"]["score"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"{path} is not a current per-fold metrics file. "
                "Rerun training with the current script before aggregation."
            ) from error
        if fold in seen_folds:
            raise ValueError(f"Duplicate test fold {fold}: each fold may appear once.")
        seen_folds.add(fold)
        payload["_path"] = str(path)
        runs.append(payload)
    return sorted(runs, key=lambda run: int(run["split"]["test_fold"]))


def ensure_compatible(runs: list[dict[str, Any]]) -> None:
    reference = runs[0]
    for field in ("task_definition", "alignment"):
        expected = reference.get(field)
        for run in runs[1:]:
            if run.get(field) != expected:
                raise ValueError(f"Cannot aggregate runs with different {field}.")


def finite_values(runs: list[dict[str, Any]], modality: str, metric: str) -> list[float]:
    values = [float(run["test"][modality][metric]) for run in runs]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Non-finite test metric {modality}.{metric}; resolve it before aggregation.")
    return values


def summarize(values: list[float]) -> dict[str, Any]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def aggregate_retrieval(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Summarise per-fold retrieval probes when every run reports them."""
    if all("retrieval" not in run for run in runs):
        return None
    if any("retrieval" not in run for run in runs):
        raise ValueError("Cannot aggregate runs when only some report retrieval metrics.")
    probes = sorted(runs[0]["retrieval"])
    if any(sorted(run["retrieval"]) != probes for run in runs[1:]):
        raise ValueError("Cannot aggregate runs with different retrieval probes.")
    retrieval: dict[str, Any] = {}
    for probe in probes:
        metric_names = sorted(name for name in runs[0]["retrieval"][probe] if name != "queries")
        retrieval[probe] = {}
        for metric in metric_names:
            values = [float(run["retrieval"][probe][metric]) for run in runs]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Non-finite retrieval metric {probe}.{metric}; resolve it before aggregation.")
            retrieval[probe][metric] = summarize(values)
        retrieval[probe]["queries"] = [int(run["retrieval"][probe]["queries"]) for run in runs]
    return retrieval


def build_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    modalities = ("odor", "taste")
    test: dict[str, dict[str, dict[str, Any]]] = {}
    for modality in modalities:
        labels = sorted(runs[0]["test"][modality])
        if any(sorted(run["test"][modality]) != labels for run in runs[1:]):
            raise ValueError(f"Cannot aggregate runs with different {modality} label sets.")
        test[modality] = {
            label: summarize(finite_values(runs, modality, label))
            for label in labels
        }
    score_values = [float(run["test"]["score"]) for run in runs]
    if not all(math.isfinite(value) for value in score_values):
        raise ValueError("Non-finite test score; resolve it before aggregation.")
    summary = {
        "folds": [int(run["split"]["test_fold"]) for run in runs],
        "n_folds": len(runs),
        "task_definition": runs[0]["task_definition"],
        "alignment": runs[0]["alignment"],
        "inputs": [run["_path"] for run in runs],
        "test": {**test, "score": summarize(score_values)},
    }
    retrieval = aggregate_retrieval(runs)
    if retrieval is not None:
        summary["retrieval"] = retrieval
    return summary


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Cross-sensory fold summary",
        "",
        f"Folds: {', '.join(map(str, summary['folds']))} (n={summary['n_folds']})",
        "",
        "## Protocol",
        "",
        f"- Core taste labels: {', '.join(summary['task_definition']['core_taste_labels'])}",
        f"- Low-shot labels: {', '.join(summary['task_definition']['low_shot_taste_labels'])}",
        f"- Alignment: `{json.dumps(summary['alignment'], sort_keys=True)}`",
        "",
        "## Held-out test metrics",
        "",
        "| Task | Metric | Mean ± SD | Per fold |",
        "| --- | --- | ---: | --- |",
    ]
    for modality in ("odor", "taste"):
        for metric, values in summary["test"][modality].items():
            lines.append(
                f"| {modality} | {metric} | {values['mean']:.4f} ± {values['std']:.4f} | "
                f"{', '.join(f'{value:.4f}' for value in values['values'])} |"
            )
    score = summary["test"]["score"]
    lines.append(
        f"| combined | score | {score['mean']:.4f} ± {score['std']:.4f} | "
        f"{', '.join(f'{value:.4f}' for value in score['values'])} |"
    )
    if summary.get("retrieval"):
        lines.extend([
            "",
            "## Pair retrieval probes (test projections)",
            "",
            "| Probe | Metric | Mean ± SD | Per fold |",
            "| --- | --- | ---: | --- |",
        ])
        for probe, metrics in summary["retrieval"].items():
            for metric, values in metrics.items():
                if metric == "queries":
                    continue
                lines.append(
                    f"| {probe} | {metric} | {values['mean']:.4f} ± {values['std']:.4f} | "
                    f"{', '.join(f'{value:.4f}' for value in values['values'])} |"
                )
            lines.append(f"| {probe} | queries | — | {', '.join(map(str, metrics['queries']))} |")
    lines.extend(["", "## Inputs", ""])
    lines.extend(f"- `{path}`" for path in summary["inputs"])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    runs = read_metrics(args.metrics)
    ensure_compatible(runs)
    summary = build_summary(runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "summary.md").write_text(markdown(summary), encoding="utf-8")
    print(f"wrote {args.output_dir / 'summary.json'}")
    print(f"wrote {args.output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()

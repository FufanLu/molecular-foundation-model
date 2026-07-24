"""Run and report v4 ontology-permutation and chemistry-null controls.

The output is deliberately separated by seed, because one shuffled ontology
is not evidence.  The final report gives an empirical randomisation tail
probability for the real ontology against the ontology-permutation ensemble.

Example (20 seeds while iterating; use 0:49 for the planned final check)::

    python scripts/run_ontology_controls.py --seeds 0:19
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from glob import glob
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE = PROJECT_ROOT / "scripts" / "fingerprint_baseline_v4.py"
AGGREGATOR = PROJECT_ROOT / "scripts" / "aggregate_cross_sensory.py"


def parse_seed_spec(value: str) -> list[int]:
    """Parse comma-separated integers and inclusive ``start:end`` ranges."""
    seeds: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            start_text, end_text = part.split(":", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Descending seed range {part!r} is not supported.")
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(part))
    if not seeds or any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique, non-negative integers.")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/sensory/molecules.parquet"))
    parser.add_argument("--source-records", type=Path, default=Path("data/processed/sensory/source_records.parquet"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/v4_ontology_controls"))
    parser.add_argument("--report-root", type=Path, default=Path("reports/v4_ontology_controls"))
    parser.add_argument("--seeds", type=str, default="0:19", help="Unique seed list/ranges, e.g. 0:49 or 1,4,9.")
    parser.add_argument("--folds", type=str, default=None)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--regularisation", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42, help="Logistic-regression seed, distinct from control seeds.")
    parser.add_argument("--include-mixtures", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def aggregate(metrics_dir: Path, report_dir: Path) -> Path:
    metrics = sorted(Path(path) for path in glob(str(metrics_dir / "fold*_metrics.json")))
    if not metrics:
        raise RuntimeError(f"No fold metrics found in {metrics_dir}.")
    run([sys.executable, str(AGGREGATOR), "--metrics", *(str(path) for path in metrics), "--output-dir", str(report_dir)])
    return report_dir / "summary.json"


def score(summary_path: Path) -> tuple[float, list[float]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    values = [float(value) for value in payload["test"]["odor"]["macro"]["values"]]
    return float(payload["test"]["odor"]["macro"]["mean"]), values


def build_report(
    real_summary: Path,
    ontology_summaries: dict[int, Path],
    chemistry_summaries: dict[int, Path],
    report_root: Path,
    args: argparse.Namespace,
) -> None:
    real_mean, real_folds = score(real_summary)
    ontology = {seed: score(path) for seed, path in ontology_summaries.items()}
    chemistry = {seed: score(path) for seed, path in chemistry_summaries.items()}
    ontology_values = [value[0] for value in ontology.values()]
    chemistry_values = [value[0] for value in chemistry.values()]
    upper_tail_p = (1 + sum(value >= real_mean for value in ontology_values)) / (1 + len(ontology_values))
    percentile = sum(value <= real_mean for value in ontology_values) / len(ontology_values)
    paired_deltas = {
        str(seed): [shuffled - observed for shuffled, observed in zip(values, real_folds)]
        for seed, (_, values) in ontology.items()
    }
    report = {
        "purpose": "v4 ontology validation; ontology permutations are not chance baselines",
        "real_summary": str(real_summary),
        "ontology_permutation_summaries": {str(seed): str(path) for seed, path in ontology_summaries.items()},
        "chemistry_null_summaries": {str(seed): str(path) for seed, path in chemistry_summaries.items()},
        "real_odor_macro_f1": real_mean,
        "ontology_permutation": {
            "n_seeds": len(ontology_values),
            "mean": statistics.fmean(ontology_values),
            "std": statistics.stdev(ontology_values) if len(ontology_values) > 1 else 0.0,
            "values_by_seed": {str(seed): value[0] for seed, value in ontology.items()},
            "real_percentile": percentile,
            "upper_tail_p": upper_tail_p,
            "per_fold_delta_vs_real_by_seed": paired_deltas,
        },
        "chemistry_null": {
            "n_seeds": len(chemistry_values),
            "mean": statistics.fmean(chemistry_values),
            "std": statistics.stdev(chemistry_values) if len(chemistry_values) > 1 else 0.0,
            "values_by_seed": {str(seed): value[0] for seed, value in chemistry.items()},
        },
        "run_configuration": {
            "data": str(args.data), "source_records": str(args.source_records), "folds": args.folds,
            "radius": args.radius, "n_bits": args.n_bits, "regularisation": args.regularisation,
            "max_iter": args.max_iter, "model_seed": args.seed, "control_seeds": parse_seed_spec(args.seeds),
        },
    }
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "control_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# v4 ontology-control summary", "",
        "This report separates a descriptor-family permutation from the chemistry-null control. "
        "The former retains molecular associations with raw descriptors and must not be described as chance.", "",
        f"- Real ontology odor macro-F1: **{real_mean:.4f}**",
        f"- Ontology permutations ({len(ontology_values)} seeds): "
        f"{statistics.fmean(ontology_values):.4f} ± "
        f"{(statistics.stdev(ontology_values) if len(ontology_values) > 1 else 0.0):.4f}",
        f"- Real ontology percentile in permutation distribution: {percentile:.3f}",
        f"- Upper-tail empirical p (`permuted >= real`): {upper_tail_p:.4f}",
        f"- Chemistry null ({len(chemistry_values)} seeds): {statistics.fmean(chemistry_values):.4f} ± "
        f"{(statistics.stdev(chemistry_values) if len(chemistry_values) > 1 else 0.0):.4f}",
        "", "## Interpretation gate", "",
        "Treat the ontology as supported by this check only if the real score is in the upper tail of the "
        "ontology-permutation distribution (predeclared target: at least the 95th percentile) and exceeds "
        "the chemistry-null distribution. Otherwise report the families as heuristic descriptor bins.", "",
        "## Inputs", "", f"- Real: `{real_summary}`",
    ]
    lines.extend(f"- Ontology permutation seed {seed}: `{path}`" for seed, path in ontology_summaries.items())
    lines.extend(f"- Chemistry-null seed {seed}: `{path}`" for seed, path in chemistry_summaries.items())
    (report_root / "control_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def baseline_command(args: argparse.Namespace, output_dir: Path, control: tuple[str, int] | None = None) -> list[str]:
    command = [
        sys.executable, str(BASELINE), "--data", str(args.data), "--source-records", str(args.source_records),
        "--output-dir", str(output_dir), "--radius", str(args.radius), "--n-bits", str(args.n_bits),
        "--regularisation", str(args.regularisation), "--max-iter", str(args.max_iter), "--seed", str(args.seed),
    ]
    if args.folds is not None:
        command.extend(["--folds", args.folds])
    if args.include_mixtures:
        command.append("--include-mixtures")
    if args.force:
        command.append("--force")
    if control is not None:
        command.extend([control[0], str(control[1])])
    return command


def main() -> None:
    args = parse_args()
    seeds = parse_seed_spec(args.seeds)
    run(baseline_command(args, args.output_root / "real"))
    real_summary = aggregate(args.output_root / "real", args.report_root / "real")
    ontology_summaries: dict[int, Path] = {}
    chemistry_summaries: dict[int, Path] = {}
    for seed in seeds:
        ontology_dir = args.output_root / f"ontology_shuffle_seed{seed}"
        chemistry_dir = args.output_root / f"odor_train_permutation_seed{seed}"
        run(baseline_command(args, ontology_dir, ("--shuffle-ontology", seed)))
        ontology_summaries[seed] = aggregate(ontology_dir, args.report_root / ontology_dir.name)
        run(baseline_command(args, chemistry_dir, ("--shuffle-odor-train-labels", seed)))
        chemistry_summaries[seed] = aggregate(chemistry_dir, args.report_root / chemistry_dir.name)
    build_report(real_summary, ontology_summaries, chemistry_summaries, args.report_root, args)
    print(f"wrote {args.report_root / 'control_summary.json'}")
    print(f"wrote {args.report_root / 'control_summary.md'}")


if __name__ == "__main__":
    main()

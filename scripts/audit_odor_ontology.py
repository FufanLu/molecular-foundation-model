"""Audit the odor-family ontology against the prepared corpus.

Four checks, printed as a Markdown report:

1. Alias overlap — raw terms assigned to more than one family (forced label
   coupling; see docs/ODOR_ONTOLOGY.md known issue 1).
2. Coverage — how much of the raw descriptor evidence the ontology maps, and
   the most frequent unmapped terms (input for the next revision).
3. Co-occurrence lift — whether same-family descriptor pairs co-occur on
   molecules more than cross-family pairs.  A family whose within-lift does
   not exceed the cross-family baseline is a grouping the corpus does not
   support.
4. Dual-membership impact — how many molecules carry the overlapping term
   and therefore receive both coupled labels.

Usage:
    python scripts/audit_odor_ontology.py \
      --data data/processed/sensory/molecules.parquet \
      --source-records data/processed/sensory/source_records.parquet
"""

from __future__ import annotations

import argparse
import itertools
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.dataset.sensory import ODOR_FAMILIES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/sensory/molecules.parquet"))
    parser.add_argument("--source-records", type=Path, default=Path("data/processed/sensory/source_records.parquet"))
    parser.add_argument("--min-support", type=int, default=5,
                        help="Minimum molecules carrying a term for it to enter co-occurrence statistics.")
    parser.add_argument("--output", type=Path, default=None, help="Optional Markdown report path.")
    return parser.parse_args()


def molecule_terms(source_records: pd.DataFrame) -> dict[str, set[str]]:
    """Union of raw odor terms per canonical molecule."""
    grouped: dict[str, set[str]] = {}
    for _, row in source_records.iterrows():
        smiles = row["canonical_smiles"]
        if not isinstance(smiles, str) or not smiles:
            continue
        terms = row["odor_terms"] if isinstance(row["odor_terms"], (list, tuple)) else []
        grouped.setdefault(smiles, set()).update(terms)
    return grouped


def alias_overlap() -> list[str]:
    lines = ["## 1. Alias overlap (terms in more than one family)", ""]
    seen: dict[str, list[str]] = {}
    for family, aliases in ODOR_FAMILIES.items():
        for term in aliases:
            seen.setdefault(term, []).append(family)
    overlaps = {term: families for term, families in seen.items() if len(families) > 1}
    if overlaps:
        for term, families in sorted(overlaps.items()):
            lines.append(f"- `{term}` → {', '.join(families)} (forced co-labelling)")
    else:
        lines.append("None.")
    lines.append("")
    return lines


def coverage(term_sets: dict[str, set[str]]) -> tuple[list[str], Counter]:
    mapped_terms = set().union(*ODOR_FAMILIES.values())
    df_all = Counter(term for terms in term_sets.values() for term in terms)
    unmapped = Counter({term: count for term, count in df_all.items() if term not in mapped_terms})
    total = sum(df_all.values())
    mapped_count = total - sum(unmapped.values())
    lines = ["## 2. Coverage", ""]
    lines.append(
        f"- Molecules with odor evidence: {len(term_sets):,}; descriptor tokens: {total:,}; "
        f"mapped: {mapped_count:,} ({mapped_count / max(total, 1):.1%})"
    )
    lines.append("- Top unmapped terms by molecule frequency (revision candidates):")
    lines.append("")
    lines.append("| Unmapped term | Molecules |")
    lines.append("| --- | ---: |")
    for term, count in unmapped.most_common(20):
        lines.append(f"| {term} | {count} |")
    lines.append("")
    return lines, unmapped


def cooccurrence_lift(term_sets: dict[str, set[str]], min_support: int) -> list[str]:
    df = Counter(term for terms in term_sets.values() for term in terms)
    supported = {term for term, count in df.items() if count >= min_support}
    n_molecules = len(term_sets)
    pair_df: Counter = Counter()
    for terms in term_sets.values():
        for a, b in itertools.combinations(sorted(terms & supported), 2):
            pair_df[(a, b)] += 1

    def lift(a: str, b: str) -> float:
        observed = pair_df.get((a, b), 0) or pair_df.get((b, a), 0)
        if observed == 0:
            return 0.0
        return observed * n_molecules / (df[a] * df[b])

    family_of = {term: family for family, aliases in ODOR_FAMILIES.items() for term in aliases}
    within: dict[str, list[float]] = {family: [] for family in ODOR_FAMILIES}
    cross: list[float] = []
    for a, b in itertools.combinations(sorted(supported), 2):
        fa, fb = family_of.get(a), family_of.get(b)
        value = lift(a, b)
        if fa and fb and fa == fb:
            within[fa].append(value)
        elif fa or fb:
            cross.append(value)
    cross_baseline = sum(cross) / len(cross) if cross else 0.0

    lines = ["## 3. Co-occurrence lift (within-family vs cross-family)", ""]
    lines.append(
        f"Lift = P(a∩b) / (P(a)·P(b)) at molecule level, terms with ≥{min_support} molecules. "
        f"Cross-family baseline mean lift: **{cross_baseline:.2f}**."
    )
    lines.append("")
    lines.append("| Family | Member terms (supported) | Mean within-family lift | Verdict |")
    lines.append("| --- | ---: | ---: | --- |")
    for family, aliases in ODOR_FAMILIES.items():
        values = within[family]
        mean_within = sum(values) / len(values) if values else float("nan")
        verdict = "ok" if values and mean_within > cross_baseline else "**review**"
        supported_count = len(aliases & supported)
        lines.append(f"| {family} | {supported_count}/{len(aliases)} | {mean_within:.2f} | {verdict} |")
    dead = {family: sorted(aliases - supported) for family, aliases in ODOR_FAMILIES.items() if aliases - supported}
    if dead:
        lines.append("")
        lines.append("Terms below the support threshold (dead weight or rare evidence):")
        for family, terms in sorted(dead.items()):
            lines.append(f"- {family}: {', '.join(terms)}")
    lines.append("")
    return lines


def dual_membership_impact(term_sets: dict[str, set[str]], molecules: pd.DataFrame) -> list[str]:
    seen: dict[str, list[str]] = {}
    for family, aliases in ODOR_FAMILIES.items():
        for term in aliases:
            seen.setdefault(term, []).append(family)
    overlaps = {term: families for term, families in seen.items() if len(families) > 1}
    lines = ["## 4. Dual-membership impact", ""]
    if not overlaps:
        lines.append("No overlapping terms.")
        lines.append("")
        return lines
    for term, families in sorted(overlaps.items()):
        carriers = sum(1 for terms in term_sets.values() if term in terms)
        lines.append(f"- `{term}` → {', '.join(families)}: {carriers} molecules receive every coupled label.")
        for family in families:
            family_total = int(molecules["odor_labels"].apply(lambda labels: family in list(labels)).sum())
            if family_total:
                lines.append(f"  - {family}: {carriers / family_total:.1%} of its {family_total:,} labelled molecules come via `{term}`.")
    lines.append("")
    return lines


def main() -> None:
    args = parse_args()
    source_records = pd.read_parquet(args.source_records)
    molecules = pd.read_parquet(args.data)
    term_sets = molecule_terms(source_records)
    lines = ["# Odor ontology audit", ""]
    lines += alias_overlap()
    coverage_lines, _ = coverage(term_sets)
    lines += coverage_lines
    lines += cooccurrence_lift(term_sets, args.min_support)
    lines += dual_membership_impact(term_sets, molecules)
    report = "\n".join(lines) + "\n"
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

"""Data contract and preparation pipeline for cross-sensory molecular learning.

The three bundled sources use different descriptor conventions.  This module
keeps their raw evidence intact, adds a versioned odor/taste ontology, and
aggregates records only after structure standardisation.  It deliberately does
not treat a missing descriptor as a negative label until a task-specific target
matrix is created.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


SCHEMA_VERSION = "sensory-v3"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "leffingwell"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "sensory"

# The main odor task is intentionally finer than the previous five-class
# prototype, while keeping terms that do not map to a family as raw evidence.
ODOR_FAMILIES: dict[str, set[str]] = {
    "fruity": {
        "fruity", "apple", "apricot", "banana", "berry", "cherry", "citrus",
        "coconut", "grape", "melon", "peach", "pear", "pineapple", "plum",
        "strawberry", "tropical",
    },
    "floral": {
        "floral", "geranium", "hyacinth", "jasmine", "lavender", "lily",
        "muguet", "neroli", "orange flower", "rose", "violet",
    },
    "green": {
        "cucumber", "fresh", "grassy", "green", "herbal", "leafy", "vegetable",
    },
    "woody": {
        "amber", "balsamic", "cedar", "earthy", "mossy", "oak", "pine",
        "sandalwood", "woody",
    },
    "fatty": {
        "buttery", "cheesy", "creamy", "fatty", "oily", "rancid", "waxy",
    },
    "sulfurous": {
        "alliaceous", "brothy", "burnt", "cooked", "garlic", "meaty", "onion",
        "roasted", "sulfur", "sulfurous",
    },
    "spicy": {
        "anise", "anisic", "cinnamon", "clove", "ginger", "nutmeg", "pepper",
        "spicy",
    },
    "sweet_aromatic": {
        "caramel", "chocolate", "cocoa", "honey", "sweet", "vanilla",
    },
    "nutty": {"almond", "cocoa", "nut skin", "nutty", "walnut"},
    "animalic": {"animal", "fishy", "leather", "musky", "sweaty"},
    "phenolic": {"medicinal", "phenolic", "smoky", "tobacco"},
    "aldehydic": {"aldehydic", "ethereal", "metallic", "pungent", "sharp"},
}

# Only these labels define the main supervised taste task.  ``sour`` and
# ``salty`` remain fully auditable in the prepared dataset, but their small
# curated support makes them low-shot probes rather than training targets.
TASTE_LABELS = ("sweet", "bitter", "umami")
SOUR_LABEL = "sour"
SALT_LABEL = "salty"
LOW_SHOT_TASTE_LABELS = (SOUR_LABEL, SALT_LABEL)
ALL_TASTE_LABELS = (*TASTE_LABELS, *LOW_SHOT_TASTE_LABELS)
TASTE_TERMS = {
    "sweet": {"sweet", "sweetness"},
    "bitter": {"bitter", "bitterness"},
    "sour": {"acidic", "sour", "sourness"},
    "umami": {"savory", "umami", "umami taste", "umaminess"},
    "salty": {"salt", "salty", "saltiness"},
}


def _normalise_term(value: object) -> str:
    """Normalise a descriptor without discarding the original descriptor."""
    text = str(value or "").strip().lower()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def _optional_text(value: object) -> str | None:
    """Convert scalar source values to text without turning missing values into ``nan``."""
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return None if text.lower() in {"", "*", "nan", "none", "<na>", "n/a"} else text


def _split_delimited(value: object, delimiter: str = ";") -> list[str]:
    return [
        term
        for term in (_normalise_term(part) for part in str(value or "").split(delimiter))
        if term
    ]


def _parse_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [_normalise_term(item) for item in value if _normalise_term(item)]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return _split_delimited(value, ",")
    if not isinstance(parsed, list):
        return []
    return [_normalise_term(item) for item in parsed if _normalise_term(item)]


def _map_terms(terms: Iterable[str], taxonomy: dict[str, set[str]]) -> list[str]:
    mapped: set[str] = set()
    for term in terms:
        normalised = _normalise_term(term)
        for label, aliases in taxonomy.items():
            if normalised in aliases:
                mapped.add(label)
    return sorted(mapped)


def map_odor_terms(terms: Iterable[str]) -> list[str]:
    """Map source descriptors to the versioned odor-family ontology."""
    return _map_terms(terms, ODOR_FAMILIES)


def map_taste_terms(terms: Iterable[str]) -> list[str]:
    """Map taste descriptors to basic-taste labels, including low-shot salty."""
    return _map_terms(terms, TASTE_TERMS)


def _chem_tastes_labels(class_taste: object, taste_description: object) -> list[str]:
    """Extract curated basic tastes from ChemTastesDB's class and detail fields.

    ``Class taste`` is authoritative whenever it names a basic taste.  The
    ``Multitaste`` class is expanded from its documented free-text taste field;
    explicit negative statements such as ``Non-sweet`` are removed first.
    """
    class_label = _normalise_term(class_taste)
    labels = map_taste_terms([class_label])
    if labels or class_label not in {"multitaste", "miscellaneous"}:
        return labels

    text = _normalise_term(taste_description)
    text = re.sub(r"\b(?:non|not|lacking|less)\s*-?\s*(?:sweet|bitter)\b", "", text)
    terms: list[str] = []
    for label, patterns in {
        "sweet": (r"\bsweet",),
        "bitter": (r"\bbitter",),
        "sour": (r"\bsour", r"\bacidic"),
        "umami": (r"\bumami", r"\bmsg-like", r"\bbrothy"),
        "salty": (r"\bsalty",),
    }.items():
        if any(re.search(pattern, text) for pattern in patterns):
            terms.append(label)
    return sorted(set(terms))


def _rdkit_modules():
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError as error:  # pragma: no cover - dependency error message
        raise ImportError("RDKit is required to prepare the sensory dataset.") from error
    return Chem, MurckoScaffold


def standardise_smiles(smiles: object) -> dict[str, object]:
    """Return a structure-preserving identifier record for a source SMILES.

    Dot-disconnected salts and mixtures are retained.  The ``is_mixture`` flag
    lets the salty probe keep ionic formulations while the main Uni-Mol task can
    make an explicit inclusion decision.
    """
    original = _optional_text(smiles) or ""
    return dict(_standardise_smiles_cached(original))


@lru_cache(maxsize=100_000)
def _standardise_smiles_cached(original: str) -> tuple[tuple[str, object], ...]:
    """Cache structure work because overlapping sources repeat many SMILES."""
    Chem, MurckoScaffold = _rdkit_modules()
    if not original:
        return tuple({
            "canonical_smiles": None,
            "molecule_id": None,
            "scaffold": None,
            "formal_charge": None,
            "fragment_count": 0,
            "is_mixture": False,
            "structure_status": "missing_smiles",
        }.items())
    molecule = Chem.MolFromSmiles(original)
    if molecule is None:
        return tuple({
            "canonical_smiles": None,
            "molecule_id": None,
            "scaffold": None,
            "formal_charge": None,
            "fragment_count": 0,
            "is_mixture": "." in original,
            "structure_status": "invalid_smiles",
        }.items())
    canonical = Chem.MolToSmiles(molecule, isomericSmiles=True, canonical=True)
    try:
        molecule_id = Chem.MolToInchiKey(molecule)
    except Exception:  # Some unusual inorganic records do not have InChI support.
        molecule_id = f"smiles:{canonical}"
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=True)
    return tuple({
        "canonical_smiles": canonical,
        "molecule_id": molecule_id,
        "scaffold": scaffold or canonical,
        "formal_charge": int(sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())),
        "fragment_count": len(Chem.GetMolFrags(molecule)),
        "is_mixture": len(Chem.GetMolFrags(molecule)) > 1,
        "structure_status": "valid",
    }.items())


def _source_records(raw_dir: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []

    leffingwell = pd.read_csv(raw_dir / "leffingwell_merged.csv")
    for _, row in leffingwell.iterrows():
        odor_terms = _parse_list(row.get("Labels"))
        records.append(
            _record_from_row(
                row,
                "leffingwell",
                odor_terms,
                [],
                raw_odor_evidence=[row.get("Raw Labels"), row.get("Labels")],
            )
        )

    goodscents = pd.read_csv(raw_dir / "goodscents_merged.csv")
    for _, row in goodscents.iterrows():
        odor_terms = _split_delimited(row.get("Descriptors"))
        records.append(
            _record_from_row(
                row,
                "goodscents",
                odor_terms,
                [],
                raw_odor_evidence=[row.get("Descriptors")],
            )
        )

    flavordb = pd.read_csv(raw_dir / "flavordb_merged.csv")
    for _, row in flavordb.iterrows():
        odor_terms = _split_delimited(row.get("Odor Percepts")) + _split_delimited(
            row.get("Odor Modifiers")
        )
        taste_terms = _split_delimited(row.get("Flavor Percepts")) + _split_delimited(
            row.get("Flavor Modifiers")
        )
        records.append(
            _record_from_row(
                row,
                "flavordb",
                odor_terms,
                taste_terms,
                raw_odor_evidence=[row.get("Odor Percepts"), row.get("Odor Modifiers")],
                raw_taste_evidence=[row.get("Flavor Percepts"), row.get("Flavor Modifiers")],
            )
        )
    return records


def _find_column(columns: Sequence[str], candidates: Sequence[str]) -> str:
    by_normalised = {_normalise_term(column): column for column in columns}
    for candidate in candidates:
        if _normalise_term(candidate) in by_normalised:
            return by_normalised[_normalise_term(candidate)]
    raise ValueError(f"Could not find any of {list(candidates)} in columns {list(columns)}")


def _chem_tastes_records(path: Path) -> list[dict[str, object]]:
    """Load the optional public ChemTastesDB workbook without baking it into git."""
    frame = pd.read_excel(path)
    smiles_column = _find_column(frame.columns, ("canonical smiles", "smiles", "smile"))
    taste_column = _find_column(frame.columns, ("taste",))
    class_column = _find_column(frame.columns, ("class taste", "taste class"))
    cid_column = next((column for column in frame.columns if _normalise_term(column) == "pubchem cid"), None)
    name_column = next((column for column in frame.columns if _normalise_term(column) == "name"), None)
    records: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        row_like = {
            "IsomericSMILES": row.get(smiles_column),
            "CID": row.get(cid_column) if cid_column else None,
            "name": row.get(name_column) if name_column else None,
        }
        records.append(
            _record_from_row(
                row_like,
                "chem_tastes",
                [],
                [],
                strong_taste_labels=_chem_tastes_labels(
                    row.get(class_column), row.get(taste_column)
                ),
                raw_taste_evidence=[row.get(class_column), row.get(taste_column)],
            )
        )
    return records


def _record_from_row(
    row: object,
    source: str,
    odor_terms: Sequence[str],
    taste_terms: Sequence[str],
    strong_taste_labels: Sequence[str] = (),
    raw_odor_evidence: Sequence[object] = (),
    raw_taste_evidence: Sequence[object] = (),
) -> dict[str, object]:
    values = row if isinstance(row, dict) else row.to_dict()
    structure = standardise_smiles(values.get("IsomericSMILES"))
    return {
        "source": source,
        "source_cid": _optional_text(values.get("CID")),
        "name": _optional_text(values.get("name")),
        "input_smiles": _optional_text(values.get("IsomericSMILES")),
        "odor_terms": sorted(set(odor_terms)),
        "taste_terms": sorted(set(taste_terms) | set(strong_taste_labels)),
        "taste_weak_terms": sorted(set(taste_terms)),
        "taste_strong_terms": sorted(set(strong_taste_labels)),
        "raw_odor_evidence": [str(value) for value in raw_odor_evidence if pd.notna(value) and str(value).strip()],
        "raw_taste_evidence": [str(value) for value in raw_taste_evidence if pd.notna(value) and str(value).strip()],
        "odor_labels": map_odor_terms(odor_terms),
        # FlavorDB's flavour wording is retained as weak evidence only.
        "taste_weak_labels": map_taste_terms(taste_terms),
        # ChemTastesDB's class field defines the supervised basic-taste task.
        "taste_strong_labels": sorted(set(strong_taste_labels)),
        **structure,
    }


def _unique_lists(values: Iterable[Iterable[str]]) -> list[str]:
    return sorted({item for value in values for item in value if item})


def aggregate_records(raw_records: pd.DataFrame) -> pd.DataFrame:
    """Aggregate valid source records by standardised molecular identifier."""
    valid = raw_records[raw_records["structure_status"] == "valid"].copy()
    cid_structure_counts = valid.groupby(["source", "source_cid"])["molecule_id"].nunique()
    conflicted_cids = {
        tuple(index)
        for index, count in cid_structure_counts.items()
        if index[1] is not None and count > 1
    }
    molecules: list[dict[str, object]] = []
    for molecule_id, group in valid.groupby("molecule_id", sort=False):
        first = group.iloc[0]
        odor_terms = _unique_lists(group["odor_terms"])
        taste_terms = _unique_lists(group["taste_terms"])
        taste_weak_terms = _unique_lists(group["taste_weak_terms"])
        taste_strong_terms = _unique_lists(group["taste_strong_terms"])
        raw_odor_evidence = _unique_lists(group["raw_odor_evidence"])
        raw_taste_evidence = _unique_lists(group["raw_taste_evidence"])
        odor_labels = _unique_lists(group["odor_labels"])
        taste_weak_labels = _unique_lists(group["taste_weak_labels"])
        taste_strong_labels = _unique_lists(group["taste_strong_labels"])
        source_cids = [
            {"source": row.source, "cid": row.source_cid}
            for row in group[["source", "source_cid"]].drop_duplicates().itertuples(index=False)
        ]
        molecules.append(
            {
                "schema_version": SCHEMA_VERSION,
                "molecule_id": molecule_id,
                "canonical_smiles": first.canonical_smiles,
                "scaffold": first.scaffold,
                "formal_charge": int(first.formal_charge),
                "fragment_count": int(first.fragment_count),
                "is_mixture": bool(first.is_mixture),
                "sources": sorted(group["source"].unique().tolist()),
                "source_cids": source_cids,
                "names": sorted({name for name in group["name"] if name}),
                "odor_terms": odor_terms,
                "taste_terms": taste_terms,
                "taste_weak_terms": taste_weak_terms,
                "taste_strong_terms": taste_strong_terms,
                "raw_odor_evidence": raw_odor_evidence,
                "raw_taste_evidence": raw_taste_evidence,
                "odor_labels": odor_labels,
                # ``taste_labels`` remains a concise alias for the curated
                # labels, so downstream supervised code cannot accidentally
                # train on FlavorDB's weak descriptor occurrences.
                "taste_labels": taste_strong_labels,
                "taste_weak_labels": taste_weak_labels,
                "taste_strong_labels": taste_strong_labels,
                "odor_known": bool(odor_labels),
                # ``taste_known`` drives the main three-label task.  Sour and
                # salty are retained separately for auditable low-shot probes.
                "taste_known": bool(set(taste_strong_labels) & set(TASTE_LABELS)),
                "taste_any_known": bool(taste_strong_labels),
                "taste_weak_known": bool(set(taste_weak_labels) & set(TASTE_LABELS)),
                "sour_known": SOUR_LABEL in taste_strong_labels,
                "sour_weak_known": SOUR_LABEL in taste_weak_labels,
                "salty_known": SALT_LABEL in taste_strong_labels,
                "salty_weak_known": SALT_LABEL in taste_weak_labels,
                "paired": bool(odor_labels) and bool(set(taste_strong_labels) & set(TASTE_LABELS)),
                "weak_paired": bool(odor_labels) and bool(set(taste_weak_labels) & set(TASTE_LABELS)),
                "cid_structure_conflict": any(
                    (row.source, row.source_cid) in conflicted_cids
                    for row in group[["source", "source_cid"]].itertuples(index=False)
                ),
                "source_record_count": len(group),
            }
        )
    return pd.DataFrame(molecules)


def build_masked_targets(
    records: pd.DataFrame,
    labels: Sequence[str],
    label_column: str,
    known_column: str,
) -> "object":
    """Build a multi-label target matrix with ``-1`` for unknown labels.

    A source not annotating a modality is *not* a negative example.  Rows with
    a true annotation receive 0/1 targets for every class in that modality.
    The returned NumPy array deliberately avoids a torch dependency in the
    preparation layer.
    """
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover
        raise ImportError("NumPy is required to build target matrices.") from error
    targets = np.full((len(records), len(labels)), -1.0, dtype="float32")
    for position, (_, row) in enumerate(records.iterrows()):
        if not bool(row.get(known_column, False)):
            continue
        observed = set(row.get(label_column, []))
        targets[position] = [1.0 if label in observed else 0.0 for label in labels]
    return targets


def assign_scaffold_folds(records: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Create deterministic, multilabel-balanced scaffold-disjoint folds.

    Ordinary ``GroupKFold`` prevents scaffold leakage but does not balance a
    rare task such as exact odor--taste pairs.  This greedy grouped allocator
    balances molecule count, odor labels, all curated taste labels, pairs, and
    low-shot sour/salty examples while never splitting a scaffold between folds.
    """
    if records.empty:
        records = records.copy()
        records["fold"] = pd.Series(dtype="int64")
        return records
    groups = records["scaffold"].fillna(records["molecule_id"])
    unique_groups = groups.nunique()
    if unique_groups < 2:
        raise ValueError("At least two scaffold groups are required for a split.")
    effective_splits = min(n_splits, unique_groups)
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover
        raise ImportError("NumPy is required for multilabel scaffold folds.") from error

    feature_names = (
        ["molecule"]
        + [f"odor:{label}" for label in ODOR_FAMILIES]
        + [f"taste:{label}" for label in ALL_TASTE_LABELS]
        + ["paired"]
    )
    feature_weights = np.array(
        [1.0]
        + [0.35] * len(ODOR_FAMILIES)
        + [1.0] * len(ALL_TASTE_LABELS)
        + [3.0],
        dtype="float64",
    )
    row_features = np.zeros((len(records), len(feature_names)), dtype="float64")
    row_features[:, 0] = 1.0
    for row_index, (_, row) in enumerate(records.iterrows()):
        odor_labels = set(row["odor_labels"])
        taste_labels = set(row["taste_strong_labels"])
        for label_index, label in enumerate(ODOR_FAMILIES, start=1):
            row_features[row_index, label_index] = float(label in odor_labels)
        taste_start = 1 + len(ODOR_FAMILIES)
        for label_index, label in enumerate(ALL_TASTE_LABELS, start=taste_start):
            row_features[row_index, label_index] = float(label in taste_labels)
        row_features[row_index, -1] = float(row["paired"])

    grouped_positions: dict[str, list[int]] = {}
    for position, group in enumerate(groups.astype(str)):
        grouped_positions.setdefault(group, []).append(position)
    group_items = [
        (group, positions, row_features[positions].sum(axis=0))
        for group, positions in grouped_positions.items()
    ]
    target = row_features.sum(axis=0) / effective_splits
    target[target == 0] = 1.0
    # Assign large and label-rich scaffolds first.  The group name makes ties
    # reproducible across pandas/RDKit versions.
    group_items.sort(
        key=lambda item: (
            -len(item[1]),
            -float((item[2] / target * feature_weights).sum()),
            item[0],
        )
    )
    fold_features = np.zeros((effective_splits, len(feature_names)), dtype="float64")
    assignments = np.empty(len(records), dtype="int64")
    for group_position, (group, positions, group_feature) in enumerate(group_items):
        # Seed every fold with one of the largest scaffolds.  Without this,
        # a local greedy objective can leave a fold empty.
        if group_position < effective_splits:
            selected_fold = group_position
            assignments[positions] = selected_fold
            fold_features[selected_fold] += group_feature
            continue
        scores: list[tuple[float, float, int]] = []
        for fold in range(effective_splits):
            projected_all = fold_features.copy()
            projected_all[fold] += group_feature
            balance_cost = float(
                ((((projected_all - target) / target) ** 2) * feature_weights).sum()
            )
            size_cost = float(fold_features[fold, 0])
            scores.append((balance_cost, size_cost, fold))
        _, _, selected_fold = min(scores)
        assignments[positions] = selected_fold
        fold_features[selected_fold] += group_feature

    result = records.copy()
    result["fold"] = assignments.astype(int)
    return result


def build_audit(raw_records: pd.DataFrame, molecules: pd.DataFrame) -> dict[str, object]:
    """Create JSON-serialisable audit information for the project data card."""
    source_counts = raw_records.groupby("source").size().to_dict()
    source_valid = (
        raw_records.assign(valid=raw_records["structure_status"].eq("valid"))
        .groupby("source")["valid"]
        .sum()
        .astype(int)
        .to_dict()
    )
    odor_counts = Counter(label for labels in molecules["odor_labels"] for label in labels)
    taste_counts = Counter(label for labels in molecules["taste_strong_labels"] for label in labels)
    weak_taste_counts = Counter(label for labels in molecules["taste_weak_labels"] for label in labels)
    return {
        "schema_version": SCHEMA_VERSION,
        "raw_record_count": int(len(raw_records)),
        "unique_molecule_count": int(len(molecules)),
        "source_record_counts": {source: int(count) for source, count in source_counts.items()},
        "source_valid_structure_counts": {source: int(count) for source, count in source_valid.items()},
        "paired_molecule_count": int(molecules["paired"].sum()),
        "odor_labeled_molecule_count": int(molecules["odor_known"].sum()),
        "core_taste_labeled_molecule_count": int(molecules["taste_known"].sum()),
        "curated_taste_any_labeled_molecule_count": int(molecules["taste_any_known"].sum()),
        "weak_taste_labeled_molecule_count": int(molecules["taste_weak_known"].sum()),
        "sour_molecule_count": int(molecules["sour_known"].sum()),
        "weak_sour_molecule_count": int(molecules["sour_weak_known"].sum()),
        "salty_molecule_count": int(molecules["salty_known"].sum()),
        "weak_salty_molecule_count": int(molecules["salty_weak_known"].sum()),
        "weak_paired_molecule_count": int(molecules["weak_paired"].sum()),
        "mixture_molecule_count": int(molecules["is_mixture"].sum()),
        "cid_structure_conflict_count": int(molecules["cid_structure_conflict"].sum()),
        "odor_label_counts": dict(sorted(odor_counts.items())),
        "taste_label_counts": dict(sorted(taste_counts.items())),
        "weak_taste_label_counts": dict(sorted(weak_taste_counts.items())),
        "fold_summary": {
            str(fold): {
                "molecules": int(len(group)),
                "odor_labeled": int(group["odor_known"].sum()),
                "core_taste_labeled": int(group["taste_known"].sum()),
                "paired": int(group["paired"].sum()),
                "sour": int(group["sour_known"].sum()),
                "salty": int(group["salty_known"].sum()),
            }
            for fold, group in molecules.groupby("fold", sort=True)
        },
    }


def prepare_sensory_dataset(
    raw_dir: Path | str = RAW_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    chem_tastes_path: Path | str | None = None,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Prepare records, five global scaffold folds, and an auditable data card."""
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    records = _source_records(raw_dir)
    if chem_tastes_path:
        records.extend(_chem_tastes_records(Path(chem_tastes_path)))
    raw_frame = pd.DataFrame(records)
    molecules = assign_scaffold_folds(aggregate_records(raw_frame), n_splits=n_splits)
    audit = build_audit(raw_frame, molecules)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_frame.to_parquet(output_dir / "source_records.parquet", index=False)
    molecules.to_parquet(output_dir / "molecules.parquet", index=False)
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return molecules, audit


def _main() -> None:
    # ChemTastesDB contains a few unusual ionic/hydrogen records.  Their RDKit
    # warning is non-fatal and otherwise floods Colab output; parse errors are
    # deliberately left enabled and remain visible in the audit.
    try:
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.warning")
    except ImportError:
        pass
    parser = argparse.ArgumentParser(description="Prepare cross-sensory molecular records.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chem-tastes", type=Path, default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()
    _, audit = prepare_sensory_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        chem_tastes_path=args.chem_tastes,
        n_splits=args.n_splits,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()

# Ontology validation: v4 decision record

## Status

**Implementation complete; v4 data and controls have not yet been run.** This
document is the hand-off record for the next agent or operator. Do not replace
this status with an inference from v3 outputs.

## Why v3 was not accepted as ontology validation

The v3 Morgan-fingerprint five-fold check completed on 2026-07-24 reported:

| Odor macro-F1 | Mean ± SD | Per fold |
| --- | ---: | --- |
| Real v3 ontology | 0.4849 ± 0.0069 | 0.4831, 0.4968, 0.4827, 0.4827, 0.4790 |
| One shuffled ontology (seed 0) | 0.5120 ± 0.0162 | 0.4925, 0.5359, 0.5056, 0.5185, 0.5078 |

The shuffled score exceeded the real score in all five paired folds. This does
not establish data leakage and does not invalidate structure-to-descriptor
prediction. It does invalidate the prior proposed claim that this single
shuffle would "collapse toward chance" and thereby validate the family
ontology.

The old shuffle preserved every molecule's raw descriptor evidence while
changing only descriptor-to-family membership. A random family grouping can
have different prevalence, long-tail burden, and chemical predictability; it
is an **ontology permutation**, not a chance/null baseline. Per-label
validation threshold selection further makes macro-F1 sensitive to those task
definition changes.

## v4 target-definition change

`src/dataset/sensory.py` now identifies the processed schema as `sensory-v4`
and removes `cocoa` from `nutty`, retaining it only in `sweet_aromatic`.
This executes the already documented v3 decision and removes the overlap that
could otherwise make a slot-based shuffle fail to preserve effective family
sizes. This is an ontology change, so regenerate the processed dataset and do
not compare numerical v3 and v4 headline metrics as if they used the same
task.

The historical v3 scripts and reports are intentionally untouched: several
are owned by a read-only imported user in this workspace. The v4 runners are
new files rather than silent edits to those historical artefacts.

## New controls and their meaning

`scripts/fingerprint_baseline_v4.py` provides exactly one odor control per
run and records it in each metrics JSON under `alignment.odor_control`.

| Control | Flag | What is preserved | What it tests |
| --- | --- | --- | --- |
| Real ontology | none | Everything | Main v4 task |
| Ontology permutation | `--shuffle-ontology SEED` | Molecule-to-raw-descriptor evidence; descriptor inventory; family slot counts | Whether the curated grouping scores unusually well versus arbitrary descriptor groupings; **not chance** |
| Chemistry null | `--shuffle-odor-train-labels SEED` | Full training target rows, prevalence, unknown masks, and label co-occurrence | Whether chemical features learn the real odor-label association |

For the chemistry null, only **training** target rows are permuted. Validation
and test targets remain real, and validation still selects the thresholds.
This avoids changing the held-out task while breaking the train-time
structure-to-label relationship. A fold-specific stream uses
`control_seed + test_fold`; that derivation is deterministic and recorded by
the top-level control runner configuration.

## Required run sequence

1. Regenerate `data/processed/sensory/` with the current code. Confirm its
   `schema_version` is `sensory-v4`; the v4 runner rejects v3 input.
2. Smoke-test one seed/fold first:

   ```bash
   PYTHONPATH=. python scripts/run_ontology_controls.py --folds 0 --seeds 0
   ```

3. Run the planned validation ensemble. The default is 20 seeds for iteration;
   use 50 seeds for the final reported control:

   ```bash
   PYTHONPATH=. python scripts/run_ontology_controls.py --seeds 0:49
   ```

   The run writes independent metrics to `outputs/v4_ontology_controls/` and
   per-run summaries plus `control_summary.{json,md}` to
   `reports/v4_ontology_controls/`. Do not overwrite v3 directories.

4. Inspect both `control_summary.md` and per-label supports before making a
   claim. The resulting files name all input summaries and seed scores.

## Predeclared interpretation gate

Treat the v4 grouping as supported by this check only when both conditions
hold:

1. Real ontology odor macro-F1 is at or above the 95th percentile of the
   ontology-permutation distribution (equivalently, its empirical upper-tail
   probability for `permuted >= real` is at most 0.05).
2. The real score also exceeds the chemistry-null distribution.

If either condition fails, retain any evidence that molecules predict raw odor
descriptors, but describe the 12 families only as **heuristic descriptor
bins**, not as an experimentally validated ontology. Then inspect family
support and reconsider semantically heterogeneous labels such as
`aldehydic` before any v5 change.

## Verification notes

Static compilation of the new scripts passed in this workspace. The local
terminal lacks the project's runtime `pandas` dependency, so no v4 train or
test run was performed here. Run the smoke command above in the established
Colab/project environment before launching the 50-seed job.

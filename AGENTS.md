# AGENTS.md

## Project

This repository benchmarks out-of-distribution cell-type annotation for
single-cell data. It compares PCA, scVI, and frozen foundation-model
embeddings under donor/population and laboratory/study shifts.

The codebase is an early Python prototype. Preserve scientific validity and
reproducibility over convenience.

## Repository layout

- `README.md`: benchmark motivation, experimental design, and roadmap.
- `src/main.py`: current end-to-end entry point.
- `src/manifest.py`: extracts cell metadata from an `.h5ad` file.
- `src/train_test_dev_split_data.py`: creates donor-grouped splits and loads
  AnnData views for each split.
- `data/`: local datasets, generated manifests, and split files. Treat large
  data as local artifacts unless explicitly asked to version them.

## Environment and commands

No package or test configuration is committed yet. The current code requires
Python plus `anndata`, `numpy`, and `pandas`.

Run the prototype from the repository root:

```bash
python3 src/main.py
```

The entry point currently expects:

```text
data/f89a12c2-7a3b-415b-ab87-bbc550fe17f4.h5ad
```

Basic validation:

```bash
python3 -m compileall -q src
```

There is currently no automated test suite. When tests are added, prefer
small synthetic AnnData fixtures; do not require full datasets for unit tests.

## Scientific invariants

- Never use query/test labels during preprocessing, representation learning,
  model selection, fitting, or calibration.
- Split by donor, laboratory, study, or dataset as required by the experiment;
  never randomly split individual cells across domains.
- Prevent donor overlap across train, dev, and test. Keep multi-site donor
  handling explicit.
- Fit HVG selection, PCA, scalers, and supervised heads on reference/train data
  only in the strict inductive regime.
- Keep inductive transfer and transductive unsupervised adaptation separate in
  code, outputs, and reporting.
- Foundation-model weights must stay frozen. Fit only the shared downstream
  annotation heads on labelled reference cells.
- Use the same downstream classifiers and evaluation pipeline for every
  representation.
- Preserve original labels. Make harmonization mappings and exclusions
  explicit; do not silently coerce unseen cell types into closed-set labels.
- Save reproducible split manifests with cell IDs, group assignments,
  experiment ID, and random seed.
- Aggregate statistical comparisons by donor or dataset, not by treating cells
  as independent replicates.
- Use macro-F1 as the primary metric unless the experimental protocol is
  intentionally changed and documented.

## Data handling

- Do not inspect or load large `.h5ad` files unless the task requires it.
- Prefer backed AnnData reads for large files and always close the backing file.
- Keep cell alignment explicit through stable `cell_id` values. Validate
  duplicates, missing cells, and split coverage before slicing expression data.
- Do not commit raw/processed datasets, model checkpoints, embeddings, or logs.
- Avoid materializing full expression matrices when metadata or a view is
  sufficient.

## Code conventions

- Make the smallest task-specific change; avoid unrelated refactors.
- Use `pathlib.Path` for paths and type hints on public functions.
- Pass paths, seeds, sites, and experiment settings as arguments instead of
  adding new hard-coded values.
- Use `numpy.random.default_rng(seed)` for reproducible randomness.
- Raise clear errors for invalid metadata or split conditions; do not silently
  drop cells or donors.
- Keep generated filenames deterministic and include the experiment identity
  and seed where relevant.
- Add dependencies only when necessary, and document them in project metadata
  if packaging configuration is introduced.

## Before finishing

- Run the narrowest relevant validation or test.
- Report the exact command run and any validation that could not run because
  local data or dependencies were unavailable.
- Update `README.md` when commands, required inputs, or experimental behavior
  change.

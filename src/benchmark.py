import argparse
import json
import logging
import shutil
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from classifiers.linear import LogisticRegressionClassifier
from metrics import compute_classification_metrics
from preprocessor.dataset import DataSet
from preprocessor.splits import SPLIT_NAMES, create_split_views
from representations.pca import PCARepresentation

LOGGER = logging.getLogger("sc_ood_benchmark")

DEFAULT_ADATA_FILE = Path("data/f89a12c2-7a3b-415b-ab87-bbc550fe17f4.h5ad")
DEFAULT_SPLIT_FILE = Path(
    "data/splits/"
    "within_institute_donor__"
    "institute-genome-institute-of-singapore__seed-42.csv.gz"
)


def run_pca_benchmark(
    adata_file: str | Path,
    split_file: str | Path,
    cache_dir: str | Path = "data/cache",
    runs_dir: str | Path = "runs",
    batch_size: int | None = 4096,
    incremental: bool = False,
    split_names: Sequence[str] = SPLIT_NAMES,
    log_level: str = "INFO",
) -> Path:
    selected_splits = tuple(dict.fromkeys(split_names))
    if not selected_splits:
        raise ValueError("At least one output split must be selected")
    invalid_splits = set(selected_splits).difference(SPLIT_NAMES)
    if invalid_splits:
        raise ValueError(f"Invalid output splits: {sorted(invalid_splits)}")

    split_file = Path(split_file)
    split_metadata = pd.read_csv(
        split_file,
        usecols=["cell_id", "donor_id", "experiment_id"],
        dtype={"cell_id": str},
        compression="gzip",
    )
    if split_metadata["cell_id"].duplicated().any():
        raise ValueError("Split manifest contains duplicate cell IDs")
    required_metadata = split_metadata[["cell_id", "donor_id", "experiment_id"]]
    if required_metadata.isna().any().any():
        raise ValueError("Split manifest contains missing benchmark metadata")
    experiment_ids = split_metadata["experiment_id"].drop_duplicates()
    if len(experiment_ids) != 1:
        raise ValueError("Split manifest must contain one experiment ID")
    experiment_id = experiment_ids.iat[0]
    donor_by_cell = split_metadata.set_index("cell_id")["donor_id"]
    mode = "incremental" if incremental else "exact"
    cache_file = Path(cache_dir) / f"{experiment_id}__pca-hvg-2000__seurat.csv.gz"
    created_at = datetime.now(UTC)
    timestamp = created_at.strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = Path(runs_dir) / (f"{experiment_id}__pca-{mode}__{timestamp}")
    timings: list[dict[str, str | float]] = []
    metric_rows: list[dict[str, object]] = []
    run_dir.mkdir(parents=True, exist_ok=False)
    _configure_logging(run_dir, log_level)
    LOGGER.info("Starting benchmark")
    LOGGER.info("Run directory: %s", run_dir)
    LOGGER.info("Loading dataset: %s", adata_file)

    with DataSet(filepath=adata_file, label_column="cell_type") as dataset:
        LOGGER.info(
            "Loaded dataset with %d cells and %d genes",
            dataset.data.n_obs,
            dataset.data.n_vars,
        )
        LOGGER.info("Creating train/dev/test views from %s", split_file)
        splits = create_split_views(adata=dataset.data, split_file=split_file)
        LOGGER.info(
            "Split sizes: %s",
            ", ".join(f"{name}={split.n_obs}" for name, split in splits.items()),
        )
        representation = PCARepresentation(
            n_hvgs=2000,
            n_components=20,
            seed=42,
            batch_size=batch_size,
            hvg_cache_file=cache_file,
            incremental=incremental,
        )
        LOGGER.info(
            "Fitting %s PCA on %d train cells",
            mode,
            splits["train"].n_obs,
        )
        started_at = perf_counter()
        representation.fit(splits["train"])
        _record_timing(timings, "fit_pca", "train", started_at)
        representation.save(run_dir / "pca.pkl")
        LOGGER.info("Saved PCA model")
        shutil.copy2(split_file, run_dir / "split_manifest.csv.gz")

        labels = dataset.Y()
        train_dir = run_dir / "train"
        LOGGER.info(
            "Transforming train split with %d cells", splits["train"].n_obs
        )
        started_at = perf_counter()
        representation.transform_to_file(
            splits["train"],
            matrix_file=train_dir / "embeddings.npy",
            cell_ids_file=train_dir / "cell_ids.csv.gz",
        )
        _record_timing(timings, "transform", "train", started_at)

        train_matrix, train_cell_ids = _load_embeddings(train_dir)
        train_labels = labels.reindex(train_cell_ids)
        if train_labels.isna().any():
            raise ValueError("Missing train labels")

        classifier = LogisticRegressionClassifier(
            {"max_iter": 1000, "random_state": 42}
        )
        LOGGER.info("Fitting logistic regression on %d train cells", len(train_labels))
        started_at = perf_counter()
        classifier.fit(train_matrix, train_labels.to_numpy())
        _record_timing(timings, "fit_classifier", "train", started_at)
        classifier.save(run_dir / "logistic_regression.pkl")
        LOGGER.info("Saved logistic regression model")

        LOGGER.info("Representation: %s", representation)
        for name in selected_splits:
            split = splits[name]
            split_dir = run_dir / name
            if name != "train":
                LOGGER.info("Transforming %s split with %d cells", name, split.n_obs)
                started_at = perf_counter()
                representation.transform_to_file(
                    split,
                    matrix_file=split_dir / "embeddings.npy",
                    cell_ids_file=split_dir / "cell_ids.csv.gz",
                )
                _record_timing(timings, "transform", name, started_at)

            matrix, cell_ids = _load_embeddings(split_dir)
            split_labels = labels.reindex(cell_ids)
            if split_labels.isna().any():
                raise ValueError(f"Missing {name} labels")

            LOGGER.info("Evaluating %s split", name)
            started_at = perf_counter()
            predictions = classifier.predict(matrix)
            result = compute_classification_metrics(
                split_labels.to_numpy(), predictions
            )
            _record_timing(timings, "evaluate", name, started_at)
            metric_rows.append(
                {
                    "split": name,
                    "aggregation": "split",
                    "donor_id": "",
                    "n_cells": len(cell_ids),
                    **result,
                }
            )
            donor_ids = donor_by_cell.reindex(cell_ids)
            if donor_ids.isna().any():
                raise ValueError(f"Missing {name} donor IDs")
            for donor_id in donor_ids.drop_duplicates():
                donor_mask = donor_ids.eq(donor_id).to_numpy()
                donor_result = compute_classification_metrics(
                    split_labels.to_numpy()[donor_mask], predictions[donor_mask]
                )
                metric_rows.append(
                    {
                        "split": name,
                        "aggregation": "donor",
                        "donor_id": donor_id,
                        "n_cells": int(donor_mask.sum()),
                        **donor_result,
                    }
                )
            pd.DataFrame(
                {
                    "cell_id": cell_ids,
                    "target": split_labels.to_numpy(),
                    "prediction": predictions,
                }
            ).to_csv(
                split_dir / "predictions.csv.gz", index=False, compression="gzip"
            )
            LOGGER.info(
                "%s metrics: macro_f1=%.4f accuracy=%.4f",
                name,
                result["macro_f1"],
                result["accuracy"],
            )

    metadata = {
        "experiment_id": experiment_id,
        "created_at_utc": created_at.isoformat(),
        "log_level": log_level.upper(),
        "adata_file": str(adata_file),
        "split_file": str(split_file),
        "evaluated_splits": list(selected_splits),
        "representation": {
            "type": "PCARepresentation",
            "mode": mode,
            "n_hvgs": representation.n_hvgs,
            "n_components": representation.n_components,
            "seed": representation.seed,
        },
        "classifier": {
            "type": "LogisticRegressionClassifier",
            "config": classifier.config,
        },
    }
    with (run_dir / "run.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")

    pd.DataFrame(metric_rows).to_csv(run_dir / "metrics.csv", index=False)
    pd.DataFrame(timings).to_csv(run_dir / "timings.csv", index=False)
    LOGGER.info("Benchmark complete; artifacts saved to %s", run_dir)
    return run_dir


def _configure_logging(run_dir: Path, log_level: str) -> None:
    level = getattr(logging, log_level.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    for handler in LOGGER.handlers:
        handler.close()
    LOGGER.handlers.clear()
    LOGGER.setLevel(level)
    LOGGER.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(run_dir / "benchmark.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)
    LOGGER.addHandler(file_handler)


def _record_timing(
    timings: list[dict[str, str | float]],
    operation: str,
    split: str,
    started_at: float,
) -> float:
    seconds = perf_counter() - started_at
    timings.append({"operation": operation, "split": split, "seconds": seconds})
    LOGGER.info("Completed %s for %s in %.2f seconds", operation, split, seconds)
    return seconds


def _load_embeddings(split_dir: Path) -> tuple[np.ndarray, pd.Index]:
    matrix = np.load(split_dir / "embeddings.npy", mmap_mode="r")
    cell_ids = pd.Index(
        pd.read_csv(
            split_dir / "cell_ids.csv.gz",
            dtype={"cell_id": str},
            compression="gzip",
        )["cell_id"]
    )
    if not cell_ids.is_unique:
        raise ValueError(f"Duplicate cell IDs in {split_dir}")
    if matrix.shape[0] != len(cell_ids):
        raise ValueError(f"Embedding rows and cell IDs do not match in {split_dir}")
    return matrix, cell_ids


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the train-only PCA benchmark.")
    parser.add_argument("--adata-file", type=Path, default=DEFAULT_ADATA_FILE)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument(
        "--runs-dir",
        "--embedding-dir",
        dest="runs_dir",
        type=Path,
        default=Path("runs"),
        help="Parent directory for complete benchmark runs (default: runs).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLIT_NAMES,
        default=list(SPLIT_NAMES),
        help="Splits to transform and save (default: train dev test).",
    )
    batching = parser.add_mutually_exclusive_group()
    batching.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Cells per transform/incremental-PCA batch (default: 4096).",
    )
    batching.add_argument(
        "--no-batching",
        action="store_true",
        help="Load each split as one batch.",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Use approximate incremental PCA instead of exact PCA.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING"),
        default="INFO",
        help="Terminal and file logging level (default: INFO).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        run_pca_benchmark(
            adata_file=args.adata_file,
            split_file=args.split_file,
            cache_dir=args.cache_dir,
            runs_dir=args.runs_dir,
            batch_size=None if args.no_batching else args.batch_size,
            incremental=args.incremental,
            split_names=args.splits,
            log_level=args.log_level,
        )
    except Exception:
        LOGGER.exception("Benchmark failed")
        raise


if __name__ == "__main__":
    main()

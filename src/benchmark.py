import argparse
import json
import logging
import shutil
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import pandas as pd

from classifiers import build_classifier
from classifiers.linear import LogisticRegressionClassifier
from dataclass.embedding import CellEmbeddings
from evaluation import RunTimer, SplitEvaluator
from preprocessor.dataset import DataSet
from preprocessor.splits import SPLIT_NAMES, create_split_views
from preprocessor.standard_scaler import StandardScaler
from representations.pca import PCARepresentation

LOGGER = logging.getLogger("sc_ood_benchmark")

DEFAULT_ADATA_FILE = Path("data/f89a12c2-7a3b-415b-ab87-bbc550fe17f4.h5ad")
DEFAULT_SPLIT_FILE = Path(
    "data/splits/"
    "within_institute_donor__"
    "institute-genome-institute-of-singapore__seed-42.csv.gz"
)
DEFAULT_MANIFEST_FILE = Path("data/aida_manifest.csv.gz")


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
    timer = RunTimer()
    evaluator = SplitEvaluator()
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
        timer.record("fit_pca", "train", started_at)
        representation.save(run_dir / "pca.pkl")
        LOGGER.info("Saved PCA model")
        shutil.copy2(split_file, run_dir / "split_manifest.csv.gz")

        labels = dataset.Y()
        train_dir = run_dir / "train"
        LOGGER.info("Transforming train split with %d cells", splits["train"].n_obs)
        started_at = perf_counter()
        representation.transform_to_file(
            splits["train"],
            matrix_file=train_dir / "embeddings.npy",
            cell_ids_file=train_dir / "cell_ids.csv.gz",
        )
        timer.record("transform", "train", started_at)

        train_embeddings = CellEmbeddings.load(train_dir)
        train_matrix = train_embeddings.matrix
        train_labels = labels.reindex(train_embeddings.cell_ids)
        if train_labels.isna().any():
            raise ValueError("Missing train labels")

        scaler = StandardScaler()
        train_matrix = scaler.fit_transform(train_matrix)
        scaler.save(run_dir / "scaler.pkl")
        LOGGER.info("Saved standard scaler")

        classifier = LogisticRegressionClassifier(
            {"max_iter": 1000, "random_state": 42}
        )
        LOGGER.info("Fitting logistic regression on %d train cells", len(train_labels))
        started_at = perf_counter()
        classifier.fit(train_matrix, train_labels.to_numpy())
        timer.record("fit_classifier", "train", started_at)
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
                timer.record("transform", name, started_at)

            embeddings = CellEmbeddings.load(split_dir)
            cell_ids = embeddings.cell_ids
            split_labels = labels.reindex(cell_ids)
            if split_labels.isna().any():
                raise ValueError(f"Missing {name} labels")
            matrix = scaler.transform(embeddings.matrix)

            donor_ids = donor_by_cell.reindex(cell_ids)
            if donor_ids.isna().any():
                raise ValueError(f"Missing {name} donor IDs")

            LOGGER.info("Evaluating %s split", name)
            started_at = perf_counter()
            predictions = classifier.predict(matrix)
            metric_rows.extend(
                evaluator.evaluate(
                    name, cell_ids, split_labels.to_numpy(), predictions, donor_ids
                )
            )
            timer.record("evaluate", name, started_at)

            pd.DataFrame(
                {
                    "cell_id": cell_ids,
                    "target": split_labels.to_numpy(),
                    "prediction": predictions,
                }
            ).to_csv(split_dir / "predictions.csv.gz", index=False, compression="gzip")

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
        "scaler": {"type": "StandardScaler"},
        "classifier": {
            "type": "LogisticRegressionClassifier",
            "config": classifier.config,
        },
    }
    with (run_dir / "run.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")

    pd.DataFrame(metric_rows).to_csv(run_dir / "metrics.csv", index=False)
    timer.to_dataframe().to_csv(run_dir / "timings.csv", index=False)
    LOGGER.info("Benchmark complete; artifacts saved to %s", run_dir)
    return run_dir


def run_model_sweep(
    run_dir: str | Path,
    model_configs: Mapping[str, dict],
    manifest_file: str | Path = DEFAULT_MANIFEST_FILE,
    label_column: str = "cell_type",
    eval_splits: Sequence[str] = ("dev", "test"),
) -> Path:
    """Compare classifiers on embeddings already cached in `run_dir`.

    Reuses the `train`/`dev`/`test` embeddings written by `run_pca_benchmark`
    instead of refitting PCA. Labels are read from the manifest file (not the
    `.h5ad`), keyed by `cell_id`, so no expression data is touched. Each entry
    in `model_configs` may set a `classifier` key (`logistic_regression`,
    `knn`; default `logistic_regression`) with the rest passed through as its
    config.
    """
    run_dir = Path(run_dir)
    if not model_configs:
        raise ValueError("At least one model config must be provided")

    split_manifest = pd.read_csv(
        run_dir / "split_manifest.csv.gz",
        usecols=["cell_id", "donor_id"],
        dtype={"cell_id": str},
        compression="gzip",
    )
    donor_by_cell = split_manifest.set_index("cell_id")["donor_id"]

    manifest = pd.read_csv(
        manifest_file,
        usecols=["cell_id", label_column],
        dtype={"cell_id": str},
        compression="gzip",
    )
    if manifest["cell_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate cell IDs")
    labels_by_cell = manifest.set_index("cell_id")[label_column]

    train_embeddings = CellEmbeddings.load(run_dir / "train")
    train_labels = labels_by_cell.reindex(train_embeddings.cell_ids)
    if train_labels.isna().any():
        raise ValueError("Missing train labels")

    scaler = StandardScaler()
    train_matrix = scaler.fit_transform(train_embeddings.matrix)

    eval_data = {}
    for name in eval_splits:
        embeddings = CellEmbeddings.load(run_dir / name)
        cell_ids = embeddings.cell_ids
        split_labels = labels_by_cell.reindex(cell_ids)
        if split_labels.isna().any():
            raise ValueError(f"Missing {name} labels")
        donor_ids = donor_by_cell.reindex(cell_ids)
        if donor_ids.isna().any():
            raise ValueError(f"Missing {name} donor IDs")
        eval_data[name] = (
            scaler.transform(embeddings.matrix),
            cell_ids,
            split_labels,
            donor_ids,
        )

    metric_rows: list[dict[str, object]] = []
    for model_name, model_config in model_configs.items():
        model_config = dict(model_config)
        classifier_type = model_config.pop("classifier", "logistic_regression")
        LOGGER.info("Fitting %s model %r", classifier_type, model_name)
        classifier = build_classifier(classifier_type, model_config)
        classifier.fit(train_matrix, train_labels.to_numpy())
        evaluator = SplitEvaluator({"model": model_name})

        for name, (matrix, cell_ids, split_labels, donor_ids) in eval_data.items():
            predictions = classifier.predict(matrix)
            metric_rows.extend(
                evaluator.evaluate(
                    name, cell_ids, split_labels.to_numpy(), predictions, donor_ids
                )
            )

    output_file = run_dir / "model_sweep.csv"
    pd.DataFrame(metric_rows).to_csv(output_file, index=False)
    LOGGER.info("Model sweep complete; results saved to %s", output_file)
    return output_file


def _configure_logging(run_dir: Path, log_level: str) -> None:
    level = getattr(logging, log_level.upper(), None)
    if not isinstance(level, int):
        raise TypeError(f"Invalid log level: {log_level}")

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

    subparsers = parser.add_subparsers(dest="command")
    sweep_parser = subparsers.add_parser(
        "sweep",
        help=(
            "Reuse cached embeddings from an existing run directory to "
            "compare classifier configs, without refitting PCA."
        ),
    )
    sweep_parser.add_argument(
        "--run-dir", type=Path, required=True, help="Existing run directory."
    )
    sweep_parser.add_argument(
        "--model-configs",
        type=Path,
        required=True,
        help=(
            "JSON file mapping model name to a config dict. Each dict may "
            "set 'classifier' (logistic_regression, knn; default "
            "logistic_regression); remaining keys are passed to it."
        ),
    )
    sweep_parser.add_argument(
        "--manifest-file", type=Path, default=DEFAULT_MANIFEST_FILE
    )
    sweep_parser.add_argument("--label-column", default="cell_type")
    sweep_parser.add_argument(
        "--splits",
        nargs="+",
        default=["dev", "test"],
        help="Splits to evaluate (default: dev test); train is always the fit split.",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        if getattr(args, "command", None) == "sweep":
            with args.model_configs.open("r", encoding="utf-8") as file:
                model_configs = json.load(file)
            run_model_sweep(
                run_dir=args.run_dir,
                model_configs=model_configs,
                manifest_file=args.manifest_file,
                label_column=args.label_column,
                eval_splits=args.splits,
            )
        else:
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

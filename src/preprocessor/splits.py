from __future__ import annotations

import argparse
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("cell_id", "donor_id", "institute")
SPLIT_NAMES = ("train", "dev", "test")


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def assign_donor_splits(
    manifest: pd.DataFrame,
    seed: int = 42,
    institute: str = "Genome Institute of Singapore",
) -> pd.DataFrame:
    missing_columns = set(REQUIRED_COLUMNS).difference(manifest.columns)
    if missing_columns:
        raise ValueError(
            "Manifest is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    manifest = manifest.loc[:, REQUIRED_COLUMNS].copy()
    if manifest["cell_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate cell IDs")
    if manifest.loc[:, REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Manifest contains missing cell, donor, or site values")

    institutes_per_donor = manifest.groupby("donor_id")["institute"].nunique()
    multi_institute_donors = institutes_per_donor[institutes_per_donor > 1].index
    eligible = manifest["institute"].eq(institute) & ~manifest["donor_id"].isin(
        multi_institute_donors
    )

    donors = manifest.loc[eligible, "donor_id"].drop_duplicates().to_numpy(copy=True)
    if len(donors) < 3:
        raise ValueError(
            f"Institute {institute!r} has fewer than three eligible donors"
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(donors)

    n_test = max(1, round(len(donors) * 0.15))
    n_dev = max(1, round(len(donors) * 0.15))
    if n_test + n_dev >= len(donors):
        raise ValueError("Not enough donors to create non-empty train/dev/test splits")

    assignments = {
        **dict.fromkeys(donors[n_test + n_dev :], "train"),
        **dict.fromkeys(donors[n_test : n_test + n_dev], "dev"),
        **dict.fromkeys(donors[:n_test], "test"),
    }
    experiment_id = (
        f"within_institute_donor__institute-{_slugify(institute)}__seed-{seed}"
    )

    split_manifest = manifest[["cell_id", "donor_id"]].copy()
    split_manifest["split"] = "excluded"
    split_manifest.loc[eligible, "split"] = manifest.loc[
        eligible, "donor_id"
    ].map(assignments)
    split_manifest["experiment_id"] = experiment_id
    split_manifest["institute"] = institute
    split_manifest["split_seed"] = seed

    used = split_manifest[split_manifest["split"] != "excluded"]
    if used.groupby("donor_id")["split"].nunique().max() != 1:
        raise ValueError("A donor was assigned to more than one split")
    if set(used["split"]) != set(SPLIT_NAMES):
        raise ValueError("Train, dev, and test splits must all be non-empty")

    return split_manifest


def create_train_test_dev_split(
    manifest_file: str | Path,
    seed: int = 42,
    institute: str = "Genome Institute of Singapore",
    output_dir: str | Path = "data/splits",
) -> Path:
    manifest = pd.read_csv(
        manifest_file,
        usecols=list(REQUIRED_COLUMNS),
        compression="gzip",
    )
    split_manifest = assign_donor_splits(
        manifest, seed=seed, institute=institute
    )

    experiment_id = split_manifest["experiment_id"].iat[0]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{experiment_id}.csv.gz"
    split_manifest.to_csv(output_file, index=False, compression="gzip")

    used = split_manifest[split_manifest["split"] != "excluded"]
    print(f"Created {output_file}")
    print(
        used.groupby("split", observed=True).agg(
            cells=("cell_id", "size"), donors=("donor_id", "nunique")
        )
    )
    return output_file


def load_data_splits(
    adata_file: str | Path, split_file: str | Path
) -> tuple[ad.AnnData, dict[str, ad.AnnData]]:
    adata = ad.read_h5ad(adata_file, backed="r")
    split_manifest = pd.read_csv(
        split_file,
        usecols=["cell_id", "split"],
        compression="gzip",
    )

    if split_manifest["cell_id"].duplicated().any():
        adata.file.close()
        raise ValueError("Split manifest contains duplicate cell IDs")

    invalid_splits = set(split_manifest["split"].dropna()).difference(
        (*SPLIT_NAMES, "excluded")
    )
    if invalid_splits:
        adata.file.close()
        raise ValueError(f"Split manifest contains invalid splits: {invalid_splits}")

    aligned_split = split_manifest.set_index("cell_id")["split"].reindex(
        adata.obs_names
    )
    if aligned_split.isna().any():
        adata.file.close()
        raise ValueError("Split manifest does not contain every AnnData cell")

    splits = {
        split: adata[aligned_split.eq(split).to_numpy(), :]
        for split in SPLIT_NAMES
    }
    if any(split.n_obs == 0 for split in splits.values()):
        adata.file.close()
        raise ValueError("Train, dev, and test AnnData views must all be non-empty")
    return adata, splits


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a reproducible donor-grouped train/dev/test split."
    )
    parser.add_argument("manifest_file", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--institute",
        default="Genome Institute of Singapore",
        help="Value from the manifest's institute column.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    args = parser.parse_args()

    create_train_test_dev_split(
        manifest_file=args.manifest_file,
        seed=args.seed,
        institute=args.institute,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

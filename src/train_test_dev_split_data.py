import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def create_train_test_dev_split(
    manifest_file: str | Path,
    seed: int = 42,
    site: str = "Genome Institute of Singapore",
    output_dir: str | Path = "data/splits",
) -> Path:
    manifest = pd.read_csv(
        manifest_file,
        usecols=["cell_id", "donor_id", "laboratory_or_site"],
    )

    sites_per_donor = manifest.groupby("donor_id")["laboratory_or_site"].nunique()
    multisite_donors = sites_per_donor[sites_per_donor > 1].index

    eligible = manifest["laboratory_or_site"].eq(site) & ~manifest["donor_id"].isin(
        multisite_donors
    )

    donors = (
        manifest.loc[eligible, "donor_id"]
        .drop_duplicates()
        .to_numpy(dtype=str, copy=True)
    )

    if len(donors) < 3:
        raise ValueError(f"Site {site!r} has fewer than three eligible donors")

    rng = np.random.default_rng(seed)
    rng.shuffle(donors)

    n_test = max(1, round(len(donors) * 0.15))
    n_dev = max(1, round(len(donors) * 0.15))

    if n_test + n_dev >= len(donors):
        raise ValueError("Not enough donors to create non-empty train/dev/test splits")

    test_donors = donors[:n_test]
    dev_donors = donors[n_test : n_test + n_dev]
    train_donors = donors[n_test + n_dev :]

    assignments = {
        **dict.fromkeys(train_donors, "train"),
        **dict.fromkeys(dev_donors, "dev"),
        **dict.fromkeys(test_donors, "test"),
    }

    experiment_id = (
        f"within_site_donor__site-{_slugify(site)}__seed-{seed}"
    )
    
    split_manifest = manifest[["cell_id", "donor_id"]].copy()
    split_manifest["split"] = "excluded"
    split_manifest.loc[eligible, "split"] = manifest.loc[
        eligible, "donor_id"
    ].map(assignments)
    split_manifest["experiment_id"] = experiment_id
    split_manifest["site"] = site
    split_manifest["split_seed"] = seed

    used = split_manifest[split_manifest["split"] != "excluded"]
    assert used.groupby("donor_id")["split"].nunique().max() == 1

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{experiment_id}.csv.gz"
    split_manifest.to_csv(output_file, index=False, compression="gzip")

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
    )

    if split_manifest["cell_id"].duplicated().any():
        adata.file.close()
        raise ValueError("Split manifest contains duplicate cell IDs")

    aligned_split = split_manifest.set_index("cell_id")["split"].reindex(
        adata.obs_names
    )
    if aligned_split.isna().any():
        adata.file.close()
        raise ValueError("Split manifest does not contain every AnnData cell")

    splits = {
        split: adata[aligned_split.eq(split).to_numpy(), :]
        for split in ("train", "dev", "test")
    }
    return adata, splits

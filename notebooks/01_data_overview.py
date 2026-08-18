"""Inspect basic AIDA metadata"""

# Import Path so the dataset location works from any current directory.
from pathlib import Path

# Import AnnData to open the single-cell dataset in backed/read-only mode.
import anndata as ad

# Import pandas to build the small overview table.
import pandas as pd


# Find the repository root from the location of this Python file.
REPO_ROOT = Path(__file__).resolve().parents[1]

# Build the absolute path to the local AIDA dataset.
AIDA_DATASET_PATH = REPO_ROOT / "data/f89a12c2-7a3b-415b-ab87-bbc550fe17f4.h5ad"

# Select one donor whose sample IDs will be displayed as an example.
TEST_DONOR_ID = "IN_NIB_H031"

# List only the metadata columns needed for this first exploration.
METADATA_COLUMNS = [
    "donor_id",  # Identifies the biological donor.
    "sample_id",  # Identifies the biological sample.
    "library_id",  # Identifies the sequencing library or multiplexed pool.
    "institute",  # Identifies the laboratory or study site.
    "Country",  # Records the donor or study country.
    "self_reported_ethnicity",  # Records the reported population metadata.
    "assay",  # Records the experimental protocol or chemistry.
    "tissue",  # Records the sampled tissue.
    "disease",  # Records the disease state.
    "author_cell_type",  # Preserves the authors' original cell label.
    "Annotation_Level1",  # Contains the coarse AIDA cell-type label.
    "Annotation_Level4",  # Contains the fine AIDA cell-type label.
    "cell_type",  # Contains the harmonized cell-type label.
    "cell_type_ontology_term_id",  # Links the label to the Cell Ontology.
]

# Open the large expression file without loading its matrix into memory.
adata = ad.read_h5ad(AIDA_DATASET_PATH, backed="r")

# Copy only cell metadata so it remains available after closing the backing file.
manifest = adata.obs[METADATA_COLUMNS].copy()

# Insert the AnnData row names as the stable, unique cell identifier.
manifest.insert(0, "cell_id", adata.obs_names.astype(str))

# Close the read-only backing file as soon as its metadata has been copied.
adata.file.close()

# Display the object type to confirm that the metadata is a pandas DataFrame.
print(type(manifest))

# Display the first cell and its metadata as a concrete example.
print(manifest.iloc[0], "\n")

# Keep only rows belonging to the selected donor and select sample_id.
donor_samples = manifest.loc[
    manifest["donor_id"].eq(TEST_DONOR_ID),
    "sample_id",
]

# Remove missing sample identifiers from the selected donor.
donor_samples = donor_samples.dropna()

# Keep one occurrence of each sample identifier.
donor_samples = donor_samples.drop_duplicates()

# Sort the identifiers so the output is deterministic and easy to read.
donor_samples = donor_samples.sort_values()

# Convert categorical identifiers to plain strings for clean printing.
donor_samples = donor_samples.astype(str)

# Replace the inherited cell-row index with a simple zero-based index.
donor_samples = donor_samples.reset_index(drop=True)

# Display every unique sample associated with the selected donor.
print(f"Unique samples for donor {TEST_DONOR_ID}:")

# Print the complete Series instead of showing only its first rows.
print(donor_samples.to_string(index=False), "\n")

# Build a table describing how many unique entities exist at each data level.
overview = pd.DataFrame(
    {
        "level": [
            "datasets",  # This script currently reads one AIDA dataset.
            "sites",  # Institutes represent laboratory or study sites.
            "donors",  # Donors are biological grouping units.
            "samples",  # Samples are nested within donors in this dataset.
            "libraries",  # Libraries may contain multiplexed samples.
            "cells",  # Each manifest row represents one cell.
        ],
        "count": [
            1,  # Only one .h5ad file is open in this overview.
            manifest["institute"].nunique(),  # Count distinct sites.
            manifest["donor_id"].nunique(),  # Count distinct donors.
            manifest["sample_id"].nunique(),  # Count distinct samples.
            manifest["library_id"].nunique(),  # Count distinct libraries.
            manifest["cell_id"].nunique(),  # Count distinct cells.
        ],
    }
)

# Display the final hierarchy-count table.
print(overview.to_string(index=False))

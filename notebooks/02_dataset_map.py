"""Create a documented visual map of the cell-level manifest."""

# Allow modern type hints while remaining compatible with older Python versions.
from __future__ import annotations

# Import argparse so input and output paths can be changed from the command line.
import argparse

# Import escape so metadata values are safe when inserted into the HTML report.
from html import escape

# Import Path for operating-system-independent file paths.
from pathlib import Path

# Import pandas for reading, grouping, validating, and exporting metadata.
import pandas as pd

# Find the repository root from this script's location.
REPO_ROOT = Path(__file__).resolve().parents[1]

# Use the normalized metadata manifest as the default input.
DEFAULT_MANIFEST = REPO_ROOT / "data/aida_manifest.csv.gz"

# Store generated local reports outside both notebooks and source code.
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/data_map"

# Read only the metadata fields needed to map the dataset.
MANIFEST_COLUMNS = [
    "dataset_or_study",  # Names the source dataset or study.
    "laboratory_or_site",  # Names the laboratory or collection site.
    "donor_id",  # Identifies each biological donor.
    "sample_id",  # Identifies each biological sample.
    "library_id",  # Identifies each sequencing library or multiplexed pool.
    "protocol_or_chemistry",  # Records the assay technology.
    "tissue",  # Records the sampled tissue.
    "cell_type_coarse",  # Provides broad cell-type labels.
]


def _joined_unique(values: pd.Series) -> str:
    """Return sorted unique metadata values as one readable string."""

    # Remove missing values, convert values to text, and remove duplicates.
    unique_values = {str(value) for value in values.dropna()}

    # Sort the values and join them with a visible separator.
    return " | ".join(sorted(unique_values))


def load_manifest(manifest_file: str | Path) -> pd.DataFrame:
    """Load only the manifest columns required by this exploration."""

    # Convert string paths to Path objects so path handling is consistent.
    manifest_file = Path(manifest_file)

    # Stop with a clear message when the expected manifest does not exist.
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_file}")

    # Read only the header first so required columns can be checked cheaply.
    available_columns = pd.read_csv(manifest_file, nrows=0).columns

    # Calculate which required columns are absent from the CSV header.
    missing_columns = sorted(set(MANIFEST_COLUMNS) - set(available_columns))

    # Stop instead of silently producing an incomplete report.
    if missing_columns:
        raise ValueError(
            "Manifest is missing required columns: " + ", ".join(missing_columns)
        )

    # Load only relevant columns and store repeated strings as categories.
    manifest = pd.read_csv(
        manifest_file,
        usecols=MANIFEST_COLUMNS,
        dtype={column: "category" for column in MANIFEST_COLUMNS},
    )

    # Reject an empty manifest because no map could be produced from it.
    if manifest.empty:
        raise ValueError("Manifest contains no cells")

    # Return the validated cell-level metadata table.
    return manifest


def create_tables(manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Aggregate the cell-level manifest into understandable data levels."""

    # Count each level in the dataset hierarchy.
    overview = pd.DataFrame(
        {
            "level": [
                "datasets",  # Count source datasets.
                "sites",  # Count laboratories or collection sites.
                "donors",  # Count biological donors.
                "samples",  # Count biological samples.
                "libraries",  # Count sequencing libraries.
                "cells",  # Count cell-level manifest rows.
            ],
            "count": [
                manifest["dataset_or_study"].nunique(),  # Unique datasets.
                manifest["laboratory_or_site"].nunique(),  # Unique sites.
                manifest["donor_id"].nunique(),  # Unique donors.
                manifest["sample_id"].nunique(),  # Unique samples.
                manifest["library_id"].nunique(),  # Unique libraries.
                len(manifest),  # Total cells represented by rows.
            ],
        }
    )

    # Define the ordered columns that describe the nesting structure.
    hierarchy_columns = [
        "dataset_or_study",  # Highest data level.
        "laboratory_or_site",  # Domain or site level.
        "donor_id",  # Biological grouping level.
        "sample_id",  # Sample level.
        "library_id",  # Sequencing-library level.
    ]

    # Count cells for every observed dataset-site-donor-sample-library path.
    hierarchy = (
        manifest.groupby(
            hierarchy_columns,
            dropna=False,  # Preserve paths containing missing metadata.
            observed=True,  # Avoid unused combinations of categorical values.
        )
        .size()  # Count cell rows inside each path.
        .rename("number_of_cells")  # Give the count a meaningful name.
        .reset_index()  # Convert grouped identifiers back into columns.
        .sort_values(hierarchy_columns)  # Make the exported map deterministic.
    )

    # Summarize the number of cells and related entities for every donor.
    donor_summary = (
        manifest.groupby("donor_id", dropna=False, observed=True)
        .agg(
            number_of_cells=("donor_id", "size"),  # Cells from this donor.
            number_of_samples=("sample_id", "nunique"),  # Donor samples.
            number_of_libraries=("library_id", "nunique"),  # Donor libraries.
            number_of_sites=("laboratory_or_site", "nunique"),  # Donor sites.
            number_of_coarse_cell_types=("cell_type_coarse", "nunique"),
            sample_ids=("sample_id", _joined_unique),  # All donor samples.
            laboratory_or_sites=("laboratory_or_site", _joined_unique),
        )
        .reset_index()  # Restore donor_id as a normal column.
        .sort_values("number_of_cells", ascending=False)  # Largest donors first.
    )

    # Summarize cells, donors, samples, and libraries for every site.
    site_summary = (
        manifest.groupby("laboratory_or_site", dropna=False, observed=True)
        .agg(
            number_of_cells=("laboratory_or_site", "size"),  # Site cells.
            number_of_donors=("donor_id", "nunique"),  # Site donors.
            number_of_samples=("sample_id", "nunique"),  # Site samples.
            number_of_libraries=("library_id", "nunique"),  # Site libraries.
        )
        .reset_index()  # Restore the site identifier as a normal column.
        .sort_values("number_of_cells", ascending=False)  # Largest sites first.
    )

    # Count missing entries and calculate their percentage for every column.
    missing_values = pd.DataFrame(
        {
            "column": manifest.columns,  # Metadata field being checked.
            "missing_values": manifest.isna().sum().to_numpy(),  # Missing count.
            "missing_percent": manifest.isna().mean().mul(100).to_numpy(),
        }
    ).sort_values("missing_percent", ascending=False)

    # Count every coarse cell type within each laboratory or site.
    cell_type_by_site = pd.crosstab(
        manifest["laboratory_or_site"],  # Create one row per site.
        manifest["cell_type_coarse"],  # Create one column per cell type.
        dropna=False,  # Preserve missing categories when pandas supports them.
    )

    # Convert site cell-type counts into within-site percentages.
    cell_type_percent_by_site = cell_type_by_site.div(
        cell_type_by_site.sum(axis=1),  # Calculate the total cells per site.
        axis=0,  # Divide each row by its own site total.
    ).mul(100)

    # Count the number of donors associated with each sample identifier.
    sample_donor_counts = (
        manifest.groupby("sample_id", observed=True)["donor_id"]
        .nunique()  # Count distinct donors for every sample.
        .rename("number_of_donors")  # Name the validation result.
    )

    # Keep sample identifiers that unexpectedly appear under several donors.
    sample_donor_conflicts = sample_donor_counts[
        sample_donor_counts > 1
    ].reset_index()

    # Summarize how many samples, donors, and sites occur in each library.
    library_summary = (
        manifest.groupby("library_id", dropna=False, observed=True)
        .agg(
            number_of_samples=("sample_id", "nunique"),  # Multiplexed samples.
            number_of_donors=("donor_id", "nunique"),  # Multiplexed donors.
            number_of_sites=("laboratory_or_site", "nunique"),  # Sites/library.
        )
        .reset_index()  # Restore library_id as a normal column.
        .sort_values("number_of_samples", ascending=False)  # Largest pools first.
    )

    # Select donors observed at more than one laboratory or site.
    multi_site_donors = donor_summary.loc[
        donor_summary["number_of_sites"] > 1,
        ["donor_id", "number_of_sites", "laboratory_or_sites"],
    ]

    # Give every aggregate table a stable name for export and reporting.
    return {
        "overview": overview,
        "dataset_hierarchy": hierarchy,
        "donor_summary": donor_summary,
        "site_summary": site_summary,
        "missing_values": missing_values,
        "coarse_cell_type_by_site": cell_type_by_site.reset_index(),
        "coarse_cell_type_percent_by_site": cell_type_percent_by_site.reset_index(),
        "sample_donor_conflicts": sample_donor_conflicts,
        "library_summary": library_summary,
        "multi_site_donors": multi_site_donors,
    }


def save_tables(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Save every complete aggregate table as a CSV file."""

    # Remove a misleading output name produced by an earlier script version.
    (output_dir / "library_sample_conflicts.csv").unlink(missing_ok=True)

    # Visit each named aggregate table.
    for name, table in tables.items():
        # Save the table without pandas' artificial row-number index.
        table.to_csv(output_dir / f"{name}.csv", index=False)


def _visual_table(
    table: pd.DataFrame,
    value_columns: list[str],
    fixed_maximum: float | None = None,
) -> str:
    """Convert numeric columns into labelled HTML progress bars."""

    # Copy the table so HTML formatting never changes the scientific values.
    visual_table = table.copy()

    # Escape every non-numeric label before allowing progress-bar HTML.
    for column in visual_table.columns.difference(value_columns):
        visual_table[column] = (
            visual_table[column]
            .astype("string")
            .fillna("")
            .map(lambda value: escape(str(value)))
        )

    # Add a progress bar to each selected numeric column.
    for column in value_columns:
        # Use a fixed scale for percentages or the column maximum for counts.
        maximum = fixed_maximum or max(float(table[column].max()), 1)

        # Format each value as readable text followed by its visual bar.
        visual_table[column] = table[column].map(
            lambda value: (
                f"{float(value):,.1f}"
                f'<progress value="{float(value)}" max="{maximum}"></progress>'
            )
        )

    # Return an HTML table while preserving the progress elements as markup.
    return visual_table.to_html(index=False, border=0, escape=False)


def save_html_report(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Create one self-contained visual HTML report from aggregate tables."""

    # Render the six hierarchy totals as summary cards.
    overview_cards = "".join(
        '<div class="card">'
        f'<span>{escape(str(row.level))}</span>'
        f"<strong>{int(row.count):,}</strong>"
        "</div>"
        for row in tables["overview"].itertuples(index=False)
    )

    # Display all sites and scale each count column independently.
    site_html = _visual_table(
        tables["site_summary"],
        [
            "number_of_cells",
            "number_of_donors",
            "number_of_samples",
            "number_of_libraries",
        ],
    )

    # Display cell-type percentages with a common zero-to-100 scale.
    cell_type_table = tables["coarse_cell_type_percent_by_site"]

    # Identify every percentage column while excluding the site label.
    cell_type_columns = [
        column
        for column in cell_type_table.columns
        if column != "laboratory_or_site"
    ]

    # Add percentage bars to the site-by-cell-type table.
    cell_type_html = _visual_table(
        cell_type_table,
        cell_type_columns,
        fixed_maximum=100,
    )

    # Select the most cell-rich donors for a readable on-screen table.
    top_donors = tables["donor_summary"].head(25)[
        [
            "donor_id",
            "number_of_cells",
            "number_of_samples",
            "number_of_libraries",
            "number_of_coarse_cell_types",
        ]
    ]

    # Add bars to all numeric fields in the top-donor table.
    donor_html = _visual_table(
        top_donors,
        [
            "number_of_cells",
            "number_of_samples",
            "number_of_libraries",
            "number_of_coarse_cell_types",
        ],
    )

    # Build an audit table for relationships relevant to future data splits.
    relationship_audit = pd.DataFrame(
        {
            "relationship": [
                "Sample IDs assigned to multiple donors",
                "Multiplexed libraries containing multiple samples",
                "Donors associated with multiple sites",
            ],
            "count": [
                len(tables["sample_donor_conflicts"]),
                int((tables["library_summary"]["number_of_samples"] > 1).sum()),
                len(tables["multi_site_donors"]),
            ],
        }
    )

    # Convert the relationship audit to a standard HTML table.
    audit_html = relationship_audit.to_html(index=False, border=0)

    # Copy missing-value results so percentages can be formatted for display.
    missing_values = tables["missing_values"].copy()

    # Format missing percentages to two decimal places.
    missing_values["missing_percent"] = missing_values["missing_percent"].map(
        lambda value: f"{value:.2f}%"
    )

    # Convert missing-value results to a standard HTML table.
    missing_html = missing_values.to_html(index=False, border=0)

    # Define presentation rules embedded directly in the generated report.
    style = """
    body { font-family: system-ui, sans-serif; margin: 0 auto; max-width: 1400px;
           padding: 32px; color: #172033; background: #f8fafc; }
    h1, h2 { color: #0f172a; } h2 { margin-top: 38px; }
    .note { color: #475569; }
    .hierarchy { display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; }
    .card { padding: 18px 12px; border-radius: 10px; background: white;
            box-shadow: 0 1px 4px #cbd5e1; text-align: center; }
    .card span { display: block; color: #64748b; text-transform: capitalize; }
    .card strong { display: block; font-size: 1.45rem; margin-top: 5px; }
    .table-wrap { overflow-x: auto; background: white; padding: 12px;
                  border-radius: 10px; box-shadow: 0 1px 4px #cbd5e1; }
    table { width: 100%; border-collapse: collapse; }
    th, td { border-bottom: 1px solid #e2e8f0; padding: 8px; text-align: left; }
    th { background: #e2e8f0; }
    progress { display: block; width: 130px; height: 12px; accent-color: #2563eb; }
    @media (max-width: 850px) { .hierarchy { grid-template-columns: 1fr 1fr; } }
    """

    # Assemble the report sections in their displayed order.
    report_parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Dataset metadata map</title><style>",
        style,
        "</style></head><body>",
        "<h1>Dataset metadata map</h1>",
        '<p class="note">Dataset → site → donor → sample → library → cells. '
        "Only metadata is used.</p>",
        f'<div class="hierarchy">{overview_cards}</div>',
        '<h2>Sites</h2><div class="table-wrap">',
        site_html,
        "</div>",
        '<h2>Coarse cell types by site (%)</h2><div class="table-wrap">',
        cell_type_html,
        "</div>",
        '<h2>Largest 25 donors</h2><div class="table-wrap">',
        donor_html,
        "</div>",
        "<h2>Relationship audit</h2>",
        '<p class="note">Multiplexed libraries and multi-site donors may be valid. '
        "They are reported so future split decisions remain explicit.</p>",
        f'<div class="table-wrap">{audit_html}</div>',
        f'<h2>Missing metadata</h2><div class="table-wrap">{missing_html}</div>',
        "</body></html>",
    ]

    # Join all HTML fragments into one complete document.
    report = "".join(report_parts)

    # Save the visual report with explicit UTF-8 encoding.
    (output_dir / "dataset_map.html").write_text(report, encoding="utf-8")


def create_data_map(
    manifest_file: str | Path = DEFAULT_MANIFEST,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> None:
    """Load metadata and create all mapping outputs."""

    # Convert the output argument into a Path object.
    output_dir = Path(output_dir)

    # Create the output directory and any missing parent directories.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and validate the normalized metadata manifest.
    manifest = load_manifest(manifest_file)

    # Calculate every aggregate map and relationship audit.
    tables = create_tables(manifest)

    # Save complete tables for detailed inspection.
    save_tables(tables, output_dir)

    # Save the concise visual map for browser viewing.
    save_html_report(tables, output_dir)

    # Print hierarchy counts so command-line execution gives immediate feedback.
    print(tables["overview"].to_string(index=False))

    # Print the location of the generated outputs.
    print(f"\nSaved dataset map to {output_dir}")

    # Print the relationships that matter when designing donor-aware splits.
    print(
        "Relationship audit: "
        f"{len(tables['sample_donor_conflicts'])} samples across donors, "
        f"{int((tables['library_summary']['number_of_samples'] > 1).sum())} "
        "multiplexed libraries, "
        f"{len(tables['multi_site_donors'])} donors across sites"
    )


def parse_args() -> argparse.Namespace:
    """Read optional command-line paths."""

    # Create the command-line parser and its help description.
    parser = argparse.ArgumentParser(
        description="Create aggregate tables and an HTML cell-manifest map."
    )

    # Allow callers to replace the default manifest path.
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)

    # Allow callers to replace the default report directory.
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    # Parse and return the command-line values.
    return parser.parse_args()


# Run the report only when this file is executed directly.
if __name__ == "__main__":
    # Read optional manifest and output-directory arguments.
    args = parse_args()

    # Generate the complete metadata map using those arguments.
    create_data_map(args.manifest, args.output_dir)

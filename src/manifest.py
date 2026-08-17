from pathlib import Path

import anndata as ad


def create_manifest(filepath: str | Path, manifest_file: str | Path = "data/aida_manifest.csv.gz"):

    filepath = Path(filepath)
    manifest_file = Path(manifest_file)

    adata = ad.read_h5ad(filepath, backed='r')

    manifest = adata.obs[
            [
                "donor_id",
                "sample_id",
                "library_id",
                "institute",
                "Country",
                "self_reported_ethnicity",
                "assay",
                "tissue",
                "disease",
                "author_cell_type",
                "Annotation_Level1",
                "Annotation_Level4",
                "cell_type",
                "cell_type_ontology_term_id",
            ]
        ].copy()

    manifest.rename(
            columns={
                "institute": "laboratory_or_site",
                "Country": "country",
                "self_reported_ethnicity": "population",
                "assay": "protocol_or_chemistry",
                "disease": "disease_state",
                "author_cell_type": "cell_type_original",
                "Annotation_Level1": "cell_type_coarse",
                "Annotation_Level4": "cell_type_fine",
                "cell_type": "cell_type_harmonized",
            },
            inplace=True,
        )
    
    manifest.insert(0, "cell_id", adata.obs_names)
    manifest.insert(1, "dataset_or_study", "AIDA_Phase_1_Data_Freeze_v2")
    manifest.to_csv(manifest_file, index=False, compression="gzip")

    adata.file.close()
    print(f"Created {manifest_file} with {len(manifest):,} cells")

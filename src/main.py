from pathlib import Path

from preprocessor.dataset import DataSet
from preprocessor.manifest import Manifest

AIDA_FILE = "data/f89a12c2-7a3b-415b-ab87-bbc550fe17f4.h5ad"
MANIFEST_FILE = Path("data/aida_manifest.csv.gz")


"""
    cell_id = manifest.iloc[0]["cell_id"]
    expression_row = adata.obs_names.get_loc(cell_id)

    expression = adata.X[expression_row]
    metadata = manifest.loc[manifest["cell_id"] == cell_id]
"""

def main() -> None:
    with DataSet(AIDA_FILE) as dataset:
        manifest = Manifest(dataset=dataset)
        manifest.init()
        manifest.save(MANIFEST_FILE)

    print(f"Created {MANIFEST_FILE}")

if __name__ == "__main__":
    main()

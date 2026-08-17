from pathlib import Path

from manifest import create_manifest
from train_test_dev_split_data import create_train_test_dev_split, load_data_splits

AIDA_FILE = "data/f89a12c2-7a3b-415b-ab87-bbc550fe17f4.h5ad"
MANIFEST_FILE = Path("data/aida_manifest.csv.gz")


"""
    cell_id = manifest.iloc[0]["cell_id"]
    expression_row = adata.obs_names.get_loc(cell_id)

    expression = adata.X[expression_row]
    metadata = manifest.loc[manifest["cell_id"] == cell_id]
"""

def main():

    if not MANIFEST_FILE.exists():
        create_manifest(AIDA_FILE, MANIFEST_FILE)

    split_file = create_train_test_dev_split(MANIFEST_FILE)
    adata, splits = load_data_splits(AIDA_FILE, split_file)
    try:
        for split, split_adata in splits.items():
            print(split, split_adata.shape)
    finally:
        adata.file.close()


if __name__ == "__main__":
    main()

from pathlib import Path

from preprocessor.dataset import DataSet
from preprocessor.splits import create_split_views
from representations.pca import PCARepresentation


def run_pca_benchmark(
    adata_file: str | Path,
    split_file: str | Path,
) -> None:
    with DataSet(filepath=adata_file, label_column="cell_type") as dataset:
        splits = create_split_views(adata=dataset.data, split_file=split_file)

        representation = PCARepresentation(
            n_hvgs=2000,
            n_components=20,
            seed=42,
        )

        representation.fit(splits["train"])

        embeddings = {
            name: representation.transform(split) for name, split in splits.items()
        }

        # Retrieve labels through DataSet and align using cell IDs.
        train_labels = dataset.Y().reindex(embeddings["train"].cell_ids)

        if train_labels.isna().any():
            raise ValueError("Missing train labels")

        print(representation)
        for name, result in embeddings.items():
            print(name, result.matrix.shape)

        # classifier.fit(
        #     embeddings["train"].matrix,
        #     train_labels.to_numpy(),
        # )


def main() -> None:
    run_pca_benchmark(
        adata_file=("data/f89a12c2-7a3b-415b-ab87-bbc550fe17f4.h5ad"),
        split_file=(
            "data/splits/"
            "within_institute_donor__"
            "institute-genome-institute-of-singapore__seed-42.csv.gz"
        ),
    )


if __name__ == "__main__":
    main()

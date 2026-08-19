from representations.base import Representation


class PCARepresentation(Representation):
    """
        - PCARepresentation:
            - Uses normalized adata.X.
            - Selects HVGs using train only.
            - Fits PCA using train only.
            - Stores HVGs and PCA parameters.
    """


    def __init__(self):
        pass

    def fit(self, train) -> None:
        pass
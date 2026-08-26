import pickle
from pathlib import Path

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from typing_extensions import Self

from classifiers.base import BaseClassifier

CONFIG = {
    "n_neighbors": 5,
    "weights": "uniform",
    "algorithm": "auto",
    "leaf_size": 30,
    "p": 2,
    "metric": "minkowski",
    "metric_params": None,
    "n_jobs": None,
}


class KNNClassifier(BaseClassifier):

    def __init__(self, config: dict | None = None) -> None:
        config = {} if config is None else config
        unknown_parameters = set(config).difference(CONFIG)
        if unknown_parameters:
            raise ValueError(
                "Unknown KNN parameters: " + ", ".join(sorted(unknown_parameters))
            )
        self.config = {**CONFIG, **config}
        self.model = KNeighborsClassifier(**self.config)

    def save(self, filepath: str | Path) -> None:
        if not hasattr(self.model, "classes_"):
            raise RuntimeError("KNN classifier has not been fitted")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "classifier": "KNNClassifier",
            "config": self.config,
            "model": self.model,
        }
        with filepath.open("wb") as file:
            pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filepath: str | Path) -> Self:
        """Load a fitted classifier from a trusted local file."""
        with Path(filepath).open("rb") as file:
            payload = pickle.load(file)

        if not isinstance(payload, dict):
            raise TypeError("Invalid KNN classifier file")
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported KNN classifier format")
        if payload.get("classifier") != "KNNClassifier":
            raise ValueError("File does not contain a KNN classifier")

        config = payload.get("config")
        model = payload.get("model")
        if not isinstance(config, dict):
            raise TypeError("KNN configuration is missing")
        if not isinstance(model, KNeighborsClassifier) or not hasattr(
            model, "classes_"
        ):
            raise ValueError("Fitted KNN model is missing")

        classifier = cls(config=config)
        classifier.model = model
        return classifier

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

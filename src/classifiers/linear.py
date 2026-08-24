import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from typing_extensions import Self

from classifiers.base import BaseClassifier

CONFIG = {
    "C": 1.0,
    "l1_ratio": 0.0,
    "dual": False,
    "tol": 0.0001,
    "fit_intercept": True,
    "intercept_scaling": 1,
    "class_weight": None,
    "random_state": None,
    "solver": "lbfgs",
    "max_iter": 100,
    "verbose": 0,
    "warm_start": False,
    "n_jobs": None,
}


class LogisticRegressionClassifier(BaseClassifier):

    def __init__(self, config: dict | None = None) -> None:
        config = {} if config is None else config
        unknown_parameters = set(config).difference(CONFIG)
        if unknown_parameters:
            raise ValueError(
                "Unknown logistic regression parameters: "
                + ", ".join(sorted(unknown_parameters))
            )
        self.config = {**CONFIG, **config}
        self.model = LogisticRegression(**self.config)

    def save(self, filepath: str | Path) -> None:
        if not hasattr(self.model, "classes_"):
            raise RuntimeError("Logistic regression classifier has not been fitted")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "classifier": "LogisticRegressionClassifier",
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
            raise TypeError("Invalid logistic regression classifier file")
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported logistic regression classifier format")
        if payload.get("classifier") != "LogisticRegressionClassifier":
            raise ValueError("File does not contain a logistic regression classifier")

        config = payload.get("config")
        model = payload.get("model")
        if not isinstance(config, dict):
            raise TypeError("Logistic regression configuration is missing")
        if not isinstance(model, LogisticRegression) or not hasattr(model, "classes_"):
            raise ValueError("Fitted logistic regression model is missing")

        classifier = cls(config=config)
        classifier.model = model
        return classifier

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

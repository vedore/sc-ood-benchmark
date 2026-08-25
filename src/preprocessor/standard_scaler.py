import pickle
from pathlib import Path

from sklearn.preprocessing import StandardScaler as _StandardScaler
from typing_extensions import Self


class StandardScaler:
    """Standardize features to zero mean and unit variance.

    Thin wrapper around sklearn's StandardScaler, following the same
    fit/transform/save/load shape as the classifiers and representations
    in this codebase.
    """

    def __init__(self) -> None:
        self.model = _StandardScaler()

    def fit(self, X) -> Self:
        self.model.fit(X)
        return self

    def transform(self, X):
        return self.model.transform(X)

    def fit_transform(self, X):
        return self.model.fit_transform(X)

    def save(self, filepath: str | Path) -> None:
        if not hasattr(self.model, "mean_"):
            raise RuntimeError("StandardScaler has not been fitted")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "scaler": "StandardScaler",
            "model": self.model,
        }
        with filepath.open("wb") as file:
            pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filepath: str | Path) -> Self:
        """Load a fitted scaler from a trusted local file."""
        with Path(filepath).open("rb") as file:
            payload = pickle.load(file)

        if not isinstance(payload, dict):
            raise TypeError("Invalid StandardScaler file")
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported StandardScaler format")
        if payload.get("scaler") != "StandardScaler":
            raise ValueError("File does not contain a StandardScaler")

        model = payload.get("model")
        if not isinstance(model, _StandardScaler) or not hasattr(model, "mean_"):
            raise ValueError("Fitted StandardScaler model is missing")

        scaler = cls()
        scaler.model = model
        return scaler

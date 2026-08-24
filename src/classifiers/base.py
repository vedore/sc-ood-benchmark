from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from typing_extensions import Self


class BaseClassifier(ABC):
    @abstractmethod
    def save(self, filepath: str | Path) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, filepath: str | Path) -> Self:
        pass

    @abstractmethod
    def fit(self, X: Any, y: Any) -> Self:
        pass

    @abstractmethod
    def predict(self, X: Any) -> Any:
        pass

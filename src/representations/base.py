from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from typing_extensions import Self


class BaseRepresentation(ABC):
    @abstractmethod
    def fit(self, train_data: Any) -> Self:
        pass

    @abstractmethod
    def transform(self, data: Any) -> Any:
        pass

    @abstractmethod
    def save(self, path: str | Path) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> Self:
        pass

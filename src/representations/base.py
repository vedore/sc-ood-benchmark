from abc import ABC, abstractmethod
from typing import Any

from typing_extensions import Self


class BaseRepresentation(ABC):
    @abstractmethod
    def fit(self, train_data: Any) -> Self:
        pass

    @abstractmethod
    def transform(self, data: Any) -> Any:
        pass

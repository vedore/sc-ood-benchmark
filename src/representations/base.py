from abc import ABC, abstractmethod


class BaseRepresentation(ABC):
    @abstractmethod
    def fit(self, train_data) -> None:
        pass

    @abstractmethod
    def transform(self, data):
        pass

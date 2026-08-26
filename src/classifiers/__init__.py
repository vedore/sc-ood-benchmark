from classifiers.base import BaseClassifier
from classifiers.knn import KNNClassifier
from classifiers.linear import LogisticRegressionClassifier

CLASSIFIERS: dict[str, type[BaseClassifier]] = {
    "logistic_regression": LogisticRegressionClassifier,
    "knn": KNNClassifier,
}


def build_classifier(classifier_type: str, config: dict | None = None) -> BaseClassifier:
    try:
        classifier_cls = CLASSIFIERS[classifier_type]
    except KeyError:
        raise ValueError(
            f"Unknown classifier {classifier_type!r}; "
            f"choose one of {sorted(CLASSIFIERS)}"
        ) from None
    return classifier_cls(config)

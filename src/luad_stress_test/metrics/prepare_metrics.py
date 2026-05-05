from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassAccuracy,
    MulticlassCohenKappa,
)


def prepare_metrics(num_classes: int) -> MetricCollection:
    metrics: dict[str, Metric | MetricCollection] = {
        "f1": MulticlassF1Score(num_classes),
        "precision": MulticlassPrecision(num_classes),
        "recall": MulticlassRecall(num_classes),
        "accuracy": MulticlassAccuracy(num_classes),
        "kappa": MulticlassCohenKappa(num_classes),
    }

    collection = MetricCollection(metrics)

    return collection
